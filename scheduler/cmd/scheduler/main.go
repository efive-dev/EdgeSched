// Command scheduler is the Go control plane's entrypoint
// connects to a fixed,  set of inference engines and exposes an HTTP API
// where the caller names which engine to use.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"edgesched/scheduler/internal/api"
	"edgesched/scheduler/internal/engineclient"
	"edgesched/scheduler/internal/metrics"
	"edgesched/scheduler/internal/routing"
	"edgesched/scheduler/internal/sysmonitor"
	"edgesched/scheduler/internal/workerpool"
	"edgesched/scheduler/web"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// parseEngines parses "name1=addr1,name2=addr2" into a map.
func parseEngines(spec string) (map[string]string, error) {
	result := make(map[string]string)
	for _, pair := range strings.Split(spec, ",") {
		pair = strings.TrimSpace(pair)
		if pair == "" {
			continue
		}
		parts := strings.SplitN(pair, "=", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid engine spec %q, expected name=address", pair)
		}
		result[parts[0]] = parts[1]
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("no engines specified")
	}
	return result, nil
}

func buildPolicy(policyName, cheapTierSpec string, tempThreshold float64, cheapTierCount int) (routing.Policy, error) {
	switch policyName {
	case "least-queue":
		return &routing.LeastQueuePolicy{}, nil
	case "thermal-aware":
		if cheapTierSpec == "" {
			return nil, fmt.Errorf("-routing-policy=thermal-aware requires -cheap-tier")
		}
		tiers := strings.Split(cheapTierSpec, ",")
		for i := range tiers {
			tiers[i] = strings.TrimSpace(tiers[i])
		}
		return &routing.ThermalAwarePolicy{
			Tiers:          tiers,
			TempThresholdC: tempThreshold,
			CheapTierCount: cheapTierCount,
		}, nil
	default:
		return nil, fmt.Errorf("unknown -routing-policy: %q (want %q or %q)",
			policyName, "least-queue", "thermal-aware")
	}
}

func main() {
	enginesFlag := flag.String("engines", "",
		`comma-separated name=address pairs, e.g. "yolo26n_int8=localhost:50051,yolo26m_fp16=localhost:50052"`)
	listenAddr := flag.String("listen", ":8080", "HTTP listen address")
	queueSize := flag.Int("queue-size", 20,
		"max requests allowed to wait per engine before Submit starts rejecting (503)")
	concurrency := flag.Int("concurrency", 4,
		"max concurrent Predict calls per engine (workers per pool)")

	policyName := flag.String("routing-policy", "least-queue",
		`routing policy when "engine" query param is omitted: "least-queue" or "thermal-aware"`)
	cheapTierFlag := flag.String("cheap-tier", "",
		"comma-separated engine names, cheapest first (required for -routing-policy=thermal-aware)")
	tempThreshold := flag.Float64("temp-threshold-c", 60.0,
		"MaxTempC at/above which thermal-aware policy restricts to the cheap tier")
	cheapTierCount := flag.Int("cheap-tier-count", 1,
		"how many of the leading cheap-tier engines stay eligible when hot")
	maxLatencyBudgetMs := flag.Int64("max-latency-budget-ms", 30000,
		"upper bound on any client-requested latency_budget_ms")
	flag.Parse()
	if *enginesFlag == "" {
		log.Fatal("must specify -engines")
	}
	engineAddrs, err := parseEngines(*enginesFlag)
	if err != nil {
		log.Fatalf("bad -engines flag: %v", err)
	}
	policy, err := buildPolicy(*policyName, *cheapTierFlag, *tempThreshold, *cheapTierCount)
	if err != nil {
		log.Fatalf("bad routing policy configuration: %v", err)
	}
	slog.Info("using routing policy", "policy", *policyName)

	clients := make(map[string]*engineclient.Client)
	pools := make(map[string]*workerpool.Pool)
	for name, addr := range engineAddrs {
		client, err := engineclient.New(name, addr)
		if err != nil {
			log.Fatalf("failed to create client for engine %q: %v", name, err)
		}
		clients[name] = client
		pools[name] = workerpool.New(client, *queueSize, *concurrency)
		slog.Info("registered engine", "name", name, "addr", addr,
			"queue_size", *queueSize, "concurrency", *concurrency)
	}
	monitor := sysmonitor.New()
	go func() {
		if err := monitor.Run(context.Background()); err != nil {
			slog.Error("sysmonitor stopped", "error", err)
		}
	}()
	metricsRegistry := metrics.New(pools, monitor)
	metricsRegistry.MustRegister(prometheus.DefaultRegisterer)
	server := api.NewServer(clients, pools, monitor, policy, *policyName,
		time.Duration(*maxLatencyBudgetMs)*time.Millisecond, metricsRegistry)
	mux := http.NewServeMux()
	mux.HandleFunc("POST /predict", server.HandlePredict)
	mux.HandleFunc("GET /health", server.HandleHealth)
	mux.HandleFunc("GET /status", server.HandleStatus)
	mux.HandleFunc("GET /last-result", server.HandleLastResult)
	mux.Handle("GET /metrics", promhttp.Handler())
	mux.Handle("/", web.Handler())
	slog.Info("scheduler listening", "addr", *listenAddr)
	if err := http.ListenAndServe(*listenAddr, mux); err != nil {
		log.Fatal(err)
	}
}
