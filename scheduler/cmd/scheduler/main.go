// Command scheduler is the Go control plane's entrypoint
// connects to a fixed,  set of inference engines and exposes an HTTP API
// where the caller names which engine to use.
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"strings"

	"edgesched/scheduler/internal/api"
	"edgesched/scheduler/internal/engineclient"
	"edgesched/scheduler/internal/workerpool"
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

func main() {
	enginesFlag := flag.String("engines", "",
		`comma-separated name=address pairs, e.g. "yolo26n_int8=localhost:50051,yolo26m_fp16=localhost:50052"`)
	listenAddr := flag.String("listen", ":8080", "HTTP listen address")
	queueSize := flag.Int("queue-size", 20,
		"max requests allowed to wait per engine before Submit starts rejecting (503)")
	concurrency := flag.Int("concurrency", 4,
		"max concurrent Predict calls per engine (workers per pool)")
	flag.Parse()
	if *enginesFlag == "" {
		log.Fatal("must specify -engines")
	}
	engineAddrs, err := parseEngines(*enginesFlag)
	if err != nil {
		log.Fatalf("bad -engines flag: %v", err)
	}
	clients := make(map[string]*engineclient.Client)
	pools := make(map[string]*workerpool.Pool)
	for name, addr := range engineAddrs {
		client, err := engineclient.New(name, addr)
		if err != nil {
			log.Fatalf("failed to create client for engine %q: %v", name, err)
		}
		clients[name] = client
		pools[name] = workerpool.New(client, *queueSize, *concurrency)
		log.Printf("registered engine %q -> %s (queue=%d, concurrency=%d)",
			name, addr, *queueSize, *concurrency)
	}
	server := api.NewServer(clients, pools)
	mux := http.NewServeMux()
	mux.HandleFunc("POST /predict", server.HandlePredict)
	mux.HandleFunc("GET /health", server.HandleHealth)
	log.Printf("scheduler listening on %s", *listenAddr)
	if err := http.ListenAndServe(*listenAddr, mux); err != nil {
		log.Fatal(err)
	}
}
