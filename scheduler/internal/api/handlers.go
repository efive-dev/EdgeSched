// Package api implements the scheduler's client facing HTTP surface.
package api

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sort"
	"strconv"
	"sync"
	"time"

	"edgesched/scheduler/internal/engineclient"
	"edgesched/scheduler/internal/healthcheck"
	"edgesched/scheduler/internal/metrics"
	"edgesched/scheduler/internal/routing"
	"edgesched/scheduler/internal/sysmonitor"
	"edgesched/scheduler/internal/workerpool"
)

const defaultLatencyBudget = 10 * time.Second
const maxImageBytes = 20 << 20 // 20MB

// ServerConfig groups everything NewServer needs
type ServerConfig struct {
	Clients          map[string]*engineclient.Client
	Pools            map[string]*workerpool.Pool
	Monitor          *sysmonitor.Monitor
	HealthMonitor    *healthcheck.Monitor
	Policy           routing.Policy
	PolicyName       string
	MaxLatencyBudget time.Duration
	Metrics          *metrics.Registry
}

type Server struct {
	clients          map[string]*engineclient.Client
	pools            map[string]*workerpool.Pool
	monitor          *sysmonitor.Monitor
	healthMonitor    *healthcheck.Monitor
	policy           routing.Policy
	policyName       string
	metrics          *metrics.Registry
	maxLatencyBudget time.Duration
	// lastResult caches the most recently successful prediction
	// image + detections
	lastResultMu sync.RWMutex
	lastResult   *lastResultJSON
}

func NewServer(cfg ServerConfig) *Server {
	return &Server{
		clients:          cfg.Clients,
		pools:            cfg.Pools,
		monitor:          cfg.Monitor,
		healthMonitor:    cfg.HealthMonitor,
		policy:           cfg.Policy,
		policyName:       cfg.PolicyName,
		metrics:          cfg.Metrics,
		maxLatencyBudget: cfg.MaxLatencyBudget,
	}
}

type detectionJSON struct {
	X1         float32 `json:"x1"`
	Y1         float32 `json:"y1"`
	X2         float32 `json:"x2"`
	Y2         float32 `json:"y2"`
	Confidence float32 `json:"confidence"`
	ClassID    int32   `json:"class_id"`
	ClassName  string  `json:"class_name"`
}

type predictResponseJSON struct {
	Engine          string          `json:"engine"`
	Detections      []detectionJSON `json:"detections"`
	PreprocessMs    float32         `json:"preprocess_ms"`
	InferenceMs     float32         `json:"inference_ms"`
	PostprocessMs   float32         `json:"postprocess_ms"`
	QueueDepth      int             `json:"queue_depth_at_submit"`
	AutoRouted      bool            `json:"auto_routed"`
	LatencyBudgetMs int64           `json:"latency_budget_ms"`
}

type engineStatusJSON struct {
	Name       string `json:"name"`
	QueueDepth int    `json:"queue_depth"`
	Healthy    bool   `json:"healthy"`
}

type systemStatusJSON struct {
	Valid      bool    `json:"valid"`
	MaxTempC   float64 `json:"max_temp_c"`
	PowerMW    float64 `json:"power_mw"`
	GPUUtilPct float64 `json:"gpu_util_pct"`
	RAMUsedMB  int     `json:"ram_used_mb"`
	RAMTotalMB int     `json:"ram_total_mb"`
	PowerMode  string  `json:"power_mode"`
}

type statusJSON struct {
	Policy       string             `json:"policy"`
	Engines      []engineStatusJSON `json:"engines"`
	System       systemStatusJSON   `json:"system"`
	LastResultAt int64              `json:"last_result_at"`
}

type lastResultJSON struct {
	ImageBase64   string          `json:"image_base64"`
	Engine        string          `json:"engine"`
	AutoRouted    bool            `json:"auto_routed"`
	PreprocessMs  float32         `json:"preprocess_ms"`
	InferenceMs   float32         `json:"inference_ms"`
	PostprocessMs float32         `json:"postprocess_ms"`
	Detections    []detectionJSON `json:"detections"`
	TimestampMs   int64           `json:"timestamp_ms"`
}

// parseLatencyBudget reads latency_budget_ms from the query string
func parseLatencyBudget(r *http.Request, maxBudget time.Duration) (time.Duration, error) {
	raw := r.URL.Query().Get("latency_budget_ms")
	if raw == "" {
		return defaultLatencyBudget, nil
	}
	ms, err := strconv.Atoi(raw)
	if err != nil || ms <= 0 {
		return 0, fmt.Errorf("invalid latency_budget_ms: %q (must be a positive integer)", raw)
	}
	budget := time.Duration(ms) * time.Millisecond
	if budget > maxBudget {
		budget = maxBudget
	}
	return budget, nil
}

// engineStates snapshots current queue depth across every configured
// engine, for the routing policy to choose from
func (s *Server) engineStates() []routing.EngineState {
	states := make([]routing.EngineState, 0, len(s.pools))
	for name, pool := range s.pools {
		states = append(states, routing.EngineState{
			Name:       name,
			QueueDepth: pool.QueueDepth(),
			Healthy:    s.healthMonitor.Healthy(name),
		})
	}
	return states
}

func boolLabel(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

// HandlePredict handles POST /predict, optionally with ?engine=<name>
// (manual override) and/or ?latency_budget_ms=<n>.
func (s *Server) HandlePredict(w http.ResponseWriter, r *http.Request) {
	requestStart := time.Now()
	status := "unknown"
	engineName := ""
	defer func() {
		if engineName != "" {
			s.metrics.RequestsTotal.WithLabelValues(engineName, status).Inc()
		}
	}()
	budget, err := parseLatencyBudget(r, s.maxLatencyBudget)
	if err != nil {
		status = "bad_request"
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), budget)
	defer cancel()
	engineName = r.URL.Query().Get("engine")
	autoRouted := false
	if engineName == "" {
		autoRouted = true
		sysState := s.monitor.Current()
		routingReq := routing.Request{LatencyBudget: budget}
		selected, err := s.policy.SelectEngine(routingReq, sysState, s.engineStates())
		if err != nil {
			status = "routing_failed"
			slog.Error("routing failed", "error", err)
			http.Error(w, "routing failed: "+err.Error(), http.StatusServiceUnavailable)
			return
		}
		engineName = selected
		slog.Info("routing decision",
			"engine", engineName,
			"auto_routed", true,
			"system_temp_c", sysState.MaxTempC,
			"system_state_valid", sysState.Valid,
		)
	} else if !s.healthMonitor.Healthy(engineName) {
		status = "engine_unhealthy"
		slog.Warn("rejected request for unhealthy engine", "engine", engineName)
		http.Error(w, fmt.Sprintf("engine %q is currently unhealthy", engineName),
			http.StatusServiceUnavailable)
		return
	}
	pool, ok := s.pools[engineName]
	if !ok {
		status = "bad_request"
		http.Error(w, fmt.Sprintf("unknown engine: %q", engineName), http.StatusBadRequest)
		return
	}
	imageData, err := io.ReadAll(io.LimitReader(r.Body, maxImageBytes))
	if err != nil {
		status = "bad_request"
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}
	if len(imageData) == 0 {
		status = "bad_request"
		http.Error(w, "empty request body -- expected raw image bytes", http.StatusBadRequest)
		return
	}
	queueDepthAtSubmit := pool.QueueDepth()
	resultCh := make(chan workerpool.Result, 1)
	if !pool.Submit(workerpool.Request{Ctx: ctx, ImageData: imageData, Result: resultCh}) {
		status = "rejected"
		s.metrics.AdmissionRejections.WithLabelValues(engineName).Inc()
		slog.Warn("admission rejected: queue full",
			"engine", engineName, "queue_depth", queueDepthAtSubmit)
		w.Header().Set("Retry-After", "1")
		http.Error(w, fmt.Sprintf("engine %q is overloaded (queue full), try again shortly", engineName),
			http.StatusServiceUnavailable)
		return
	}
	select {
	case result := <-resultCh:
		if result.Err != nil {
			status = "error"
			slog.Error("predict failed", "engine", engineName, "error", result.Err)
			http.Error(w, "inference failed: "+result.Err.Error(), http.StatusInternalServerError)
			return
		}
		status = "success"
		resp := result.Response
		s.metrics.RequestDuration.WithLabelValues(engineName, "preprocess").
			Observe(float64(resp.GetPreprocessMs()) / 1000)
		s.metrics.RequestDuration.WithLabelValues(engineName, "inference").
			Observe(float64(resp.GetInferenceMs()) / 1000)
		s.metrics.RequestDuration.WithLabelValues(engineName, "postprocess").
			Observe(float64(resp.GetPostprocessMs()) / 1000)
		s.metrics.RequestDuration.WithLabelValues(engineName, "total").
			Observe(time.Since(requestStart).Seconds())
		out := predictResponseJSON{
			Engine:          engineName,
			PreprocessMs:    resp.GetPreprocessMs(),
			InferenceMs:     resp.GetInferenceMs(),
			PostprocessMs:   resp.GetPostprocessMs(),
			QueueDepth:      queueDepthAtSubmit,
			AutoRouted:      autoRouted,
			LatencyBudgetMs: budget.Milliseconds(),
		}
		for _, d := range resp.GetDetections() {
			out.Detections = append(out.Detections, detectionJSON{
				X1: d.GetX1(), Y1: d.GetY1(), X2: d.GetX2(), Y2: d.GetY2(),
				Confidence: d.GetConfidence(),
				ClassID:    d.GetClassId(),
				ClassName:  d.GetClassName(),
			})
		}
		s.lastResultMu.Lock()
		s.lastResult = &lastResultJSON{
			ImageBase64:   base64.StdEncoding.EncodeToString(imageData),
			Engine:        engineName,
			AutoRouted:    autoRouted,
			PreprocessMs:  resp.GetPreprocessMs(),
			InferenceMs:   resp.GetInferenceMs(),
			PostprocessMs: resp.GetPostprocessMs(),
			Detections:    out.Detections,
			TimestampMs:   time.Now().UnixMilli(),
		}
		s.lastResultMu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(out); err != nil {
			slog.Error("failed to encode response", "error", err)
		}
	case <-ctx.Done():
		status = "timeout"
		http.Error(w, "request timed out waiting for inference (latency budget exceeded)",
			http.StatusGatewayTimeout)
	}
}

// HandleHealth handles GET /health?engine=<name> check one
// engine manually without going through Predict
func (s *Server) HandleHealth(w http.ResponseWriter, r *http.Request) {
	engineName := r.URL.Query().Get("engine")
	if engineName == "" {
		http.Error(w, "missing required query param: engine", http.StatusBadRequest)
		return
	}
	client, ok := s.clients[engineName]
	if !ok {
		http.Error(w, fmt.Sprintf("unknown engine: %q", engineName), http.StatusBadRequest)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
	defer cancel()
	resp, err := client.HealthCheck(ctx)
	if err != nil {
		http.Error(w, "health check failed: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"serving":     resp.GetServing(),
		"engine_name": resp.GetEngineName(),
	})
}

// HandleStatus handles GET /status  a lightweight JSON snapshot of
// live queue/system state, purpose-built for the dashboard (web/) to
// poll
func (s *Server) HandleStatus(w http.ResponseWriter, r *http.Request) {
	rawStates := s.engineStates()
	sort.Slice(rawStates, func(i, j int) bool { return rawStates[i].Name < rawStates[j].Name })
	engines := make([]engineStatusJSON, 0, len(rawStates))
	for _, es := range rawStates {
		engines = append(engines, engineStatusJSON{
			Name: es.Name, QueueDepth: es.QueueDepth, Healthy: es.Healthy,
		})
	}
	state := s.monitor.Current()
	s.lastResultMu.RLock()
	var lastResultAt int64
	if s.lastResult != nil {
		lastResultAt = s.lastResult.TimestampMs
	}
	s.lastResultMu.RUnlock()
	out := statusJSON{
		Policy:       s.policyName,
		Engines:      engines,
		LastResultAt: lastResultAt,
		System: systemStatusJSON{
			Valid:      state.Valid,
			MaxTempC:   state.MaxTempC,
			PowerMW:    state.PowerMW,
			GPUUtilPct: state.GPUUtilPct,
			RAMUsedMB:  state.RAMUsedMB,
			RAMTotalMB: state.RAMTotalMB,
			PowerMode:  state.PowerMode,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

// HandleLastResult handles GET /last-result, returns the most
// recently successfully processed image (base64) plus its detections
func (s *Server) HandleLastResult(w http.ResponseWriter, r *http.Request) {
	s.lastResultMu.RLock()
	result := s.lastResult
	s.lastResultMu.RUnlock()
	if result == nil {
		http.Error(w, "no successful predictions yet", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}
