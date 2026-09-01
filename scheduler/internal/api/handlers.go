// Package api implements the scheduler's client facing HTTP surface.
package api

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"edgesched/scheduler/internal/engineclient"
	"edgesched/scheduler/internal/workerpool"
)

type Server struct {
	// clients is used for HealthCheck, which bypasses the worker pool
	// there's no need to bound concurrency or queue a health check.
	clients map[string]*engineclient.Client
	// pools is used for Predict, every inference request goes through
	// its engine's bounded pool, not directly through the client.
	pools map[string]*workerpool.Pool
}

func NewServer(clients map[string]*engineclient.Client, pools map[string]*workerpool.Pool) *Server {
	return &Server{clients: clients, pools: pools}
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
	Engine        string          `json:"engine"`
	Detections    []detectionJSON `json:"detections"`
	PreprocessMs  float32         `json:"preprocess_ms"`
	InferenceMs   float32         `json:"inference_ms"`
	PostprocessMs float32         `json:"postprocess_ms"`
	QueueDepth    int             `json:"queue_depth_at_submit"`
}

const maxImageBytes = 20 << 20 // 20MB

// HandlePredict handles POST /predict?engine=<name>, with the raw image
// bytes as the request body
func (s *Server) HandlePredict(w http.ResponseWriter, r *http.Request) {
	engineName := r.URL.Query().Get("engine")
	if engineName == "" {
		http.Error(w, "missing required query param: engine", http.StatusBadRequest)
		return
	}
	pool, ok := s.pools[engineName]
	if !ok {
		http.Error(w, fmt.Sprintf("unknown engine: %q", engineName), http.StatusBadRequest)
		return
	}
	imageData, err := io.ReadAll(io.LimitReader(r.Body, maxImageBytes))
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}
	if len(imageData) == 0 {
		http.Error(w, "empty request body -- expected raw image bytes", http.StatusBadRequest)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()
	queueDepthAtSubmit := pool.QueueDepth()
	resultCh := make(chan workerpool.Result, 1)
	if !pool.Submit(workerpool.Request{Ctx: ctx, ImageData: imageData, Result: resultCh}) {
		// Admission control: the queue is already full. Reject now rather
		// than let this request wait behind an already saturated engine
		// with no bound on how long that wait could be
		w.Header().Set("Retry-After", "1")
		http.Error(w, fmt.Sprintf("engine %q is overloaded (queue full), try again shortly", engineName),
			http.StatusServiceUnavailable)
		return
	}
	select {
	case result := <-resultCh:
		if result.Err != nil {
			log.Printf("predict failed on engine %q: %v", engineName, result.Err)
			http.Error(w, "inference failed: "+result.Err.Error(), http.StatusInternalServerError)
			return
		}
		resp := result.Response
		out := predictResponseJSON{
			Engine:        engineName,
			PreprocessMs:  resp.GetPreprocessMs(),
			InferenceMs:   resp.GetInferenceMs(),
			PostprocessMs: resp.GetPostprocessMs(),
			QueueDepth:    queueDepthAtSubmit,
		}
		for _, d := range resp.GetDetections() {
			out.Detections = append(out.Detections, detectionJSON{
				X1: d.GetX1(), Y1: d.GetY1(), X2: d.GetX2(), Y2: d.GetY2(),
				Confidence: d.GetConfidence(),
				ClassID:    d.GetClassId(),
				ClassName:  d.GetClassName(),
			})
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(out); err != nil {
			log.Printf("failed to encode response: %v", err)
		}
	case <-ctx.Done():
		// Either the request timed out waiting in queue, or the client
		// disconnected
		http.Error(w, "request timed out waiting for inference", http.StatusGatewayTimeout)
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
