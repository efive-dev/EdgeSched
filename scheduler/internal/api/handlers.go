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
)

type Server struct {
	engines map[string]*engineclient.Client
}

func NewServer(engines map[string]*engineclient.Client) *Server {
	return &Server{engines: engines}
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
	client, ok := s.engines[engineName]
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
	// TODO: implement actual scheduler behaviour
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()
	resp, err := client.Predict(ctx, imageData)
	if err != nil {
		log.Printf("predict failed on engine %q: %v", engineName, err)
		http.Error(w, "inference failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	out := predictResponseJSON{
		Engine:        engineName,
		PreprocessMs:  resp.GetPreprocessMs(),
		InferenceMs:   resp.GetInferenceMs(),
		PostprocessMs: resp.GetPostprocessMs(),
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
}

// HandleHealth handles GET /health?engine=<name> check one
// engine manually without going through Predict
func (s *Server) HandleHealth(w http.ResponseWriter, r *http.Request) {
	engineName := r.URL.Query().Get("engine")
	if engineName == "" {
		http.Error(w, "missing required query param: engine", http.StatusBadRequest)
		return
	}
	client, ok := s.engines[engineName]
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
