package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestBuildPredictURL_NoParams(t *testing.T) {
	got := buildPredictURL("http://localhost:8080", "", 0)
	want := "http://localhost:8080/predict"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestBuildPredictURL_WithEngine(t *testing.T) {
	got := buildPredictURL("http://localhost:8080", "yolo26n_int8", 0)
	want := "http://localhost:8080/predict?engine=yolo26n_int8"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestBuildPredictURL_WithBudget(t *testing.T) {
	got := buildPredictURL("http://localhost:8080", "", 500)
	want := "http://localhost:8080/predict?latency_budget_ms=500"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestBuildPredictURL_WithBothParams(t *testing.T) {
	got := buildPredictURL("http://localhost:8080", "yolo26m_fp16", 500)
	want := "http://localhost:8080/predict?engine=yolo26m_fp16&latency_budget_ms=500"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestBuildPredictURL_TrimsTrailingSlash(t *testing.T) {
	got := buildPredictURL("http://localhost:8080/", "", 0)
	want := "http://localhost:8080/predict"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestPercentile_KnownValues(t *testing.T) {
	sorted := []float64{10, 20, 30, 40, 50}
	if got := percentile(sorted, 0); got != 10 {
		t.Errorf("p0: got %v, want 10", got)
	}
	if got := percentile(sorted, 100); got != 50 {
		t.Errorf("p100: got %v, want 50", got)
	}
}

func TestPercentile_EmptySliceReturnsZero(t *testing.T) {
	if got := percentile(nil, 50); got != 0 {
		t.Errorf("got %v, want 0 for empty input", got)
	}
}

func TestFindImages_FiltersByExtensionCaseInsensitively(t *testing.T) {
	dir := t.TempDir()
	files := []string{"a.jpg", "b.PNG", "c.txt", "d.jpeg", "notes.md"}
	for _, f := range files {
		if err := os.WriteFile(filepath.Join(dir, f), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	got, err := findImages(dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 3 {
		t.Errorf("got %d images, want 3 (a.jpg, b.PNG, d.jpeg): %v", len(got), got)
	}
}

func TestFindImages_EmptyDirReturnsEmpty(t *testing.T) {
	dir := t.TempDir()
	got, err := findImages(dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("got %d, want 0", len(got))
	}
}
