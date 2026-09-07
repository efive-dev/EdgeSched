package api

import (
	"net/http/httptest"
	"testing"
	"time"
)

func TestParseLatencyBudget_DefaultWhenMissing(t *testing.T) {
	r := httptest.NewRequest("POST", "/predict", nil)
	got, err := parseLatencyBudget(r, 30*time.Second)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != defaultLatencyBudget {
		t.Errorf("got %v, want default %v", got, defaultLatencyBudget)
	}
}

func TestParseLatencyBudget_ParsesValidValue(t *testing.T) {
	r := httptest.NewRequest("POST", "/predict?latency_budget_ms=500", nil)
	got, err := parseLatencyBudget(r, 30*time.Second)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 500*time.Millisecond {
		t.Errorf("got %v, want 500ms", got)
	}
}

func TestParseLatencyBudget_ClampsToMax(t *testing.T) {
	r := httptest.NewRequest("POST", "/predict?latency_budget_ms=999999", nil)

	got, err := parseLatencyBudget(r, 2*time.Second)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 2*time.Second {
		t.Errorf("got %v, want clamped to max 2s", got)
	}
}

func TestParseLatencyBudget_RejectsInvalidValues(t *testing.T) {
	cases := []string{"abc", "0", "-100"}
	for _, c := range cases {
		r := httptest.NewRequest("POST", "/predict?latency_budget_ms="+c, nil)
		if _, err := parseLatencyBudget(r, 30*time.Second); err == nil {
			t.Errorf("expected error for latency_budget_ms=%q, got nil", c)
		}
	}
}
