package routing

import (
	"testing"

	"edgesched/scheduler/internal/sysmonitor"
)

func TestLeastQueuePolicy_PicksLowestQueueDepth(t *testing.T) {
	policy := &LeastQueuePolicy{}
	engines := []EngineState{
		{Name: "a", QueueDepth: 5},
		{Name: "b", QueueDepth: 1},
		{Name: "c", QueueDepth: 3},
	}

	name, err := policy.SelectEngine(Request{}, sysmonitor.State{}, engines)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if name != "b" {
		t.Errorf("got %q, want %q (lowest queue depth)", name, "b")
	}
}

func TestLeastQueuePolicy_ErrorsOnNoEngines(t *testing.T) {
	policy := &LeastQueuePolicy{}
	_, err := policy.SelectEngine(Request{}, sysmonitor.State{}, nil)
	if err == nil {
		t.Fatal("expected an error for an empty engine list, got nil")
	}
}

// This is the test that would have caught the map-iteration-order bug:
// with two engines genuinely tied, repeated calls must alternate between
// them in a deterministic, sorted-name-based round-robin -- not favor one
// arbitrarily based on the order the caller's slice happened to be built
// in (e.g. from ranging over a map).
func TestLeastQueuePolicy_RoundRobinsFairlyAmongTies(t *testing.T) {
	policy := &LeastQueuePolicy{}
	engines := []EngineState{
		{Name: "yolo26m_fp16", QueueDepth: 0},
		{Name: "yolo26n_int8", QueueDepth: 0},
	}

	counts := map[string]int{}
	const n = 100
	for i := 0; i < n; i++ {
		name, err := policy.SelectEngine(Request{}, sysmonitor.State{}, engines)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		counts[name]++
	}

	if counts["yolo26n_int8"] != n/2 || counts["yolo26m_fp16"] != n/2 {
		t.Errorf("expected an exact 50/50 split over %d calls, got %+v", n, counts)
	}
}

func TestLeastQueuePolicy_RoundRobinIsDeterministicSequence(t *testing.T) {
	// Sorted alphabetically: a, b, c. With all three tied, the sequence
	// should cycle a, b, c, a, b, c, ... regardless of the input slice's
	// order.
	policy := &LeastQueuePolicy{}
	engines := []EngineState{
		{Name: "c", QueueDepth: 0},
		{Name: "a", QueueDepth: 0},
		{Name: "b", QueueDepth: 0},
	}

	want := []string{"a", "b", "c", "a", "b", "c"}
	for i, w := range want {
		got, err := policy.SelectEngine(Request{}, sysmonitor.State{}, engines)
		if err != nil {
			t.Fatalf("call %d: unexpected error: %v", i, err)
		}
		if got != w {
			t.Errorf("call %d: got %q, want %q", i, got, w)
		}
	}
}

func TestLeastQueuePolicy_NoRoundRobinWhenNotTied(t *testing.T) {
	// A clear winner should be picked every time, not cycled through.
	policy := &LeastQueuePolicy{}
	engines := []EngineState{
		{Name: "a", QueueDepth: 5},
		{Name: "b", QueueDepth: 1}, // strictly lowest, never tied
	}

	for i := 0; i < 10; i++ {
		name, err := policy.SelectEngine(Request{}, sysmonitor.State{}, engines)
		if err != nil {
			t.Fatalf("call %d: unexpected error: %v", i, err)
		}
		if name != "b" {
			t.Errorf("call %d: got %q, want %q (should never vary when not tied)", i, name, "b")
		}
	}
}

func TestThermalAwarePolicy_BehavesLikeLeastQueueWhenCool(t *testing.T) {
	policy := &ThermalAwarePolicy{
		Tiers:          []string{"cheap", "expensive"},
		TempThresholdC: 60,
		CheapTierCount: 1,
	}
	engines := []EngineState{
		{Name: "cheap", QueueDepth: 3},
		{Name: "expensive", QueueDepth: 0},
	}
	sys := sysmonitor.State{Valid: true, MaxTempC: 50} // below threshold

	name, err := policy.SelectEngine(Request{}, sys, engines)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if name != "expensive" {
		t.Errorf("got %q, want %q -- below threshold, should load-balance across everyone",
			name, "expensive")
	}
}

func TestThermalAwarePolicy_RestrictsToCheapTierWhenHot(t *testing.T) {
	policy := &ThermalAwarePolicy{
		Tiers:          []string{"cheap", "expensive"},
		TempThresholdC: 60,
		CheapTierCount: 1,
	}
	engines := []EngineState{
		{Name: "cheap", QueueDepth: 3},
		{Name: "expensive", QueueDepth: 0}, // would win on queue depth alone
	}
	sys := sysmonitor.State{Valid: true, MaxTempC: 65} // above threshold

	name, err := policy.SelectEngine(Request{}, sys, engines)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if name != "cheap" {
		t.Errorf("got %q, want %q -- above threshold, must restrict to the cheap tier "+
			"even though 'expensive' has a lower queue depth", name, "cheap")
	}
}

func TestThermalAwarePolicy_IgnoresInvalidSystemState(t *testing.T) {
	policy := &ThermalAwarePolicy{
		Tiers:          []string{"cheap", "expensive"},
		TempThresholdC: 60,
		CheapTierCount: 1,
	}
	engines := []EngineState{
		{Name: "cheap", QueueDepth: 3},
		{Name: "expensive", QueueDepth: 0},
	}
	sys := sysmonitor.State{Valid: false, MaxTempC: 99} // would be "hot" if trusted

	name, err := policy.SelectEngine(Request{}, sys, engines)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if name != "expensive" {
		t.Errorf("got %q, want %q -- an invalid State must not trigger thermal restriction",
			name, "expensive")
	}
}

func TestThermalAwarePolicy_FallsBackWhenCheapTierNotRegistered(t *testing.T) {
	policy := &ThermalAwarePolicy{
		Tiers:          []string{"cheap", "expensive"},
		TempThresholdC: 60,
		CheapTierCount: 1,
	}
	engines := []EngineState{
		{Name: "expensive", QueueDepth: 0}, // "cheap" is not in this list
	}
	sys := sysmonitor.State{Valid: true, MaxTempC: 65}

	name, err := policy.SelectEngine(Request{}, sys, engines)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if name != "expensive" {
		t.Errorf("got %q, want %q -- should fall back when the cheap tier is unavailable",
			name, "expensive")
	}
}

func TestThermalAwarePolicy_CheapTierCountClampedToValidRange(t *testing.T) {
	policy := &ThermalAwarePolicy{
		Tiers:          []string{"cheap", "mid", "expensive"},
		TempThresholdC: 60,
		CheapTierCount: 0,
	}
	engines := []EngineState{
		{Name: "cheap", QueueDepth: 0},
		{Name: "mid", QueueDepth: 0},
		{Name: "expensive", QueueDepth: 0},
	}
	sys := sysmonitor.State{Valid: true, MaxTempC: 65}

	name, err := policy.SelectEngine(Request{}, sys, engines)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if name != "cheap" {
		t.Errorf("got %q, want %q -- CheapTierCount=0 should clamp to 1 (just the cheapest tier)",
			name, "cheap")
	}
}

func TestThermalAwarePolicy_ErrorsOnNoEngines(t *testing.T) {
	policy := &ThermalAwarePolicy{Tiers: []string{"cheap"}, TempThresholdC: 60, CheapTierCount: 1}
	_, err := policy.SelectEngine(Request{}, sysmonitor.State{}, nil)
	if err == nil {
		t.Fatal("expected an error for an empty engine list, got nil")
	}
}
