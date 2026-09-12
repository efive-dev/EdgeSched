// Package routing selects which inference engine should handle a
// request, given the request itself, current system state (thermal,
// from sysmonitor), and the current load (queue depth) of every
// available engine
package routing

import (
	"fmt"
	"sort"
	"sync/atomic"
	"time"

	"edgesched/scheduler/internal/sysmonitor"
)

// Request carries per request hints a policy might use
type Request struct {
	LatencyBudget time.Duration
}

// EngineState is one engine's current load, as known to the scheduler
type EngineState struct {
	Name       string
	QueueDepth int
	Healthy    bool
}

// Policy selects which engine should handle a request
type Policy interface {
	SelectEngine(req Request, sys sysmonitor.State, engines []EngineState) (string, error)
}

// filterHealthy returns only the engines currently reporting healthy
func filterHealthy(engines []EngineState) []EngineState {
	out := make([]EngineState, 0, len(engines))
	for _, e := range engines {
		if e.Healthy {
			out = append(out, e)
		}
	}
	return out
}

type tieBreaker struct {
	counter atomic.Uint64
}

func (tb *tieBreaker) pick(engines []EngineState) (string, error) {
	if len(engines) == 0 {
		return "", fmt.Errorf("no engines available")
	}
	sorted := append([]EngineState(nil), engines...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })
	minDepth := sorted[0].QueueDepth
	for _, e := range sorted[1:] {
		if e.QueueDepth < minDepth {
			minDepth = e.QueueDepth
		}
	}
	tied := make([]EngineState, 0, len(sorted))
	for _, e := range sorted {
		if e.QueueDepth == minDepth {
			tied = append(tied, e)
		}
	}
	idx := (tb.counter.Add(1) - 1) % uint64(len(tied))
	return tied[idx].Name, nil
}

// LeastQueuePolicy always picks the engine with the lowest current queue
// depth, ignoring system/thermal state entirely. Ties are round robined
// fairly via tieBreaker
type LeastQueuePolicy struct {
	tb tieBreaker
}

func (p *LeastQueuePolicy) SelectEngine(_ Request, _ sysmonitor.State, engines []EngineState) (string, error) {
	healthy := filterHealthy(engines)
	if len(healthy) == 0 {
		return "", fmt.Errorf("no healthy engines available")
	}
	return p.tb.pick(healthy)
}

// ThermalAwarePolicy load-balances across all engines like
// LeastQueuePolicy under normal conditions, but restricts candidates to
// a configured "cheap tier" once system temperature crosses a threshold
type ThermalAwarePolicy struct {
	// Tiers is ordered cheapest -> most expensive engine name.
	Tiers          []string
	TempThresholdC float64
	CheapTierCount int
	tb             tieBreaker
}

func (p *ThermalAwarePolicy) SelectEngine(_ Request, sys sysmonitor.State, engines []EngineState) (string, error) {
	healthy := filterHealthy(engines)
	if len(healthy) == 0 {
		return "", fmt.Errorf("no healthy engines available")
	}
	candidates := healthy
	if sys.Valid && sys.MaxTempC >= p.TempThresholdC {
		if filtered := filterToCheapTier(healthy, p.Tiers, p.CheapTierCount); len(filtered) > 0 {
			candidates = filtered
		}
	}
	return p.tb.pick(candidates)
}

func filterToCheapTier(engines []EngineState, tiers []string, count int) []EngineState {
	if count < 1 {
		count = 1
	}
	if count > len(tiers) {
		count = len(tiers)
	}
	allowed := make(map[string]bool, count)
	for _, name := range tiers[:count] {
		allowed[name] = true
	}
	filtered := make([]EngineState, 0, len(engines))
	for _, e := range engines {
		if allowed[e.Name] {
			filtered = append(filtered, e)
		}
	}
	return filtered
}
