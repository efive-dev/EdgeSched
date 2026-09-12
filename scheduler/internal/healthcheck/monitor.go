// Package healthcheck polls each engine's HealthCheck RPC independently
// and in the background, tracking per engine up/down status so the
// routing layer can avoid sending traffic to a dead engine instead of
// discovering it only when a request fails or times out.
package healthcheck

import (
	"context"
	"maps"
	"sync"
	"time"

	"edgesched/scheduler/internal/engineclient"
)

// Checker is the subset of engineclient.Client this package depends on,
// defined as an interface
type Checker interface {
	HealthCheck(ctx context.Context) (serving bool, err error)
}

// clientChecker adapts *engineclient.Client's HealthCheck (which returns
// a *pb.HealthCheckResponse) to the simpler Checker interface above.
type clientChecker struct {
	client *engineclient.Client
}

func (c clientChecker) HealthCheck(ctx context.Context) (bool, error) {
	resp, err := c.client.HealthCheck(ctx)
	if err != nil {
		return false, err
	}
	return resp.GetServing(), nil
}

// Monitor tracks the health of a fixed set of engines, polling each one
// independently on its own goroutine.
type Monitor struct {
	mu       sync.RWMutex
	statuses map[string]bool // engine name -> healthy
	checkers map[string]Checker
	interval time.Duration
	timeout  time.Duration
}

// New builds a Monitor for the given clients
func New(clients map[string]*engineclient.Client, interval, timeout time.Duration) *Monitor {
	checkers := make(map[string]Checker, len(clients))
	statuses := make(map[string]bool, len(clients))
	for name, c := range clients {
		checkers[name] = clientChecker{client: c}
		statuses[name] = true
	}
	return &Monitor{
		statuses: statuses,
		checkers: checkers,
		interval: interval,
		timeout:  timeout,
	}
}

// Run starts one polling goroutine per engine and blocks until ctx is
// canceled.
func (m *Monitor) Run(ctx context.Context) {
	var wg sync.WaitGroup
	for name, checker := range m.checkers {
		wg.Add(1)
		go func(name string, checker Checker) {
			defer wg.Done()
			m.pollLoop(ctx, name, checker)
		}(name, checker)
	}
	wg.Wait()
}

func (m *Monitor) pollLoop(ctx context.Context, name string, checker Checker) {
	ticker := time.NewTicker(m.interval)
	defer ticker.Stop()
	m.check(ctx, name, checker) // check immediately, don't wait a full interval first
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.check(ctx, name, checker)
		}
	}
}

func (m *Monitor) check(ctx context.Context, name string, checker Checker) {
	checkCtx, cancel := context.WithTimeout(ctx, m.timeout)
	defer cancel()
	serving, err := checker.HealthCheck(checkCtx)
	healthy := err == nil && serving

	m.mu.Lock()
	m.statuses[name] = healthy
	m.mu.Unlock()
}

func (m *Monitor) Healthy(name string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.statuses[name]
}

func (m *Monitor) Snapshot() map[string]bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make(map[string]bool, len(m.statuses))
	maps.Copy(out, m.statuses)
	return out
}
