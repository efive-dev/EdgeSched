package healthcheck

import (
	"context"
	"errors"
	"testing"
	"time"

	"edgesched/scheduler/internal/engineclient"
)

type fakeChecker struct {
	serving bool
	err     error
	calls   int
}

func (f *fakeChecker) HealthCheck(ctx context.Context) (bool, error) {
	f.calls++
	return f.serving, f.err
}

func newTestMonitor(name string, checker Checker) *Monitor {
	return &Monitor{
		statuses: map[string]bool{name: true},
		checkers: map[string]Checker{name: checker},
		interval: time.Hour, // irrelevant
		timeout:  time.Second,
	}
}

func TestMonitor_HealthyAfterSuccessfulCheck(t *testing.T) {
	fc := &fakeChecker{serving: true}
	m := newTestMonitor("engine-a", fc)
	m.check(context.Background(), "engine-a", fc)
	if !m.Healthy("engine-a") {
		t.Error("expected engine-a to be healthy after a successful check")
	}
	if fc.calls != 1 {
		t.Errorf("expected 1 call to HealthCheck, got %d", fc.calls)
	}
}

func TestMonitor_UnhealthyAfterFailedCheck(t *testing.T) {
	fc := &fakeChecker{err: errors.New("connection refused")}
	m := newTestMonitor("engine-a", fc)
	m.check(context.Background(), "engine-a", fc)
	if m.Healthy("engine-a") {
		t.Error("expected engine-a to be unhealthy after a failed check")
	}
}

func TestMonitor_UnhealthyWhenServingFalse(t *testing.T) {
	fc := &fakeChecker{serving: false, err: nil}
	m := newTestMonitor("engine-a", fc)
	m.check(context.Background(), "engine-a", fc)
	if m.Healthy("engine-a") {
		t.Error("expected engine-a to be unhealthy when serving=false, even with no error")
	}
}

func TestMonitor_RecoversAfterComingBackUp(t *testing.T) {
	fc := &fakeChecker{err: errors.New("down")}
	m := newTestMonitor("engine-a", fc)
	m.check(context.Background(), "engine-a", fc)
	if m.Healthy("engine-a") {
		t.Fatal("expected engine-a to be unhealthy first")
	}
	fc.err = nil
	fc.serving = true
	m.check(context.Background(), "engine-a", fc)
	if !m.Healthy("engine-a") {
		t.Error("expected engine-a to be healthy again after recovering")
	}
}

func TestMonitor_UnknownEngineReportsUnhealthy(t *testing.T) {
	m := New(map[string]*engineclient.Client{}, time.Second, time.Second)
	if m.Healthy("does-not-exist") {
		t.Error("expected an unknown engine name to report unhealthy, not panic or default true")
	}
}

func TestMonitor_NewAssumesHealthyBeforeFirstCheck(t *testing.T) {
	m := newTestMonitor("engine-a", &fakeChecker{})
	if !m.Healthy("engine-a") {
		t.Error("expected engine-a to start assumed-healthy before any check has run")
	}
}

func TestMonitor_SnapshotReturnsIndependentCopy(t *testing.T) {
	fc := &fakeChecker{serving: true}
	m := newTestMonitor("engine-a", fc)
	m.check(context.Background(), "engine-a", fc)
	snap := m.Snapshot()
	snap["engine-a"] = false // mutate the returned copy
	if !m.Healthy("engine-a") {
		t.Error("mutating a Snapshot() result should not affect the Monitor's internal state")
	}
}
