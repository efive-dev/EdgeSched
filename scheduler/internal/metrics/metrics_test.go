package metrics

import (
	"testing"

	"edgesched/scheduler/internal/sysmonitor"
	"edgesched/scheduler/internal/workerpool"

	"github.com/prometheus/client_golang/prometheus"
)

func TestRegistry_Collect_SkipsSystemGaugesWhenInvalid(t *testing.T) {
	pools := map[string]*workerpool.Pool{} // empty -- avoids needing a real Predictor
	monitor := sysmonitor.New() // fresh: Valid=false until first poll
	r := New(pools, monitor)
	ch := make(chan prometheus.Metric, 10)
	r.Collect(ch)
	close(ch)
	count := 0
	for range ch {
		count++
	}
	// With no engines registered and an invalid monitor state, only
	// edgesched_system_monitor_valid (=0) should be emitted
	if count != 1 {
		t.Errorf("got %d metrics emitted, want 1 (just monitor_valid=0, "+
			"no system gauges before the monitor's first valid reading)", count)
	}
}
