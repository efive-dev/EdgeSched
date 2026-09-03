// Package metrics exposes EdgeSched's internal state in Prometheus
// format
package metrics

import (
	"edgesched/scheduler/internal/sysmonitor"
	"edgesched/scheduler/internal/workerpool"

	"github.com/prometheus/client_golang/prometheus"
)

// Registry holds every metric EdgeSched exposes. Construct with New,
// then call MustRegister once at startup
type Registry struct {
	// RequestsTotal counts every /predict outcome
	RequestsTotal *prometheus.CounterVec
	// RequestDuration records per-stage timing, labeled by engine and
	// stage ("preprocess", "inference", "postprocess", "total")
	RequestDuration *prometheus.HistogramVec
	// RoutingDecisions counts which engine was selected, labeled by
	// engine and whether the caller specified it manually
	// (auto_routed="false") or the RoutingPolicy chose it
	// (auto_routed="true")
	RoutingDecisions *prometheus.CounterVec
	// AdmissionRejections counts 503s caused by a full worker pool queue,
	// labeled by engine
	AdmissionRejections *prometheus.CounterVec
	pools   map[string]*workerpool.Pool
	monitor *sysmonitor.Monitor
	queueDepthDesc   *prometheus.Desc
	tempDesc         *prometheus.Desc
	powerDesc        *prometheus.Desc
	gpuUtilDesc      *prometheus.Desc
	ramUsedDesc      *prometheus.Desc
	monitorValidDesc *prometheus.Desc
}

// New builds a Registry, pools and monitor are read (not copied) on
// every scrape via Collect
func New(pools map[string]*workerpool.Pool, monitor *sysmonitor.Monitor) *Registry {
	return &Registry{
		RequestsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "edgesched_requests_total",
			Help: "Total /predict requests, by engine and outcome.",
		}, []string{"engine", "status"}),
		RequestDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name: "edgesched_request_duration_seconds",
			Help: "Per stage request duration, by engine and stage.",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 14),
		}, []string{"engine", "stage"}),
		RoutingDecisions: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "edgesched_routing_decisions_total",
			Help: "Number of times each engine was selected, by engine and whether routing was automatic.",
		}, []string{"engine", "auto_routed"}),
		AdmissionRejections: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "edgesched_admission_rejections_total",
			Help: "Requests rejected with 503 due to a full worker-pool queue, by engine.",
		}, []string{"engine"}),
		pools:   pools,
		monitor: monitor,
		queueDepthDesc: prometheus.NewDesc(
			"edgesched_queue_depth",
			"Current number of requests waiting in an engine's worker pool queue.",
			[]string{"engine"}, nil),
		tempDesc: prometheus.NewDesc(
			"edgesched_system_max_temp_celsius",
			"Highest reported sensor temperature.",
			nil, nil),
		powerDesc: prometheus.NewDesc(
			"edgesched_system_power_milliwatts",
			"Total system power draw (VDD_IN).",
			nil, nil),
		gpuUtilDesc: prometheus.NewDesc(
			"edgesched_system_gpu_util_percent",
			"Instantaneous GPU utilization.",
			nil, nil),
		ramUsedDesc: prometheus.NewDesc(
			"edgesched_system_ram_used_mb",
			"Used system RAM in MB.",
			nil, nil),
		monitorValidDesc: prometheus.NewDesc(
			"edgesched_system_monitor_valid",
			"1 if the system monitor has a valid reading yet, 0 before its first successful poll.",
			nil, nil),
	}
}

// MustRegister registers every counter/histogram plus this Registry
// itself (as a Collector, for the pull-based gauges) with reg
func (r *Registry) MustRegister(reg prometheus.Registerer) {
	reg.MustRegister(r.RequestsTotal, r.RequestDuration, r.RoutingDecisions, r.AdmissionRejections, r)
}

// Describe implements prometheus.Collector
func (r *Registry) Describe(ch chan<- *prometheus.Desc) {
	ch <- r.queueDepthDesc
	ch <- r.tempDesc
	ch <- r.powerDesc
	ch <- r.gpuUtilDesc
	ch <- r.ramUsedDesc
	ch <- r.monitorValidDesc
}

// Collect implements prometheus.Collector
func (r *Registry) Collect(ch chan<- prometheus.Metric) {
	for name, pool := range r.pools {
		ch <- prometheus.MustNewConstMetric(
			r.queueDepthDesc, prometheus.GaugeValue, float64(pool.QueueDepth()), name)
	}
	state := r.monitor.Current()
	validVal := 0.0
	if state.Valid {
		validVal = 1.0
	}
	ch <- prometheus.MustNewConstMetric(r.monitorValidDesc, prometheus.GaugeValue, validVal)
	// Only emit system gauges once the monitor has a real reading
	if state.Valid {
		ch <- prometheus.MustNewConstMetric(r.tempDesc, prometheus.GaugeValue, state.MaxTempC)
		ch <- prometheus.MustNewConstMetric(r.powerDesc, prometheus.GaugeValue, state.PowerMW)
		ch <- prometheus.MustNewConstMetric(r.gpuUtilDesc, prometheus.GaugeValue, state.GPUUtilPct)
		ch <- prometheus.MustNewConstMetric(r.ramUsedDesc, prometheus.GaugeValue, float64(state.RAMUsedMB))
	}
}
