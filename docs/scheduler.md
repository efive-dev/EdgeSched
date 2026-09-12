# Scheduler (Go Control Plane)

The Go control plane is the client-facing side of EdgeSched, it connects to one or more C++ inference engines over gRPC (see `proto.md` for the wire contract, `engine.md` for what's on the other end of that connection) and exposes an HTTP API in front of them.

---

## Why Go here specifically
- Goroutines and channels map directly onto the eventual problem this component exists to solve: bounded per engine worker 
  pools, request queuing, backpressure, timeouts;
- Recognizable pattern in infra/platform engineering generally.

---

## Components

| Component | Responsibility |
|---|---|
| **`cmd/scheduler/main.go`** | Process entrypoint. Parses `-engines` (name=address pairs) and `-listen` flags, constructs one `engineclient.Client` per configured engine, starts the HTTP server. |
| **`internal/engineclient.Client`** | Wraps one gRPC connection to one inference engine process. Exposes `Predict(ctx, imageData)` and `HealthCheck(ctx)`. One instance per engine tier. |
| **`internal/api.Server`** | HTTP handlers (`POST /predict`, `GET /health`), each taking an `engine` query parameter and proxying to the matching `Client`. |
| **`internal/sysmonitor.Monitor`** | Polls `tegrastats`/`nvpmodel` in the background, exposes the latest system state (thermal, power, memory, GPU utilization) via a lock free atomic snapshot. |
| **`cmd/sysmonitor_debug`** | Standalone tool to validate `sysmonitor`'s parsing against real device output before trusting it anywhere else. |
| **`internal/routing.Policy`** | Selects which engine handles a request. `LeastQueuePolicy` (pure load balancing) and `ThermalAwarePolicy` (restricts to a cheap tier once hot) implement it. |
| **`internal/metrics.Registry`** | Prometheus metrics: imperative counters/histograms for requests and routing, pull-based gauges for queue depth and system state. |
| **`web.Handler()`** | Serves the embedded dashboard (system state, live charts, predict form, Live Feed panel showing predictions from any source). |
| **`cmd/loadgen`** | CLI: sends every image in a directory to the scheduler concurrently, reports latency/throughput stats, optional CSV export. |
| **`internal/healthcheck.Monitor`** | Polls each engine's `HealthCheck` RPC in the background; both routing policies exclude unhealthy engines automatically. |

---

## Diagrams
Eventually a scheduler will be implemented and this is generally how it will work:

![scheduler control plane](img/schedulerControlPlane.png)
## System Monitor
`internal/sysmonitor` polls Jetson system state in the background and exposes the latest reading via a lock free atomic snapshot (`atomic.Pointer[State]`), so routing decisions can read current state without blocking on or triggering fresh I/O.

### Design

- **`tegrastats`** is run as one long lived subprocess (`--interval 1000`) without needing multiple calls;
- **`nvpmodel -q`** is polled separately, on a slower 5-second cadence, since power mode changes rarely and doesn't need per
  second freshness;

---

## Dashboard and Load Generation 

### Dashboard (`web/`)

A single static HTML/JS page, embedded directly into the compiled
scheduler binary via `go:embed` (`web/embed.go`) — no separate directory
needs to be deployed alongside the binary, and the dashboard can never
drift out of sync with the server it ships inside.

Served at `/` (Go 1.22's `ServeMux` treats a bare `/` pattern as a
subtree match, so it only handles requests not claimed by the more
specific `/predict`, `/health`, `/metrics`, `/status`, `/last-result`
patterns).

**Two purposes, both from the original plan**:
- **Passive monitoring** — live system state (temp, power, GPU%, RAM,
  power mode) and per-engine queue depth, each on a real line chart
  (Chart.js, vendored locally — see below), polling `GET /status` every
  second.
- **Interactive** — a form to send an image directly (with an optional
  engine override or latency budget), rendering the routed result with
  bounding boxes drawn client-side on canvas.

**Charting library**: Chart.js 4.5.1, vendored as a minified UMD build
(`web/static/vendor/chart.umd.min.js`, fetched via `npm pack chart.js@4`
and copied verbatim from the official package — not hand-rolled or
modified) rather than loaded from a CDN. This matters specifically
because the scheduler may be accessed over a private, internet-less
Ethernet link (as it was during development) — a CDN `<script>` tag
would silently fail there.

**Styling**: dark background, high-contrast accent color, sharp borders,
uppercase display text — deliberately chosen, not a default template.
Two elements considered and explicitly removed after review: a
custom vendored font (network restrictions during development prevented
fetching an offline-safe copy, and it wasn't worth the added complexity)
and a scrolling marquee ticker (added, then cut for being unnecessary
visual noise on a monitoring tool where clarity matters more than
flourish). Both are easy to reintroduce later if wanted — the CSS
variables and structure are already there.

### Live Feed: showing predictions from any source

The dashboard's own predict form can render its own results immediately
(it has the image bytes right there in the browser). But a request sent
via `curl` or `loadgen` never touches the browser at all — without
something extra, the dashboard would have no way to show it.

**Fix**: the server caches the most recently *successful* prediction —
image bytes (base64-encoded) plus its detections — in memory
(`Server.lastResult`, guarded by a `sync.RWMutex`), updated inside
`HandlePredict` regardless of what triggered the request. A new
`GET /last-result` endpoint serves that cache; `GET /status` gained a
lightweight `last_result_at` timestamp specifically so the dashboard can
poll cheaply and only fetch the (larger, image-carrying) `/last-result`
payload when that timestamp actually changes, rather than
re-downloading the same image every second regardless of whether
anything new happened.

This means: open the dashboard, then run a `curl` command or a full
`loadgen` dataset from a completely different terminal (or a different
machine on the network) — the image and its boxes appear in the
dashboard's Live Feed panel automatically, with no interaction on the
dashboard's own form required.

### Load generation (`cmd/loadgen`)

A CLI tool that sends every image in a directory to the scheduler's
`/predict` endpoint with configurable client-side concurrency, and
reports aggregate latency/throughput/success statistics, with optional
per-request CSV export.

This serves two purposes with the same underlying tool, not two separate
tools: running a whole dataset through the pipeline conveniently (what it
was built for, immediately), and characterizing the scheduler under
concurrent load, the only difference is what
`-concurrency` value you pass and whether you look at the CSV afterward.

Talks to the **scheduler's HTTP API**, not the C++ gRPC service directly
— deliberately, so a load test exercises the full path: routing, worker
pools, admission control, metrics — not just raw inference throughput.

---

## Live Camera Feed
"Start Camera" button that captures the
**dashboard viewer's own webcam** (typically a laptop browsing the
dashboard, not a camera physically attached to the Jetson) and streams
frames through the existing `/predict` endpoint, drawing results back in
real time

### Design choices

- **Self throttling loop.** The next frame is
  captured and sent only after the previous request's response has been
  received and rendered.
- **Frames downscaled to 640px wide before sending.**

NB:

- **Secure-context requirement**: browsers restrict `getUserMedia` to
  HTTPS or `localhost`/`127.0.0.1` origins. Accessing the dashboard as
  `http://<jetson-ip>:8080` from another machine may be blocked outright
  depending on browser policy. Workaround used during testing: SSH
  port-forwarding (`ssh -L 8080:localhost:8080 orin@<jetson-ip>`) so the
  browser sees `localhost`.
- **No routing-policy hot-reload.** Policy is read once from flags at
  process start; changing it requires restarting the scheduler process.

## Health Monitoring and Failure Recovery

### Design

`internal/healthcheck.Monitor` polls every engine's `HealthCheck` RPC
independently, on its own goroutine, at a configurable interval
(`-health-check-interval`, default 3s) with a configurable per-check timeout
(`-health-check-timeout`, default 2s). Tracks per-engine `healthy` state
behind a `sync.RWMutex`.
- **Assumed healthy at startup**, avoids a window where
  nothing is routable before the first poll completes. Each engine's first
  check happens immediately when `Run` starts, not after waiting a full
  interval, keeping that theoretical window small regardless.
- **`Checker` interface**, not a direct dependency on `*engineclient.Client`
  — same pattern as `workerpool.Predictor`: lets the polling/status logic be
  tested against a fake, no real gRPC connection required.
- **`serving=false` (no error) is treated identically to a transport
  error**

### Routing integration

`routing.EngineState` gained a `Healthy` field. Both `LeastQueuePolicy` and
`ThermalAwarePolicy` filter to healthy engines **first**, before any
queue depth or thermal logic runs,  there's no scenario where routing to a
known dead engine is the right answer, so this filtering happens
unconditionally rather than being a policy specific choice.
