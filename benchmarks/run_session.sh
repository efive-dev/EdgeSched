#!/usr/bin/env bash
#
# run_session.sh — runs ONE labeled, sustained benchmark session:
# waits for a thermal cooldown gate, then runs loadgen (sustained mode)
# and statscraper concurrently, saving everything under
# benchmarks/results/<label>/.
#
# The cooldown gate exists because thermal state has real physical
# inertia: running the second session immediately after the first would
# start it from an artificially hot baseline, which is not a fair
# comparison. If the device won't cool down to the target within the
# timeout, the script starts anyway and records the ACTUAL starting
# temperature in meta.json rather than silently pretending the
# comparison was clean.
#
# Usage:
#   ./run_session.sh --label least_queue --dir /tmp/burst_images --duration 5m --concurrency 16
#   (restart scheduler with -routing-policy=thermal-aware ...)
#   ./run_session.sh --label thermal_aware --dir /tmp/burst_images --duration 5m --concurrency 16

set -euo pipefail

LABEL=""
DURATION="5m"
DIR=""
CONCURRENCY=16
ADDR="http://localhost:8080"
COOLDOWN_MAX_TEMP=52
COOLDOWN_TIMEOUT=120

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --dir) DIR="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --addr) ADDR="$2"; shift 2 ;;
    --cooldown-max-temp) COOLDOWN_MAX_TEMP="$2"; shift 2 ;;
    --cooldown-timeout) COOLDOWN_TIMEOUT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$LABEL" || -z "$DIR" ]]; then
  echo "usage: $0 --label NAME --dir IMAGES_DIR [--duration 5m] [--concurrency 16] [--addr http://localhost:8080]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/benchmarks/results/$LABEL"
mkdir -p "$OUT_DIR"

echo "=== Session: $LABEL ==="
echo "Cooldown gate: waiting for max_temp_c <= ${COOLDOWN_MAX_TEMP}C (timeout ${COOLDOWN_TIMEOUT}s)..."

start_wait=$(date +%s)
while true; do
  temp=$(curl -s "$ADDR/status" | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(d['system']['max_temp_c'] if d['system']['valid'] else 999)")
  echo -ne "  current temp: ${temp}C   \r"

  below_threshold=$(python3 -c "print(1 if $temp <= $COOLDOWN_MAX_TEMP else 0)")
  if [[ "$below_threshold" == "1" ]]; then
    echo ""
    echo "  temp OK, starting."
    break
  fi

  now=$(date +%s)
  if (( now - start_wait > COOLDOWN_TIMEOUT )); then
    echo ""
    echo "  WARNING: cooldown timeout reached, starting anyway at ${temp}C."
    echo "  This is recorded in meta.json -- comparison against a run that"
    echo "  started cooler may not be entirely fair; report the starting"
    echo "  temperatures alongside the results."
    break
  fi
  sleep 2
done

start_temp=$(curl -s "$ADDR/status" | python3 -c \
  "import sys,json; print(json.load(sys.stdin)['system']['max_temp_c'])")
start_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cat > "$OUT_DIR/meta.json" <<EOF
{
  "label": "$LABEL",
  "start_time_utc": "$start_time",
  "start_temp_c": $start_temp,
  "duration": "$DURATION",
  "concurrency": $CONCURRENCY,
  "addr": "$ADDR"
}
EOF
echo "  Recorded starting temp: ${start_temp}C (see $OUT_DIR/meta.json)"

cd "$REPO_ROOT/scheduler"

echo ""
echo "Starting statscraper -> $OUT_DIR/timeseries.csv"
go run ./cmd/statscraper -addr "$ADDR" -interval 1s -duration "$DURATION" \
  -out "$OUT_DIR/timeseries.csv" &
SCRAPER_PID=$!

echo "Running loadgen (sustained ${DURATION}, concurrency=${CONCURRENCY}) -> $OUT_DIR/requests.csv"
go run ./cmd/loadgen -addr "$ADDR" -dir "$DIR" -concurrency "$CONCURRENCY" \
  -duration "$DURATION" -label "$LABEL" \
  -out "$OUT_DIR/requests.csv" -summary-json "$OUT_DIR/summary.json"

echo ""
echo "Waiting for statscraper to finish..."
wait "$SCRAPER_PID" 2>/dev/null || true

echo ""
echo "=== Session '$LABEL' complete. Results in $OUT_DIR ==="
echo "  meta.json       -- starting conditions"
echo "  timeseries.csv  -- system state over time (temp, queue depth)"
echo "  requests.csv    -- per-request latency detail"
echo "  summary.json    -- aggregate stats"
