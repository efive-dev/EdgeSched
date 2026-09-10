#!/usr/bin/env bash
#
# run_sweep.sh — runs loadgen once per concurrency level, in SUSTAINED
# mode (fixed wall clock duration per level, not "drain the directory
# once") so every level is measured over a comparable time window
# regardless of dataset size. Without this, a small image set finishes
# almost instantly at high concurrency, giving an unreliable throughput
# number from too few samples -- fixed duration avoids that entirely.
#
# Usage:
#   ./run_sweep.sh --dir /tmp/burst_images
#   ./run_sweep.sh --dir /tmp/burst_images --engine yolo26n_int8 --levels "1 2 4 8 16 32 64" --duration 20s

set -euo pipefail

DIR=""
ADDR="http://localhost:8080"
ENGINE=""
LEVELS="1 2 4 8 16 32 64"
WARMUP=5
DURATION="20s"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --addr) ADDR="$2"; shift 2 ;;
    --engine) ENGINE="$2"; shift 2 ;;
    --levels) LEVELS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$DIR" ]]; then
  echo "usage: $0 --dir IMAGES_DIR [--engine NAME] [--levels \"1 2 4 8 16\"] [--duration 20s] [--addr http://localhost:8080]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/benchmarks/results/sweep"
mkdir -p "$OUT_DIR"

echo "Sweep levels: $LEVELS"
echo "Duration per level: $DURATION (sustained -- every level measured over the same wall-clock window)"
echo "Output: $OUT_DIR"
echo ""

cd "$REPO_ROOT/scheduler"

for c in $LEVELS; do
  echo "=== concurrency=$c ==="
  ENGINE_ARG=()
  if [[ -n "$ENGINE" ]]; then ENGINE_ARG=(-engine "$ENGINE"); fi

  go run ./cmd/loadgen \
    -addr "$ADDR" \
    -dir "$DIR" \
    -concurrency "$c" \
    -duration "$DURATION" \
    -warmup "$WARMUP" \
    "${ENGINE_ARG[@]}" \
    -label "c$c" \
    -summary-json "$OUT_DIR/summary_c$c.json" \
    -out "$OUT_DIR/requests_c$c.csv"

  echo ""
  sleep 3  # brief pause between levels, lets any residual queue drain
done

echo "Sweep complete. Summaries: $OUT_DIR/summary_c*.json"
echo "Plot with: python3 benchmarks/plot_sweep.py $OUT_DIR"
