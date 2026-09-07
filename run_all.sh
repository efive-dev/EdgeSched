#!/usr/bin/env bash
#
# run_all.sh — starts both inference engines and the Go scheduler
# together, waits for each engine's port to come up before starting the
# scheduler, and cleans up every process on Ctrl+C.
#
# Usage:
#   ./run_all.sh
#   ./run_all.sh -routing-policy=thermal-aware -cheap-tier=yolo26n_int8,yolo26m_fp16 -temp-threshold-c=55

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_BIN="$REPO_ROOT/inference_engine/build/inference_server"
ENGINE_DIR="$REPO_ROOT/inference_model/models/engine"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

if [ ! -x "$ENGINE_BIN" ]; then
  echo "error: inference_server binary not found at $ENGINE_BIN" >&2
  echo "Build it first: cd inference_engine/build && make inference_server" >&2
  exit 1
fi

for engine_file in yolo26n_int8.engine yolo26m_fp16.engine; do
  if [ ! -f "$ENGINE_DIR/$engine_file" ]; then
    echo "error: engine file not found: $ENGINE_DIR/$engine_file" >&2
    exit 1
  fi
done

PIDS=()

cleanup() {
  echo ""
  echo "Shutting down..."
  for pid in "${PIDS[@]:-}"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT INT TERM

# Polls a TCP port using bash's built-in /dev/tcp pseudo-device -- no
# external dependency (nc, curl) required just to check readiness.
wait_for_port() {
  local port="$1" name="$2"
  for _ in $(seq 1 30); do
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
      exec 3>&- 3<&-
      echo "  $name is up on :$port"
      return 0
    fi
    sleep 0.5
  done
  echo "error: timed out waiting for $name on :$port -- check $LOG_DIR/$name.log" >&2
  return 1
}

echo "Starting yolo26n_int8 on :50051..."
"$ENGINE_BIN" "$ENGINE_DIR/yolo26n_int8.engine" 50051 yolo26n_int8 \
  > "$LOG_DIR/yolo26n_int8.log" 2>&1 &
PIDS+=("$!")

echo "Starting yolo26m_fp16 on :50052..."
"$ENGINE_BIN" "$ENGINE_DIR/yolo26m_fp16.engine" 50052 yolo26m_fp16 \
  > "$LOG_DIR/yolo26m_fp16.log" 2>&1 &
PIDS+=("$!")

echo "Waiting for both engines to come up..."
wait_for_port 50051 yolo26n_int8
wait_for_port 50052 yolo26m_fp16

echo "Starting scheduler on :8080..."
echo "  (engine logs: $LOG_DIR/*.log -- scheduler logs print below)"
echo ""
cd "$REPO_ROOT/scheduler"
go run ./cmd/scheduler \
  -engines="yolo26n_int8=localhost:50051,yolo26m_fp16=localhost:50052" \
  "$@"
