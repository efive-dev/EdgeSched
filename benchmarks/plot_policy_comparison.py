#!/usr/bin/env python3
"""
plot_policy_comparison.py, overlays two run_session.sh outputs (one per
routing policy)

Usage:
    python3 plot_policy_comparison.py \
        benchmarks/results/least_queue benchmarks/results/thermal_aware
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

BIN_SECONDS = 5  # throughput bucket width


def load_timeseries(session_dir: Path):
    rows = []
    with open(session_dir / "timeseries.csv") as f:
        reader = csv.DictReader(f)
        engine_cols = [c for c in reader.fieldnames if c.startswith("queue_")]
        for row in reader:
            rows.append(row)
    return rows, engine_cols


def load_requests(session_dir: Path):
    path = session_dir / "requests.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status_code"] == "200" and row["error"] == "":
                rows.append(row)
    if not rows:
        return [], []
    t0 = min(int(r["timestamp_ms"]) for r in rows)
    elapsed = [(int(r["timestamp_ms"]) - t0) / 1000.0 for r in rows]
    wall_ms = [float(r["wall_ms"]) for r in rows]
    return elapsed, wall_ms


def bin_throughput(elapsed_sec, bin_width=BIN_SECONDS):
    if not elapsed_sec:
        return [], []
    max_t = max(elapsed_sec)
    n_bins = int(max_t // bin_width) + 1
    counts = [0] * n_bins
    for t in elapsed_sec:
        counts[int(t // bin_width)] += 1
    centers = [(i + 0.5) * bin_width for i in range(n_bins)]
    rps = [c / bin_width for c in counts]
    return centers, rps


def load_meta(session_dir: Path):
    meta_path = session_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def load_summary(session_dir: Path):
    summary_path = session_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {}


def valid_rows(rows):
    return [r for r in rows if r["valid"].lower() == "true"]


def plot_temperature_comparison(sessions, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, s in sessions.items():
        rows = valid_rows(s["ts_rows"])
        elapsed = [float(r["elapsed_sec"]) for r in rows]
        temps = [float(r["max_temp_c"]) for r in rows]
        ax.plot(elapsed, temps, label=label)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Max temperature (°C)")
    ax.set_title("Temperature over time: routing policy comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "policy_comparison_temperature.png")


def plot_engine_traffic_share(sessions, out_dir: Path):
    fig, axes = plt.subplots(1, len(sessions), figsize=(6 * len(sessions), 5), sharey=True)
    if len(sessions) == 1:
        axes = [axes]
    for ax, (label, s) in zip(axes, sessions.items()):
        rows = s["ts_rows"]
        elapsed = [float(r["elapsed_sec"]) for r in rows]
        for col in s["engine_cols"]:
            engine_name = col.replace("queue_", "")
            values = [int(r[col]) for r in rows]
            ax.plot(elapsed, values, label=engine_name)
        ax.set_xlabel("Elapsed time (s)")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Queue depth")
    fig.suptitle("Per-engine queue depth over time: routing policy comparison")
    fig.tight_layout()
    _save(fig, out_dir / "policy_comparison_queue_depth.png")


def plot_throughput_over_time(sessions, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, s in sessions.items():
        centers, rps = bin_throughput(s["req_elapsed"])
        ax.plot(centers, rps, marker="o", markersize=3, label=label)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel(f"Throughput (req/sec, {BIN_SECONDS}s bins)")
    ax.set_title("Throughput over time: routing policy comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "policy_comparison_throughput_over_time.png")


def plot_latency_over_time(sessions, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, s in sessions.items():
        ax.scatter(s["req_elapsed"], s["req_wall_ms"], s=6, alpha=0.35, label=label)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Round-trip latency (ms)")
    ax.set_title("Per-request latency over time: routing policy comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "policy_comparison_latency_over_time.png")


def plot_throughput_bar(sessions, out_dir: Path):
    labels = list(sessions.keys())
    throughputs = [sessions[l]["summary"].get("throughput_rps", 0) for l in labels]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, throughputs, color=["tab:blue", "tab:orange"][: len(labels)])
    ax.set_ylabel("Overall throughput (req/sec)")
    ax.set_title("Aggregate throughput: routing policy comparison")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}",
                ha="center", va="bottom")
    fig.tight_layout()
    _save(fig, out_dir / "policy_comparison_throughput_bar.png")


def plot_combined(sessions, out_dir: Path):
    fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=True)
    for label, s in sessions.items():
        rows = valid_rows(s["ts_rows"])
        elapsed = [float(r["elapsed_sec"]) for r in rows]
        temps = [float(r["max_temp_c"]) for r in rows]
        axes[0].plot(elapsed, temps, label=label)
        ts_elapsed_all = [float(r["elapsed_sec"]) for r in s["ts_rows"]]
        total_queue = [sum(int(r[c]) for c in s["engine_cols"]) for r in s["ts_rows"]]
        axes[1].plot(ts_elapsed_all, total_queue, label=label)
        axes[2].scatter(s["req_elapsed"], s["req_wall_ms"], s=5, alpha=0.3, label=label)
        centers, rps = bin_throughput(s["req_elapsed"])
        axes[3].plot(centers, rps, marker="o", markersize=3, label=label)
    axes[0].set_ylabel("Max temp (°C)")
    axes[0].set_title("Combined overview: routing policy comparison")
    axes[1].set_ylabel("Total queue depth\n(all engines summed)")
    axes[2].set_ylabel("Latency (ms)")
    axes[3].set_ylabel(f"Throughput\n(req/s, {BIN_SECONDS}s bins)")
    axes[3].set_xlabel("Elapsed time (s)")
    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "policy_comparison_combined.png")


def _save(fig, path: Path):
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def print_summary(sessions):
    print("\n=== Session summaries ===")
    for label, s in sessions.items():
        meta = s["meta"]
        summary = s["summary"]
        print(f"\n{label}:")
        print(f"  starting temp: {meta.get('start_temp_c', 'unknown')}C")
        print(f"  duration:      {meta.get('duration', summary.get('duration_sec', 'unknown'))}")
        print(f"  requests:      {summary.get('total', '?')} "
              f"({summary.get('succeeded', '?')} succeeded, {summary.get('failed', '?')} failed)")
        print(f"  throughput:    {summary.get('throughput_rps', '?')} req/s")
        print(f"  latency p50/p95/p99: "
              f"{summary.get('latency_p50_ms', '?')}/"
              f"{summary.get('latency_p95_ms', '?')}/"
              f"{summary.get('latency_p99_ms', '?')} ms")
        print(f"  engine breakdown: {summary.get('engine_counts', {})}")


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <session_dir_1> <session_dir_2> [more session dirs...]",
              file=sys.stderr)
        sys.exit(1)
    session_dirs = [Path(p) for p in sys.argv[1:]]
    sessions = {}
    for d in session_dirs:
        if not (d / "timeseries.csv").exists() or not (d / "requests.csv").exists():
            print(f"warning: {d} missing timeseries.csv or requests.csv, skipping", file=sys.stderr)
            continue
        ts_rows, engine_cols = load_timeseries(d)
        req_elapsed, req_wall_ms = load_requests(d)
        sessions[d.name] = {
            "ts_rows": ts_rows,
            "engine_cols": engine_cols,
            "req_elapsed": req_elapsed,
            "req_wall_ms": req_wall_ms,
            "meta": load_meta(d),
            "summary": load_summary(d),
        }
    if len(sessions) < 2:
        print("need at least 2 valid session directories to compare", file=sys.stderr)
        sys.exit(1)
    out_dir = session_dirs[0].parent
    plot_temperature_comparison(sessions, out_dir)
    plot_engine_traffic_share(sessions, out_dir)
    plot_throughput_over_time(sessions, out_dir)
    plot_latency_over_time(sessions, out_dir)
    plot_throughput_bar(sessions, out_dir)
    plot_combined(sessions, out_dir)
    print_summary(sessions)


if __name__ == "__main__":
    main()
