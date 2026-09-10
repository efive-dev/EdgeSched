#!/usr/bin/env python3
"""
plot_sweep.py — turns run_sweep.sh's summary_c*.json (and requests_c*.csv)
files into charts

Usage:
    python3 plot_sweep.py benchmarks/results/sweep
    (reads summary_c*.json and requests_c*.csv from that directory,
    writes PNGs there too)
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_summaries(results_dir: Path):
    summaries = []
    for path in sorted(results_dir.glob("summary_c*.json")):
        with open(path) as f:
            summaries.append(json.load(f))
    summaries.sort(key=lambda s: s["concurrency"])
    return summaries


def load_level_requests(results_dir: Path, concurrency: int):
    path = results_dir / f"requests_c{concurrency}.csv"
    if not path.exists():
        return [], []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["status_code"] == "200" and row["error"] == "":
                rows.append(row)
    if not rows:
        return [], []
    t0 = min(int(r["timestamp_ms"]) for r in rows)
    elapsed = [(int(r["timestamp_ms"]) - t0) / 1000.0 for r in rows]
    wall_ms = [float(r["wall_ms"]) for r in rows]
    return elapsed, wall_ms


def plot_latency_drift(summaries, results_dir: Path):
    n = len(summaries)
    cols = min(4, n)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.5 * cols, 3.5 * rows_n), squeeze=False)
    any_data = False
    for i, s in enumerate(summaries):
        ax = axes[i // cols][i % cols]
        elapsed, wall_ms = load_level_requests(results_dir, s["concurrency"])
        if elapsed:
            any_data = True
            ax.scatter(elapsed, wall_ms, s=5, alpha=0.4, color="tab:blue")
        ax.set_title(f"concurrency={s['concurrency']}", fontsize=10)
        ax.grid(True, alpha=0.3)

    for j in range(n, rows_n * cols):
        axes[j // cols][j % cols].axis("off")

    if not any_data:
        print("no requests_c*.csv files with timestamp_ms found -- "
              "run_sweep.sh needs to have been run with the updated loadgen")
        plt.close(fig)
        return

    fig.supxlabel("Elapsed time within this level's run (s)")
    fig.supylabel("Round-trip latency (ms)")
    fig.suptitle("Latency stability within each concurrency level")
    fig.tight_layout()
    out_path = results_dir / "latency_drift_by_level.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def plot_latency_vs_concurrency(summaries, out_dir: Path):
    concurrency = [s["concurrency"] for s in summaries]
    p50 = [s["latency_p50_ms"] for s in summaries]
    p95 = [s["latency_p95_ms"] for s in summaries]
    p99 = [s["latency_p99_ms"] for s in summaries]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(concurrency, p50, marker="o", label="p50")
    ax.plot(concurrency, p95, marker="o", label="p95")
    ax.plot(concurrency, p99, marker="o", label="p99")
    ax.set_xlabel("Client concurrency (offered load)")
    ax.set_ylabel("Round-trip latency (ms)")
    ax.set_title("Latency vs. offered load")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "latency_vs_concurrency.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def plot_throughput_vs_concurrency(summaries, out_dir: Path):
    concurrency = [s["concurrency"] for s in summaries]
    throughput = [s["throughput_rps"] for s in summaries]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(concurrency, throughput, marker="o", color="tab:green")
    ax.set_xlabel("Client concurrency (offered load)")
    ax.set_ylabel("Throughput (requests/sec)")
    ax.set_title("Throughput vs. offered load")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "throughput_vs_concurrency.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def plot_rejection_rate(summaries, out_dir: Path):
    concurrency = [s["concurrency"] for s in summaries]
    rejection_pct = []
    for s in summaries:
        total = s["total"]
        rejected = s.get("status_counts", {}).get("503", 0)
        rejection_pct.append(100.0 * rejected / total if total else 0.0)
    if all(p == 0 for p in rejection_pct):
        print("no rejections (503s) observed at any tested concurrency level -- "
              "try higher -levels in run_sweep.sh to actually exercise admission control")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(concurrency, rejection_pct, marker="o", color="tab:red")
    ax.set_xlabel("Client concurrency (offered load)")
    ax.set_ylabel("Rejected requests (%)")
    ax.set_title("Admission control: rejection rate vs. offered load")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "rejection_rate_vs_concurrency.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def print_baseline(summaries):
    if not summaries:
        return
    baseline = summaries[0]
    print("\n=== Baseline (chapter 1) ===")
    print(f"concurrency={baseline['concurrency']}: "
          f"p50={baseline['latency_p50_ms']:.1f}ms "
          f"p95={baseline['latency_p95_ms']:.1f}ms "
          f"p99={baseline['latency_p99_ms']:.1f}ms "
          f"throughput={baseline['throughput_rps']:.2f} req/s")


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <results_dir>", file=sys.stderr)
        sys.exit(1)
    results_dir = Path(sys.argv[1])
    summaries = load_summaries(results_dir)
    if not summaries:
        print(f"no summary_c*.json files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(summaries)} sweep levels: "
          f"{[s['concurrency'] for s in summaries]}")
    print_baseline(summaries)
    plot_latency_vs_concurrency(summaries, results_dir)
    plot_throughput_vs_concurrency(summaries, results_dir)
    plot_rejection_rate(summaries, results_dir)
    plot_latency_drift(summaries, results_dir)


if __name__ == "__main__":
    main()
