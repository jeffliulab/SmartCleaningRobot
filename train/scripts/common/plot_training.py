#!/usr/bin/env python3
"""Plot training curves from training_metrics.csv.

Usage::

    # From project root:
    python train/scripts/common/plot_training.py <log_dir>
    python train/scripts/common/plot_training.py <log_dir> --smooth 10

    # Example with a specific run:
    python train/scripts/common/plot_training.py \\
        train/logs/skrl/maze_escape/2026-03-16_10-00-00_sac_torch

Reads ``training_metrics.csv`` from the given log directory and generates
a 2x2 training curve figure saved as ``training_curves.png``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np


def load_csv(path: str) -> dict[str, np.ndarray]:
    """Load a CSV file into a dict of {column_name: numpy_array}."""
    with open(path) as f:
        reader = csv.DictReader(f)
        columns: dict[str, list[float]] = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k, v in row.items():
                columns[k].append(float(v))
    return {k: np.array(v) for k, v in columns.items()}


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average smoothing."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from CSV logs")
    parser.add_argument("log_dir", help="Path to training log directory")
    parser.add_argument(
        "--smooth", type=int, default=5,
        help="Smoothing window size for moving average (default: 5)",
    )
    parser.add_argument(
        "--save_only", action="store_true",
        help="Save PNG without showing the interactive plot window",
    )
    args = parser.parse_args()

    csv_path = os.path.join(args.log_dir, "training_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    try:
        import matplotlib
        if args.save_only:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required.  pip install matplotlib")
        sys.exit(1)

    data = load_csv(csv_path)
    ts = data["timestep"]
    w = args.smooth

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- 1. Mean episode reward ----
    ax = axes[0, 0]
    raw = data["mean_reward"]
    ax.plot(ts, raw, alpha=0.3, color="C0", linewidth=0.8)
    if len(raw) >= w:
        ts_s = ts[w - 1:]
        ax.plot(ts_s, smooth(raw, w), color="C0", linewidth=2,
                label=f"smooth({w})")
    ax.fill_between(
        ts,
        raw - data["std_reward"],
        raw + data["std_reward"],
        alpha=0.12, color="C0",
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Mean Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 2. Mean episode length ----
    ax = axes[0, 1]
    raw = data["mean_ep_length"]
    ax.plot(ts, raw, alpha=0.3, color="C1", linewidth=0.8)
    if len(raw) >= w:
        ts_s = ts[w - 1:]
        ax.plot(ts_s, smooth(raw, w), color="C1", linewidth=2,
                label=f"smooth({w})")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Episode Length (steps)")
    ax.set_title("Mean Episode Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 3. Escape success rate ----
    ax = axes[1, 0]
    esc = data["escape_rate"] * 100
    ax.plot(ts, esc, alpha=0.3, color="C2", linewidth=0.8)
    if len(esc) >= w:
        ts_s = ts[w - 1:]
        ax.plot(ts_s, smooth(esc, w), color="C2", linewidth=2,
                label=f"smooth({w})")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Escape Rate (%)")
    ax.set_title("Maze Escape Success Rate")
    ax.set_ylim(-5, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 4. Rolling reward + escape rate (dual axis) ----
    ax = axes[1, 1]
    ax.plot(ts, data["rolling_mean_reward"], color="C0", linewidth=1.5,
            label="Rolling reward")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")

    ax2 = ax.twinx()
    ax2.plot(ts, data["rolling_escape_rate"] * 100, color="C2",
             linewidth=1.5, linestyle="--", label="Rolling escape %")
    ax2.set_ylabel("Escape Rate (%)", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")
    ax2.set_ylim(-5, 105)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax.set_title("Rolling Averages (last 200 episodes)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Training Progress \u2014 Maze Escape",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(args.log_dir, "training_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if not args.save_only:
        plt.show()


if __name__ == "__main__":
    main()
