#!/usr/bin/env python3
"""Maze navigation algorithm benchmark.

Compares three algorithms on the same 8x8 DFS maze (seed=42):
  1. Left-hand rule (wall follower)  — reactive, no map
  2. A* + pure pursuit               — global planner, full map
  3. RL agent (SAC)                   — learned, no map

All three share identical movement rules:
  - Differential drive: max 0.3 m/s linear, 2.0 rad/s angular
  - Same collision model (robot radius 0.06m)
  - Same LiDAR (360 rays, 3.5m range, downsampled to 36)
  - Same maze, same start/exit

Usage:
    python run_benchmark.py                           # all algorithms, center start
    python run_benchmark.py --rl_model path/to/model  # use real RL model
    python run_benchmark.py --num_trials 10            # multiple random starts
    python run_benchmark.py --no_plot                  # skip visualization
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from maze_sim import (
    CELL_SIZE,
    COLS,
    EXIT_POS,
    MAX_EPISODE_TIME,
    ROWS,
    MazeSim,
    Metrics,
    build_maze_grid,
    cell_to_world,
    compute_bfs_distance_field,
)
from algo_wall_follower import WallFollowerPolicy
from algo_astar import AStarPolicy
from algo_rl_agent import make_rl_policy


# ---------------------------------------------------------------------------
# Helper: get valid spawn positions for multi-trial experiments
# ---------------------------------------------------------------------------

def get_valid_cell_positions() -> list[tuple[float, float]]:
    """Return world (x, y) for every passable cell."""
    grid = build_maze_grid()
    positions = []
    for r in range(ROWS):
        for c in range(COLS):
            if not grid[2 * r + 1][2 * c + 1]:
                positions.append(cell_to_world(r, c))
    return positions


# ---------------------------------------------------------------------------
# Run a single episode
# ---------------------------------------------------------------------------

def run_episode(
    sim: MazeSim,
    policy,
    algo_name: str,
    start_pos: tuple[float, float] = (0.0, 0.0),
    start_yaw: float = 0.0,
    verbose: bool = False,
) -> Metrics:
    """Run one episode to completion and return metrics."""
    obs = sim.reset(start_pos=start_pos, start_yaw=start_yaw)

    policy.reset()
    # A* needs to plan a path AFTER reset (reset clears waypoints)
    if hasattr(policy, "plan"):
        policy.plan(start_pos[0], start_pos[1])

    step_count = 0
    v, w = 0.0, 0.0

    while not obs.done:
        # Build action based on algorithm type
        if algo_name == "A*":
            v, w = policy.compute_action(obs.x, obs.y, obs.yaw, obs.lidar_36)
        elif algo_name == "Wall Follower":
            v, w = policy.compute_action(obs.lidar_36)
        elif algo_name == "RL Agent":
            obs_vec = sim.get_obs_vector(obs, v, w)
            v, w = policy.compute_action(obs_vec)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        obs = sim.step(v, w)
        step_count += 1

        if verbose and step_count % 600 == 0:
            print(f"  [{algo_name}] t={obs.time:.1f}s  pos=({obs.x:.2f}, {obs.y:.2f})  "
                  f"dist_to_exit={obs.exit_dist_norm * 5.7:.2f}m")

    return sim.metrics


# ---------------------------------------------------------------------------
# Benchmark multiple algorithms
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    algo_name: str
    trial_id: int
    start_pos: tuple[float, float]
    metrics: Metrics


def run_benchmark(
    num_trials: int = 1,
    rl_model_path: str | None = None,
    verbose: bool = True,
    fixed_start: tuple[float, float] | None = None,
    fixed_yaw: float | None = None,
) -> list[TrialResult]:
    """Run comparative benchmark across all algorithms."""

    # Initialize policies
    wall_policy = WallFollowerPolicy()
    astar_policy = AStarPolicy()
    rl_policy = make_rl_policy(rl_model_path)

    algorithms = [
        ("Wall Follower", wall_policy),
        ("A*", astar_policy),
        ("RL Agent", rl_policy),
    ]

    # Generate start positions
    if fixed_start is not None:
        starts = [fixed_start] * num_trials
    elif num_trials == 1:
        starts = [(0.0, 0.0)]  # center cell
    else:
        valid_positions = get_valid_cell_positions()
        rng = np.random.RandomState(123)
        indices = rng.choice(len(valid_positions), size=num_trials, replace=True)
        starts = [valid_positions[i] for i in indices]

    # Generate start yaws
    if fixed_yaw is not None:
        yaws = [fixed_yaw] * num_trials
    else:
        rng = np.random.RandomState(456)
        yaws = [float(rng.uniform(-math.pi, math.pi)) for _ in range(num_trials)]

    results: list[TrialResult] = []

    sim = MazeSim()

    for trial_id in range(num_trials):
        sx, sy = starts[trial_id]
        yaw = yaws[trial_id]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Trial {trial_id + 1}/{num_trials}  "
                  f"start=({sx:.2f}, {sy:.2f})  yaw={math.degrees(yaw):.0f}°")
            print(f"{'='*60}")

        for algo_name, policy in algorithms:
            t0 = time.perf_counter()
            metrics = run_episode(sim, policy, algo_name,
                                  start_pos=(sx, sy), start_yaw=yaw,
                                  verbose=verbose)
            wall_time = time.perf_counter() - t0

            results.append(TrialResult(algo_name, trial_id, (sx, sy), metrics))

            if verbose:
                status = "ESCAPED" if metrics.escaped else "TIMEOUT"
                print(f"  [{algo_name:15s}] {status}  "
                      f"time={metrics.total_time:6.1f}s  "
                      f"path={metrics.path_length:6.2f}m  "
                      f"collisions={metrics.num_collisions:4d}  "
                      f"cells={metrics.cells_visited:3d}  "
                      f"(compute: {wall_time:.2f}s)")

    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(results: list[TrialResult]) -> None:
    """Print aggregated comparison table."""
    algo_names = sorted(set(r.algo_name for r in results),
                        key=lambda n: ["Wall Follower", "A*", "RL Agent"].index(n))

    print(f"\n{'='*80}")
    print(f"{'BENCHMARK SUMMARY':^80}")
    print(f"{'='*80}")

    header = (f"{'Algorithm':15s} | {'Success':>7s} | {'Avg Time':>9s} | "
              f"{'Avg Path':>9s} | {'Avg Coll':>8s} | {'Avg Cells':>9s}")
    print(header)
    print("-" * 80)

    for name in algo_names:
        algo_results = [r for r in results if r.algo_name == name]
        n = len(algo_results)
        escaped = sum(1 for r in algo_results if r.metrics.escaped)
        success_rate = escaped / n if n > 0 else 0.0

        escaped_results = [r for r in algo_results if r.metrics.escaped]
        if escaped_results:
            avg_time = np.mean([r.metrics.total_time for r in escaped_results])
            avg_path = np.mean([r.metrics.path_length for r in escaped_results])
        else:
            avg_time = float("inf")
            avg_path = float("inf")

        avg_coll = np.mean([r.metrics.num_collisions for r in algo_results])
        avg_cells = np.mean([r.metrics.cells_visited for r in algo_results])

        time_str = f"{avg_time:7.1f}s" if avg_time < float("inf") else "    N/A"
        path_str = f"{avg_path:7.2f}m" if avg_path < float("inf") else "    N/A"

        print(f"{name:15s} | {escaped:3d}/{n:<3d} | {time_str:>9s} | "
              f"{path_str:>9s} | {avg_coll:>8.1f} | {avg_cells:>9.1f}")

    print(f"{'='*80}")

    # Per-algorithm detailed stats for escaped episodes
    for name in algo_names:
        escaped_results = [r for r in results if r.algo_name == name and r.metrics.escaped]
        if not escaped_results:
            continue
        times = [r.metrics.total_time for r in escaped_results]
        paths = [r.metrics.path_length for r in escaped_results]
        print(f"\n  {name} (escaped only):")
        print(f"    Time:  min={min(times):.1f}s  max={max(times):.1f}s  "
              f"std={np.std(times):.1f}s")
        print(f"    Path:  min={min(paths):.2f}m  max={max(paths):.2f}m  "
              f"std={np.std(paths):.2f}m")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(results: list[TrialResult], save_path: str | None = None) -> None:
    """Generate comparison plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots.")
        return

    grid = build_maze_grid()
    h = len(grid)
    w = len(grid[0])

    colors = {
        "Wall Follower": "#e74c3c",  # red
        "A*": "#2ecc71",             # green
        "RL Agent": "#3498db",       # blue
    }

    # --- Figure 1: Trajectory overlay on maze ---
    fig1, axes1 = plt.subplots(1, 1, figsize=(10, 10))
    ax = axes1

    # Draw maze walls
    _CX_OFF = 2 * (ROWS // 2) + 1
    _CY_OFF = 2 * (COLS // 2) + 1
    for row in range(h):
        for col in range(w):
            if grid[row][col]:
                cx = (col - _CX_OFF) * CELL_SIZE
                cy = (_CY_OFF - row) * CELL_SIZE
                half = CELL_SIZE / 2.0
                rect = patches.Rectangle(
                    (cx - half, cy - half), CELL_SIZE, CELL_SIZE,
                    linewidth=0, edgecolor="none",
                    facecolor="#8e8ea0", alpha=0.8
                )
                ax.add_patch(rect)

    # Draw trajectories (first trial only for clarity)
    trial0 = [r for r in results if r.trial_id == 0]
    for r in trial0:
        traj = r.metrics.trajectory
        if len(traj) > 1:
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            ax.plot(xs, ys, color=colors.get(r.algo_name, "gray"),
                    linewidth=1.5, alpha=0.8, label=r.algo_name)
            # Start marker
            ax.plot(xs[0], ys[0], "o", color=colors.get(r.algo_name, "gray"),
                    markersize=8)
            # End marker
            ax.plot(xs[-1], ys[-1], "s", color=colors.get(r.algo_name, "gray"),
                    markersize=8)

    # Mark exit
    ax.plot(EXIT_POS[0], EXIT_POS[1], "*", color="gold", markersize=20,
            markeredgecolor="black", markeredgewidth=1, zorder=10, label="Exit")

    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal")
    ax.set_title("Maze Navigation: Trajectory Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.2)
    fig1.tight_layout()

    # --- Figure 2: Bar chart comparison ---
    algo_names = ["Wall Follower", "A*", "RL Agent"]
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

    # Completion time
    ax2 = axes2[0]
    for i, name in enumerate(algo_names):
        escaped = [r for r in results if r.algo_name == name and r.metrics.escaped]
        if escaped:
            times = [r.metrics.total_time for r in escaped]
            ax2.bar(i, np.mean(times), color=colors[name], alpha=0.8,
                    yerr=np.std(times) if len(times) > 1 else 0, capsize=5)
        else:
            ax2.bar(i, MAX_EPISODE_TIME, color=colors[name], alpha=0.3)
            ax2.text(i, MAX_EPISODE_TIME / 2, "N/A", ha="center", fontweight="bold")
    ax2.set_xticks(range(len(algo_names)))
    ax2.set_xticklabels(algo_names, fontsize=10)
    ax2.set_ylabel("Time (s)")
    ax2.set_title("Completion Time (lower = better)")

    # Path length
    ax3 = axes2[1]
    for i, name in enumerate(algo_names):
        escaped = [r for r in results if r.algo_name == name and r.metrics.escaped]
        if escaped:
            paths = [r.metrics.path_length for r in escaped]
            ax3.bar(i, np.mean(paths), color=colors[name], alpha=0.8,
                    yerr=np.std(paths) if len(paths) > 1 else 0, capsize=5)
        else:
            ax3.bar(i, 0, color=colors[name], alpha=0.3)
            ax3.text(i, 0.1, "N/A", ha="center", fontweight="bold")
    ax3.set_xticks(range(len(algo_names)))
    ax3.set_xticklabels(algo_names, fontsize=10)
    ax3.set_ylabel("Path Length (m)")
    ax3.set_title("Path Length (lower = better)")

    # Collisions
    ax4 = axes2[2]
    for i, name in enumerate(algo_names):
        algo_r = [r for r in results if r.algo_name == name]
        colls = [r.metrics.num_collisions for r in algo_r]
        ax4.bar(i, np.mean(colls), color=colors[name], alpha=0.8,
                yerr=np.std(colls) if len(colls) > 1 else 0, capsize=5)
    ax4.set_xticks(range(len(algo_names)))
    ax4.set_xticklabels(algo_names, fontsize=10)
    ax4.set_ylabel("Collisions")
    ax4.set_title("Collision Count (lower = better)")

    fig2.suptitle("Maze Benchmark: Algorithm Comparison", fontsize=14, fontweight="bold")
    fig2.tight_layout()

    if save_path:
        base = Path(save_path)
        fig1.savefig(str(base / "trajectory_comparison.png"), dpi=150, bbox_inches="tight")
        fig2.savefig(str(base / "metrics_comparison.png"), dpi=150, bbox_inches="tight")
        print(f"\nPlots saved to {base}/")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Maze navigation algorithm benchmark")
    parser.add_argument("--num_trials", type=int, default=1,
                        help="Number of trials with different start positions (default: 1)")
    parser.add_argument("--rl_model", type=str, default=None,
                        help="Path to trained RL model (.pt TorchScript)")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip visualization")
    parser.add_argument("--save_plots", type=str, default=None,
                        help="Directory to save plots (default: show interactively)")
    parser.add_argument("--start_x", type=float, default=None,
                        help="Fixed start X position")
    parser.add_argument("--start_y", type=float, default=None,
                        help="Fixed start Y position")
    parser.add_argument("--start_yaw", type=float, default=None,
                        help="Fixed start yaw in degrees")
    args = parser.parse_args()

    fixed_start = None
    if args.start_x is not None and args.start_y is not None:
        fixed_start = (args.start_x, args.start_y)

    fixed_yaw = None
    if args.start_yaw is not None:
        fixed_yaw = math.radians(args.start_yaw)

    print("=" * 60)
    print("  MAZE NAVIGATION BENCHMARK")
    print("  Maze: 8x8 DFS (seed=42)")
    print(f"  Trials: {args.num_trials}")
    print(f"  RL model: {args.rl_model or 'DummyRLPolicy (heuristic)'}")
    print("  Shared rules: diff-drive, v_max=0.3m/s, ω_max=2.0rad/s")
    print("=" * 60)

    results = run_benchmark(
        num_trials=args.num_trials,
        rl_model_path=args.rl_model,
        verbose=True,
        fixed_start=fixed_start,
        fixed_yaw=fixed_yaw,
    )

    print_summary(results)

    if not args.no_plot:
        plot_results(results, save_path=args.save_plots)

    return results


if __name__ == "__main__":
    main()
