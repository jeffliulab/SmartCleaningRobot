"""Training metric logger for Isaac Lab RL environments.

Wraps a vectorized environment to track per-episode metrics (reward,
length, escape success) and write them to CSV files at regular intervals.

The aggregate CSV can be visualized with ``plot_training.py``::

    python train/scripts/common/plot_training.py <log_dir>

Usage in train.py::

    from metric_logger import TrainingMetricLogger
    env = gym.make(task, cfg=env_cfg)
    env = TrainingMetricLogger(env, log_dir=log_dir)
    env = SkrlVecEnvWrapper(env)
"""

from __future__ import annotations

import csv
import os
import time
from collections import deque

import gymnasium as gym
import numpy as np


class TrainingMetricLogger(gym.Wrapper):
    """Gym wrapper that logs per-episode training metrics to CSV.

    Sits between the Isaac Lab env and the skrl wrapper.  Intercepts
    ``step()`` calls to accumulate per-env rewards.  When an episode
    ends (terminated or truncated), the episode total reward, length,
    and escape status are recorded.

    Two CSV files are produced in *log_dir*:

    ``training_metrics.csv``
        Aggregated window statistics (one row per *log_interval* trainer
        steps).  Best for plotting training curves.

    ``training_episodes.csv``
        One row per completed episode.  Best for fine-grained analysis.
    """

    def __init__(
        self,
        env: gym.Env,
        log_dir: str,
        log_interval: int = 5000,
    ):
        super().__init__(env)
        self._log_dir = log_dir
        self._log_interval = log_interval

        num_envs = env.unwrapped.num_envs
        self._num_envs = num_envs

        # Per-env episode accumulators
        self._ep_rew = np.zeros(num_envs, dtype=np.float64)
        self._ep_len = np.zeros(num_envs, dtype=np.int64)

        # Completed-episode buffer (flushed every log_interval)
        self._ep_buf: list[dict] = []

        # Rolling window for smoothed console display
        self._recent_rew: deque[float] = deque(maxlen=200)
        self._recent_esc: deque[bool] = deque(maxlen=200)

        # Counters
        self._step_count = 0        # trainer steps (each = num_envs env steps)
        self._total_episodes = 0
        self._t0 = time.time()
        self._last_flush = 0

        # ---------- Aggregate CSV ----------
        os.makedirs(log_dir, exist_ok=True)
        self._agg_path = os.path.join(log_dir, "training_metrics.csv")
        with open(self._agg_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestep", "wall_time_s", "total_episodes", "window_episodes",
                "mean_reward", "std_reward", "min_reward", "max_reward",
                "mean_ep_length", "std_ep_length",
                "escape_rate", "escape_count",
                "rolling_mean_reward", "rolling_escape_rate",
            ])

        # ---------- Per-episode CSV ----------
        self._ep_path = os.path.join(log_dir, "training_episodes.csv")
        self._ep_rows: list[list] = []
        with open(self._ep_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestep", "wall_time_s", "episode_id",
                "reward", "length", "escaped",
            ])

    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._ep_rew[:] = 0.0
        self._ep_len[:] = 0
        return result

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        rew = reward.detach().cpu().numpy().reshape(-1)
        term = terminated.detach().cpu().numpy().reshape(-1).astype(bool)
        trunc = truncated.detach().cpu().numpy().reshape(-1).astype(bool)

        self._ep_rew += rew
        self._ep_len += 1

        done = term | trunc
        done_idx = np.where(done)[0]

        if done_idx.size:
            ts = self._step_count * self._num_envs
            wt = time.time() - self._t0
            for i in done_idx:
                self._total_episodes += 1
                ep = dict(
                    reward=float(self._ep_rew[i]),
                    length=int(self._ep_len[i]),
                    escaped=bool(term[i]),
                )
                self._ep_buf.append(ep)
                self._recent_rew.append(ep["reward"])
                self._recent_esc.append(ep["escaped"])
                self._ep_rows.append([
                    ts, f"{wt:.1f}", self._total_episodes,
                    f"{ep['reward']:.2f}", ep["length"], int(ep["escaped"]),
                ])
                self._ep_rew[i] = 0.0
                self._ep_len[i] = 0

        if self._step_count - self._last_flush >= self._log_interval:
            self._flush()
            self._last_flush = self._step_count

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------

    def _flush(self):
        """Write buffered episode data and aggregate stats to CSV."""
        # Per-episode rows
        if self._ep_rows:
            with open(self._ep_path, "a", newline="") as f:
                csv.writer(f).writerows(self._ep_rows)
            self._ep_rows.clear()

        if not self._ep_buf:
            return

        rews = [e["reward"] for e in self._ep_buf]
        lens = [e["length"] for e in self._ep_buf]
        esc = sum(e["escaped"] for e in self._ep_buf)
        n = len(self._ep_buf)

        ts = self._step_count * self._num_envs
        wt = time.time() - self._t0

        roll_rew = float(np.mean(list(self._recent_rew))) if self._recent_rew else 0.0
        roll_esc = (
            sum(self._recent_esc) / len(self._recent_esc)
            if self._recent_esc
            else 0.0
        )

        with open(self._agg_path, "a", newline="") as f:
            csv.writer(f).writerow([
                ts, f"{wt:.1f}", self._total_episodes, n,
                f"{np.mean(rews):.4f}", f"{np.std(rews):.4f}",
                f"{np.min(rews):.4f}", f"{np.max(rews):.4f}",
                f"{np.mean(lens):.1f}", f"{np.std(lens):.1f}",
                f"{esc / n:.4f}", esc,
                f"{roll_rew:.4f}", f"{roll_esc:.4f}",
            ])

        esc_pct = 100 * esc / n
        print(
            f"  [Metrics] step={ts:>12,d} | "
            f"ep={self._total_episodes:>7,d} | "
            f"rew={np.mean(rews):>8.1f} +/-{np.std(rews):>6.1f} | "
            f"len={np.mean(lens):>6.0f} | "
            f"esc={esc}/{n} ({esc_pct:>5.1f}%) | "
            f"t={wt:>7.0f}s"
        )

        self._ep_buf.clear()

    def close(self):
        self._flush()
        wt = time.time() - self._t0
        print(f"\n{'=' * 60}")
        print("  Training metrics saved to:")
        print(f"    {self._agg_path}")
        print(f"    {self._ep_path}")
        print(f"  Total episodes:  {self._total_episodes:,d}")
        print(f"  Total wall time: {wt:.0f}s ({wt / 3600:.1f}h)")
        if self._recent_rew:
            print(f"  Final reward (rolling): {np.mean(list(self._recent_rew)):.2f}")
            esc_rate = 100 * sum(self._recent_esc) / len(self._recent_esc)
            print(f"  Final escape rate:      {esc_rate:.1f}%")
        print(f"{'=' * 60}")
        super().close()
