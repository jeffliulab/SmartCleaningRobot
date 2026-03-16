"""Evaluation metrics for cleaning performance.

Metrics:
    - coverage_ratio: fraction of free area covered [0, 1]
    - completion_time: time to reach coverage threshold (seconds)
    - path_length: total distance traveled (meters)
    - collision_count: number of collisions during episode
    - revisit_ratio: fraction of path spent on already-covered areas
"""

from __future__ import annotations

import torch


class EpisodeMetrics:
    """Tracks per-episode performance metrics across batched environments."""

    def __init__(self, num_envs: int, device: str | torch.device):
        self.num_envs = num_envs
        self.device = device
        self.reset(list(range(num_envs)))

    def update(
        self,
        robot_positions: torch.Tensor,
        coverage_ratios: torch.Tensor,
        collision_flags: torch.Tensor,
        newly_covered: torch.Tensor,
        dt: float,
    ):
        """Update metrics with current step data.

        Args:
            robot_positions: (N, 2) current robot (x, y) positions.
            coverage_ratios: (N,) current coverage ratios.
            collision_flags: (N,) boolean flags for collision detection.
            newly_covered: (N,) number of newly covered cells this step.
            dt: Simulation time step in seconds.
        """
        if self._prev_positions is not None:
            displacement = torch.norm(robot_positions - self._prev_positions, dim=-1)
            self.path_length += displacement
        self._prev_positions = robot_positions.clone()

        self.collision_count += collision_flags.long()
        self.elapsed_time += dt

        revisiting = (newly_covered == 0).float()
        self._revisit_steps += revisiting
        self._total_steps += 1.0

    def get_summary(self) -> dict[str, torch.Tensor]:
        """Return a dict of per-env metric tensors."""
        return {
            "path_length": self.path_length,
            "collision_count": self.collision_count,
            "elapsed_time": self.elapsed_time,
            "revisit_ratio": self._revisit_steps / self._total_steps.clamp(min=1.0),
        }

    def reset(self, env_ids: list[int]):
        """Reset metrics for the specified environments."""
        if not hasattr(self, "path_length"):
            self.path_length = torch.zeros(self.num_envs, device=self.device)
            self.collision_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.elapsed_time = torch.zeros(self.num_envs, device=self.device)
            self._revisit_steps = torch.zeros(self.num_envs, device=self.device)
            self._total_steps = torch.zeros(self.num_envs, device=self.device)
            self._prev_positions = None
            return

        self.path_length[env_ids] = 0.0
        self.collision_count[env_ids] = 0
        self.elapsed_time[env_ids] = 0.0
        self._revisit_steps[env_ids] = 0.0
        self._total_steps[env_ids] = 0.0
