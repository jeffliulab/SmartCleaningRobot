"""Simple random exploration policy for pipeline validation.

Drives the robot forward at a constant speed. When the LiDAR detects an
obstacle ahead (minimum distance below threshold), the robot turns a random
angle before resuming forward motion. No planning or path following -- just
"go straight, bump-and-turn".

This policy validates that the full pipeline works:
    - Robot moves in the scene
    - LiDAR detects obstacles
    - Coverage tracker updates
    - Camera captures data
    - Map viewer renders results
"""

from __future__ import annotations

import torch


class RandomExplorerPolicy:
    """Bump-and-turn random exploration policy.

    Action output: (linear_vel_normalized, angular_vel_normalized) in [-1, 1].
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        forward_speed: float = 0.7,
        turn_speed: float = 0.8,
        obstacle_threshold: float = 0.3,
        turn_duration: int = 15,
    ):
        self.num_envs = num_envs
        self.device = device
        self.forward_speed = forward_speed
        self.turn_speed = turn_speed
        self.obstacle_threshold = obstacle_threshold
        self.turn_duration = turn_duration

        self._turning = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._turn_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._turn_direction = torch.ones(num_envs, device=device)

    def compute_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute actions from observations.

        Expects the observation layout from Cleaningrobotv2Env:
            [0:3]   pose (x, y, yaw)
            [3:5]   velocity (vx, omega)
            [5:41]  lidar (36 normalized distances)
            [41:1065] local coverage map
            [1065]  coverage ratio

        Args:
            obs: (num_envs, obs_dim) observation tensor.

        Returns:
            (num_envs, 2) action tensor with (linear_vel, angular_vel) normalized.
        """
        lidar = obs[:, 5:41]

        # Check front-facing rays (indices around 0 = forward direction)
        front_rays = torch.cat([lidar[:, :3], lidar[:, -3:]], dim=-1)
        front_min = front_rays.min(dim=-1).values

        obstacle_detected = front_min < self.obstacle_threshold

        # Start turning for envs that detect obstacles and aren't already turning
        start_turn = obstacle_detected & ~self._turning
        if start_turn.any():
            self._turning[start_turn] = True
            self._turn_counter[start_turn] = self.turn_duration
            self._turn_direction[start_turn] = (
                torch.randint(0, 2, (start_turn.sum(),), device=self.device).float() * 2.0 - 1.0
            )

        actions = torch.zeros(self.num_envs, 2, device=self.device)

        # Turning envs: spin in place
        actions[self._turning, 0] = 0.0
        actions[self._turning, 1] = self._turn_direction[self._turning] * self.turn_speed

        # Forward envs: drive straight
        forward_mask = ~self._turning
        actions[forward_mask, 0] = self.forward_speed
        actions[forward_mask, 1] = 0.0

        # Decrement turn counter
        self._turn_counter[self._turning] -= 1
        done_turning = self._turning & (self._turn_counter <= 0)
        self._turning[done_turning] = False

        return actions

    def reset(self, env_ids: torch.Tensor | list[int]):
        """Reset policy state for the specified environments."""
        self._turning[env_ids] = False
        self._turn_counter[env_ids] = 0
