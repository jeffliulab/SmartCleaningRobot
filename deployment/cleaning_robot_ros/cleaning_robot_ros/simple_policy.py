"""Simple bump-and-turn reactive policy for deployment pipeline validation.

This policy uses only LiDAR data to navigate:
  1. Move forward when the path is clear
  2. When an obstacle is detected in front, stop and turn
  3. Turn toward the side with more free space
  4. Add a slight spiral bias to improve floor coverage

No RL model is required — this lets you validate the full ROS 2
deployment pipeline (Gazebo → sensors → policy_node → /cmd_vel)
before the RL model is trained.
"""

from __future__ import annotations

import random

import numpy as np


class SimplePolicy:
    """Reactive bump-and-turn cleaning policy."""

    _FORWARD = 0
    _TURNING = 1

    def __init__(
        self,
        forward_speed: float = 0.15,
        turn_speed: float = 1.2,
        obstacle_threshold: float = 0.25,
        clear_threshold: float = 0.40,
        turn_steps: int = 15,
        spiral_bias: float = 0.08,
    ):
        self._fwd = forward_speed
        self._turn = turn_speed
        self._obs_thresh = obstacle_threshold
        self._clr_thresh = clear_threshold
        self._turn_steps_max = turn_steps
        self._spiral = spiral_bias

        self._state = self._FORWARD
        self._steps_in_turn = 0
        self._turn_dir = 1.0

    def compute_action(self, lidar_normalized: np.ndarray) -> tuple[float, float]:
        """Compute (linear_vel, angular_vel) from 36 normalized LiDAR rays.

        Args:
            lidar_normalized: 36 values in [0, 1], where 0 = contact, 1 = max range.
                              Ray 0 points forward, indexed counter-clockwise.

        Returns:
            (linear_vel m/s, angular_vel rad/s) — ready for /cmd_vel.
        """
        n = len(lidar_normalized)

        front_idx = list(range(n - 3, n)) + list(range(0, 4))
        front_min = min(lidar_normalized[i] for i in front_idx)

        left_avg = float(np.mean(lidar_normalized[n // 6 : n // 3]))
        right_avg = float(np.mean(lidar_normalized[2 * n // 3 : 5 * n // 6]))

        if self._state == self._FORWARD:
            if front_min < self._obs_thresh:
                self._state = self._TURNING
                self._steps_in_turn = 0
                self._turn_dir = 1.0 if left_avg > right_avg else -1.0
                if abs(left_avg - right_avg) < 0.05:
                    self._turn_dir = random.choice([-1.0, 1.0])
                return 0.0, self._turn_dir * self._turn

            return self._fwd, self._spiral

        self._steps_in_turn += 1
        if self._steps_in_turn >= self._turn_steps_max and front_min > self._clr_thresh:
            self._state = self._FORWARD
            return self._fwd, 0.0

        return 0.0, self._turn_dir * self._turn

    def reset(self) -> None:
        self._state = self._FORWARD
        self._steps_in_turn = 0
