"""Left-hand-rule wall follower for maze benchmark.

Ported from deploy/cleaning_robot_ros/.../baselines/wall_follower.py
with the same interface as other benchmark algorithms.

This is a **reactive** algorithm with **no map knowledge**.
It relies solely on LiDAR to maintain distance to the left wall.
"""

from __future__ import annotations

import math

import numpy as np


class WallFollowerPolicy:
    """Geometric left-hand-rule wall follower (F1TENTH method).

    Uses two LiDAR rays at 50 deg and 90 deg (indices 5 and 9 in the
    36-ray downsampled scan) to compute perpendicular distance and heading
    relative to the left wall, then tracks a desired following distance
    with a PD controller.
    """

    _ANGLE_A = math.radians(50)   # ray index 5 (front-left)
    _ANGLE_B = math.radians(90)   # ray index 9 (directly left)
    _IDX_A = 5
    _IDX_B = 9

    def __init__(
        self,
        desired_wall_dist: float = 0.15,
        wall_lost_thresh: float = 0.40,
        wall_found_thresh: float = 0.32,
        front_stop_dist: float = 0.18,
        forward_speed: float = 0.12,
        turn_speed: float = 0.5,
        lookahead: float = 0.10,
        kp: float = 3.5,
        lidar_max_range: float = 3.5,
    ):
        self._desired = desired_wall_dist
        self._lost_thresh = wall_lost_thresh
        self._found_thresh = wall_found_thresh
        self._front_stop = front_stop_dist
        self._fwd = forward_speed
        self._turn = turn_speed
        self._lookahead = lookahead
        self._kp = kp
        self._max_range = lidar_max_range

        self._swing = self._ANGLE_B - self._ANGLE_A
        self._cos_swing = math.cos(self._swing)
        self._sin_swing = math.sin(self._swing)

        self._turn_left_state = False
        self._wall_present = False

    def compute_action(self, lidar_normalized: np.ndarray) -> tuple[float, float]:
        """Return (linear_vel, angular_vel) from 36 normalized LiDAR rays.

        Ray layout (CCW from front, 10 deg spacing):
            idx  0 =   0 deg  (front)
            idx  5 =  50 deg  (front-left)
            idx  9 =  90 deg  (left)
            idx 18 = 180 deg  (rear)
            idx 27 = 270 deg  (right)
        """
        rays = lidar_normalized.astype(np.float64) * self._max_range
        rays[rays < 0.01] = self._max_range

        n = len(rays)

        # Front obstacle detection (±20 deg)
        front_min = float(min(rays[0], rays[1], rays[2], rays[n - 1], rays[n - 2]))
        front_blocked = front_min < self._front_stop

        # Left wall detection with hysteresis
        left_region = rays[self._IDX_A: self._IDX_B + 1]
        left_min = float(np.min(left_region))

        if self._wall_present:
            if left_min > self._lost_thresh:
                self._wall_present = False
        else:
            if left_min < self._found_thresh:
                self._wall_present = True

        # Override 1: front blocked
        if front_blocked:
            if self._turn_left_state:
                return 0.0, self._turn
            return 0.0, -self._turn

        # Override 2: left wall lost — commit to left turn
        if not self._wall_present:
            self._turn_left_state = True
            return self._fwd * 0.3, self._turn * 0.9

        # Normal: geometric wall following
        self._turn_left_state = False

        a = float(rays[self._IDX_A])
        b = float(rays[self._IDX_B])

        theta = math.atan2(a * self._cos_swing - b, a * self._sin_swing)
        d_wall = b * math.cos(theta)
        d_ahead = d_wall + self._lookahead * math.sin(theta)

        error = d_ahead - self._desired
        angular = self._kp * error
        angular = max(-1.2, min(1.2, angular))

        linear = self._fwd * max(0.25, 1.0 - 0.5 * abs(angular) / 1.2)

        return float(linear), float(angular)

    def reset(self) -> None:
        self._turn_left_state = False
        self._wall_present = False
