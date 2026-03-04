"""F1TENTH-style left-hand-rule wall follower for maze navigation.

Uses two LiDAR rays at known angles (50 deg and 90 deg from forward)
to compute the robot's **true perpendicular distance** to the left wall
and its **heading angle** relative to that wall via trigonometry.
A single look-ahead PD controller tracks the desired following distance.

On top of the geometric follower, two overrides from V1 are preserved:
  1. Front blocked  -> stop and rotate (direction set by turn_left_state).
  2. Left wall gone  -> enter turn_left_state, rotate left to re-acquire.
Hysteresis on the wall-found/wall-lost transition prevents flip-flop.

Reference (V1): github.com/jeffliulab/brandeis_robot_lab_turtlebot3_project_codes
Reference (F1TENTH wall-follow lab): f1tenth.org/learn
"""

from __future__ import annotations

import math

import numpy as np


class WallFollowerPolicy:
    """Geometric left-hand-rule wall follower (F1TENTH method)."""

    # Angles of the two left-side rays used for geometry (radians)
    _ANGLE_A = math.radians(50)   # ray index 5  (front-left)
    _ANGLE_B = math.radians(90)   # ray index 9  (directly left)
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

        # pre-compute trig constants for the two-ray geometry
        self._swing = self._ANGLE_B - self._ANGLE_A          # 40 deg
        self._cos_swing = math.cos(self._swing)
        self._sin_swing = math.sin(self._swing)

        self._turn_left_state = False
        self._wall_present = False                            # hysteresis state

    # ------------------------------------------------------------------

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

        # --- front obstacle detection (±20 deg) ---
        front_min = float(min(rays[0], rays[1], rays[2], rays[n - 1], rays[n - 2]))
        front_blocked = front_min < self._front_stop

        # --- left wall detection with hysteresis ---
        left_region = rays[self._IDX_A : self._IDX_B + 1]    # indices 5..9
        left_min = float(np.min(left_region))

        if self._wall_present:
            if left_min > self._lost_thresh:
                self._wall_present = False
        else:
            if left_min < self._found_thresh:
                self._wall_present = True

        # === Override 1: front blocked ===
        if front_blocked:
            if self._turn_left_state:
                return 0.0, self._turn
            return 0.0, -self._turn

        # === Override 2: left wall lost — commit to left turn ===
        if not self._wall_present:
            self._turn_left_state = True
            return self._fwd * 0.3, self._turn * 0.9

        # === Normal: geometric wall following ===
        self._turn_left_state = False

        a = float(rays[self._IDX_A])          # distance at 50 deg
        b = float(rays[self._IDX_B])          # distance at 90 deg

        # heading angle of robot relative to wall (+ means nose toward wall)
        theta = math.atan2(a * self._cos_swing - b, a * self._sin_swing)

        # true perpendicular distance to wall
        d_wall = b * math.cos(theta)

        # look-ahead: predicted perpendicular distance a short distance ahead
        d_ahead = d_wall + self._lookahead * math.sin(theta)

        # PD error (positive = too far → steer left)
        error = d_ahead - self._desired

        angular = self._kp * error
        angular = max(-1.2, min(1.2, angular))

        # slow down proportionally to turn magnitude
        linear = self._fwd * max(0.25, 1.0 - 0.5 * abs(angular) / 1.2)

        return float(linear), float(angular)

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._turn_left_state = False
        self._wall_present = False
