"""Build the 41-dim observation vector for the maze escape task.

The observation is in **body frame** and matches the training environment:
  [0:2]   v_forward, omega              (body-frame velocity)
  [2:38]  lidar (36 downsampled rays, normalized to [0,1])
  [38:40] exit_dir_body (sin, cos)      (exit bearing in body frame)
  [40]    exit_dist_norm                 (normalised distance to exit)

This builder is separate from ObsBuilder (1066-dim cleaning task) to keep
the two tasks independent.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


class MazeObsBuilder:
    """Converts ROS sensor data into the 41-dim maze observation tensor."""

    OBS_DIM = 41

    def __init__(
        self,
        exit_x: float = 0.0,
        exit_y: float = -3.15,
        maze_diagonal: float = 5.7,
        lidar_num_rays: int = 360,
        lidar_downsample_factor: int = 10,
        lidar_max_distance: float = 3.5,
    ):
        self._exit = np.array([exit_x, exit_y], dtype=np.float32)
        self._maze_diag = maze_diagonal
        self._lidar_factor = lidar_downsample_factor
        self._lidar_max = lidar_max_distance
        self._expected_lidar = lidar_num_rays

        self._vel: Optional[np.ndarray] = None       # (v_fwd, omega)
        self._lidar: Optional[np.ndarray] = None      # (36,)
        self._pose_x: float = 0.0
        self._pose_y: float = 0.0
        self._yaw: float = 0.0

    # ------------------------------------------------------------------ updates

    def update_lidar(self, ranges: list[float] | np.ndarray) -> None:
        raw = np.array(ranges, dtype=np.float32)
        raw = np.clip(raw, 0.0, self._lidar_max)
        downsampled = raw[:: self._lidar_factor][: self._expected_lidar // self._lidar_factor]
        self._lidar = downsampled / self._lidar_max

    def update_pose(self, pose_msg) -> None:
        """Extract (x, y, yaw) from geometry_msgs/Pose."""
        self._pose_x = pose_msg.position.x
        self._pose_y = pose_msg.position.y
        q = pose_msg.orientation
        self._yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def update_velocity(self, twist_msg) -> None:
        """Extract (v_forward, omega) from geometry_msgs/Twist."""
        self._vel = np.array(
            [twist_msg.linear.x, twist_msg.angular.z],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ build

    def ready(self) -> bool:
        return self._vel is not None and self._lidar is not None

    def build(self) -> torch.Tensor:
        """Assemble the 41-dim observation in body frame."""
        # Exit vector in world frame
        dx_w = self._exit[0] - self._pose_x
        dy_w = self._exit[1] - self._pose_y

        # Rotate to body frame
        cos_yaw = math.cos(self._yaw)
        sin_yaw = math.sin(self._yaw)
        dx_b = dx_w * cos_yaw + dy_w * sin_yaw
        dy_b = -dx_w * sin_yaw + dy_w * cos_yaw

        # Polar representation in body frame
        angle = math.atan2(dy_b, dx_b)
        exit_dir = np.array([math.sin(angle), math.cos(angle)], dtype=np.float32)

        dist = math.sqrt(dx_w * dx_w + dy_w * dy_w)
        exit_dist_norm = np.array(
            [max(dist, 1e-4) / self._maze_diag], dtype=np.float32
        )

        obs = np.concatenate([
            self._vel,           # 2
            self._lidar,         # 36
            exit_dir,            # 2
            exit_dist_norm,      # 1
        ])                       # total = 41

        assert obs.shape[0] == self.OBS_DIM, f"obs dim mismatch: {obs.shape[0]} != {self.OBS_DIM}"
        return torch.from_numpy(obs).unsqueeze(0)  # (1, 41)
