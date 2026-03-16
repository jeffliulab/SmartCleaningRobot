"""Build the 1066-dim observation vector from ROS messages.

The observation format MUST match the training environment exactly:
  [0:3]      x, y, yaw
  [3:5]      v_forward, omega
  [5:41]     lidar (36 downsampled rays, normalized)
  [41:1065]  local coverage map (32x32 patch, flattened)
  [1065]     global coverage ratio
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


class ObsBuilder:
    """Converts ROS sensor data into the observation tensor expected by the policy."""

    OBS_DIM = 1066

    def __init__(
        self,
        lidar_num_rays: int = 360,
        lidar_downsample_factor: int = 10,
        lidar_max_distance: float = 3.5,
        coverage_local_size: int = 32,
    ):
        self._lidar_factor = lidar_downsample_factor
        self._lidar_max = lidar_max_distance
        self._expected_lidar = lidar_num_rays
        self._coverage_local_size = coverage_local_size

        self._pose: Optional[np.ndarray] = None      # (x, y, yaw)
        self._vel: Optional[np.ndarray] = None        # (v_fwd, omega)
        self._lidar: Optional[np.ndarray] = None      # (36,)
        self._coverage_local: Optional[np.ndarray] = None  # (32*32,)
        self._coverage_ratio: float = 0.0

    def update_lidar(self, ranges: list[float] | np.ndarray) -> None:
        raw = np.array(ranges, dtype=np.float32)
        raw = np.clip(raw, 0.0, self._lidar_max)
        # Downsample: take every N-th ray
        downsampled = raw[:: self._lidar_factor][: self._expected_lidar // self._lidar_factor]
        self._lidar = downsampled / self._lidar_max  # normalize to [0, 1]

    def update_pose(self, pose_msg) -> None:
        """Extract (x, y, yaw) from geometry_msgs/Pose."""
        x = pose_msg.position.x
        y = pose_msg.position.y
        q = pose_msg.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._pose = np.array([x, y, yaw], dtype=np.float32)

    def update_velocity(self, twist_msg) -> None:
        """Extract (v_forward, omega) from geometry_msgs/Twist."""
        self._vel = np.array(
            [twist_msg.linear.x, twist_msg.angular.z],
            dtype=np.float32,
        )

    def update_coverage(self, local_patch: np.ndarray, ratio: float) -> None:
        self._coverage_local = local_patch.flatten().astype(np.float32)
        self._coverage_ratio = ratio

    def ready(self) -> bool:
        return self._pose is not None and self._vel is not None and self._lidar is not None

    def build(self) -> torch.Tensor:
        if self._coverage_local is None:
            self._coverage_local = np.zeros(self._coverage_local_size ** 2, dtype=np.float32)

        obs = np.concatenate([
            self._pose,                                             # 3
            self._vel,                                              # 2
            self._lidar,                                            # 36
            self._coverage_local,                                   # 1024
            np.array([self._coverage_ratio], dtype=np.float32),     # 1
        ])                                                          # total = 1066

        assert obs.shape[0] == self.OBS_DIM, f"obs dim mismatch: {obs.shape[0]} != {self.OBS_DIM}"
        return torch.from_numpy(obs).unsqueeze(0)  # (1, 1066)
