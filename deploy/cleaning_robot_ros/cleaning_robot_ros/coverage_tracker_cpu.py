"""CPU/numpy version of the CoverageTracker for single-robot deployment.

Maintains the same grid logic as the GPU version used in training
(train/smartcleaningrobot/smartcleaningrobot/utils/coverage_tracker.py), but:
  - No batch dimension (single robot)
  - numpy instead of torch
  - No GPU requirement
"""

from __future__ import annotations

import numpy as np


class CoverageTrackerCPU:
    """Track cleaning coverage on a 2D occupancy grid using numpy."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        resolution: float = 0.05,
        cleaning_radius: float = 0.06,
    ):
        self.x_min = x_min
        self.y_min = y_min
        self.resolution = resolution
        self.cleaning_radius = cleaning_radius

        self.width = int((x_max - x_min) / resolution)
        self.height = int((y_max - y_min) / resolution)
        self.radius_cells = int(cleaning_radius / resolution)

        self.grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.total_cells = self.width * self.height

    def update(self, x: float, y: float) -> None:
        cx = int((x - self.x_min) / self.resolution)
        cy = int((y - self.y_min) / self.resolution)

        r = self.radius_cells
        y_lo = max(0, cy - r)
        y_hi = min(self.height, cy + r + 1)
        x_lo = max(0, cx - r)
        x_hi = min(self.width, cx + r + 1)

        self.grid[y_lo:y_hi, x_lo:x_hi] = 1.0

    def get_coverage_ratio(self) -> float:
        return float(np.sum(self.grid)) / self.total_cells

    def get_local_view(self, x: float, y: float, half_size: int = 16) -> np.ndarray:
        """Extract a (2*half_size x 2*half_size) patch around the robot."""
        cx = int((x - self.x_min) / self.resolution)
        cy = int((y - self.y_min) / self.resolution)

        patch = np.zeros((half_size * 2, half_size * 2), dtype=np.float32)

        src_y_lo = max(0, cy - half_size)
        src_y_hi = min(self.height, cy + half_size)
        src_x_lo = max(0, cx - half_size)
        src_x_hi = min(self.width, cx + half_size)

        dst_y_lo = half_size - (cy - src_y_lo)
        dst_y_hi = dst_y_lo + (src_y_hi - src_y_lo)
        dst_x_lo = half_size - (cx - src_x_lo)
        dst_x_hi = dst_x_lo + (src_x_hi - src_x_lo)

        patch[dst_y_lo:dst_y_hi, dst_x_lo:dst_x_hi] = self.grid[src_y_lo:src_y_hi, src_x_lo:src_x_hi]
        return patch

    def reset(self) -> None:
        self.grid[:] = 0.0
