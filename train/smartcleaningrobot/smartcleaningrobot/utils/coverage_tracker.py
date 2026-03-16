"""GPU-accelerated coverage tracking for batched RL environments.

Maintains a 2D grid per environment tracking which cells have been
visited/cleaned by the robot. All operations use torch tensors on GPU
for efficient parallel computation across environments.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class CoverageTracker:
    """Tracks cleaning coverage across multiple parallel environments.

    The grid uses the convention:
        0.0 = uncleaned
        1.0 = cleaned
    """

    def __init__(
        self,
        grid_resolution: float,
        scene_bounds: tuple[float, float, float, float],
        robot_radius: float,
        num_envs: int,
        device: str | torch.device,
    ):
        """Initialize the coverage tracker.

        Args:
            grid_resolution: Size of each grid cell in meters.
            scene_bounds: (x_min, y_min, x_max, y_max) in world coordinates.
            robot_radius: Radius of the robot's cleaning footprint in meters.
            num_envs: Number of parallel environments.
            device: Torch device (cpu or cuda).
        """
        self.resolution = grid_resolution
        self.device = device
        self.num_envs = num_envs

        x_min, y_min, x_max, y_max = scene_bounds
        self.origin_x = x_min
        self.origin_y = y_min
        self.width = max(1, int((x_max - x_min) / grid_resolution))
        self.height = max(1, int((y_max - y_min) / grid_resolution))
        self.total_cells = self.width * self.height

        self.grid = torch.zeros(num_envs, self.height, self.width, device=device)
        self.robot_radius_cells = max(1, int(robot_radius / grid_resolution))

        self._precompute_footprint()

    def _precompute_footprint(self):
        """Pre-compute the relative cell offsets for the robot's circular footprint."""
        r = self.robot_radius_cells
        offsets = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    offsets.append((dy, dx))
        self._footprint_offsets = offsets

    def update(self, robot_positions: torch.Tensor) -> torch.Tensor:
        """Mark cells under the robot as cleaned.

        Args:
            robot_positions: (num_envs, 2) tensor with (x, y) world coordinates.

        Returns:
            (num_envs,) tensor with count of newly cleaned cells per environment.
        """
        prev_covered = self.grid.sum(dim=(-1, -2))

        gx = ((robot_positions[:, 0] - self.origin_x) / self.resolution).long()
        gy = ((robot_positions[:, 1] - self.origin_y) / self.resolution).long()

        batch_idx = torch.arange(self.num_envs, device=self.device)
        for dy, dx in self._footprint_offsets:
            y = (gy + dy).clamp(0, self.height - 1)
            x = (gx + dx).clamp(0, self.width - 1)
            self.grid[batch_idx, y, x] = 1.0

        new_covered = self.grid.sum(dim=(-1, -2))
        return new_covered - prev_covered

    def get_coverage_ratio(self) -> torch.Tensor:
        """Returns per-env coverage ratio in [0, 1]. Shape: (num_envs,)."""
        return self.grid.sum(dim=(-1, -2)) / self.total_cells

    def get_local_view(
        self, robot_positions: torch.Tensor, window_size: int = 32
    ) -> torch.Tensor:
        """Extract a local coverage map patch centered on the robot.

        Args:
            robot_positions: (num_envs, 2) tensor with (x, y) world coordinates.
            window_size: Side length of the square patch in grid cells.

        Returns:
            (num_envs, window_size, window_size) tensor.
        """
        half = window_size // 2

        gx = ((robot_positions[:, 0] - self.origin_x) / self.resolution).long()
        gy = ((robot_positions[:, 1] - self.origin_y) / self.resolution).long()

        padded = F.pad(self.grid, (half, half, half, half), value=0.0)

        gx_p = (gx + half).clamp(half, padded.shape[-1] - half)
        gy_p = (gy + half).clamp(half, padded.shape[-2] - half)

        dy = torch.arange(-half, half, device=self.device)
        dx = torch.arange(-half, half, device=self.device)
        dy_grid, dx_grid = torch.meshgrid(dy, dx, indexing="ij")

        n = self.num_envs
        batch_idx = torch.arange(n, device=self.device).view(-1, 1, 1).expand(n, window_size, window_size)
        y_idx = (gy_p.view(-1, 1, 1) + dy_grid.unsqueeze(0)).clamp(0, padded.shape[-2] - 1)
        x_idx = (gx_p.view(-1, 1, 1) + dx_grid.unsqueeze(0)).clamp(0, padded.shape[-1] - 1)

        return padded[batch_idx, y_idx, x_idx]

    def get_full_map(self) -> torch.Tensor:
        """Returns the full coverage grid. Shape: (num_envs, H, W)."""
        return self.grid

    def reset(self, env_ids: torch.Tensor | list[int]):
        """Reset coverage grid for the specified environments."""
        self.grid[env_ids] = 0.0
