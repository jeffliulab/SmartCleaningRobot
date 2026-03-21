"""Lightweight 2D maze simulator for algorithm benchmarking.

Shares the same DFS maze generation (seed=42, 8x8) and differential-drive
kinematics as the Isaac Lab training environment, but runs in pure Python/NumPy
with no GPU or physics engine dependency.

Public API
----------
MazeSim           -- step-based 2D simulator with raycasting LiDAR
MazeSim.step()    -- advance one time step given (linear_vel, angular_vel)
MazeSim.reset()   -- reset robot to start position
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Maze parameters — MUST match source/.../maze.py
# ---------------------------------------------------------------------------
ROWS, COLS = 8, 8
CELL_SIZE = 0.45
WALL_H = 0.3
SEED = 42

_CX_OFF = 2 * (ROWS // 2) + 1  # 9
_CY_OFF = 2 * (COLS // 2) + 1  # 9

EXIT_CELL: tuple[int, int] = (ROWS - 1, COLS // 2)  # (7, 4)

# Robot kinematics — MUST match mazenav_env_cfg.py
MAX_LINEAR_VEL = 0.3   # m/s
MAX_ANGULAR_VEL = 2.0  # rad/s
WHEEL_RADIUS = 0.0325  # m
WHEEL_SEP = 0.118      # m
ROBOT_RADIUS = 0.06    # m (approximate collision radius)

# Simulation
SIM_DT = 1.0 / 60      # 60 Hz control (120 Hz sim / decimation=2)
LIDAR_MAX_RANGE = 3.5   # m
LIDAR_NUM_RAYS = 360
LIDAR_DOWNSAMPLE = 36   # every 10th ray

# Goal
EXIT_POS = (0.0, -3.15)
EXIT_THRESHOLD = 0.3
MAZE_DIAGONAL = 5.7
MAX_EPISODE_TIME = 300.0  # seconds (A* optimal ~142s, wall follower >>300s)


# ---------------------------------------------------------------------------
# Maze generation (identical to maze.py)
# ---------------------------------------------------------------------------

def _generate_dfs_maze(
    rows: int, cols: int, start: tuple[int, int], rng: random.Random
) -> list[list[bool]]:
    h = 2 * rows + 1
    w = 2 * cols + 1
    grid: list[list[bool]] = [[True] * w for _ in range(h)]

    sr, sc = start
    stack = [(sr, sc)]
    visited = {(sr, sc)}
    grid[2 * sr + 1][2 * sc + 1] = False

    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                neighbors.append((nr, nc, dr, dc))
        if neighbors:
            nr, nc, dr, dc = rng.choice(neighbors)
            grid[2 * r + 1 + dr][2 * c + 1 + dc] = False
            grid[2 * nr + 1][2 * nc + 1] = False
            visited.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()
    return grid


def _ensure_center_clear(grid: list[list[bool]], rows: int, cols: int) -> None:
    cr, cc = rows // 2, cols // 2
    grid[2 * cr + 1][2 * cc + 1] = False
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = cr + dr, cc + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if not grid[2 * nr + 1][2 * nc + 1]:
                grid[2 * cr + 1 + dr][2 * cc + 1 + dc] = False
                break


def build_maze_grid() -> list[list[bool]]:
    """Generate the canonical maze grid. True = wall, False = passable."""
    rng = random.Random(SEED)
    grid = _generate_dfs_maze(ROWS, COLS, start=(ROWS // 2, COLS // 2), rng=rng)
    h = 2 * ROWS + 1
    grid[h - 1][2 * (COLS // 2) + 1] = False  # open exit at south center
    _ensure_center_clear(grid, ROWS, COLS)
    return grid


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def cell_to_world(r: int, c: int) -> tuple[float, float]:
    wx = (2 * c + 1 - _CX_OFF) * CELL_SIZE
    wy = (_CY_OFF - (2 * r + 1)) * CELL_SIZE
    return (wx, wy)


def world_to_cell(x: float, y: float) -> tuple[int, int]:
    c = round(x / (2 * CELL_SIZE) + COLS // 2)
    r = round(ROWS // 2 - y / (2 * CELL_SIZE))
    c = max(0, min(COLS - 1, c))
    r = max(0, min(ROWS - 1, r))
    return (r, c)


def world_to_grid(x: float, y: float) -> tuple[int, int]:
    """Convert world (x, y) to grid indices (row, col) in the full 17x17 grid."""
    col = round(x / CELL_SIZE + _CX_OFF)
    row = round(_CY_OFF - y / CELL_SIZE)
    return (row, col)


# ---------------------------------------------------------------------------
# BFS distance field
# ---------------------------------------------------------------------------

def compute_bfs_distance_field(grid: list[list[bool]]) -> list[list[int]]:
    """BFS from exit cell, return ROWS x COLS distance grid."""
    distances: list[list[int]] = [[-1] * COLS for _ in range(ROWS)]
    start = EXIT_CELL
    distances[start[0]][start[1]] = 0
    queue: deque[tuple[int, int]] = deque([start])

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and distances[nr][nc] == -1:
                wall_r = 2 * r + 1 + dr
                wall_c = 2 * c + 1 + dc
                if not grid[wall_r][wall_c]:
                    distances[nr][nc] = distances[r][c] + 1
                    queue.append((nr, nc))
    return distances


# ---------------------------------------------------------------------------
# 2D Raycasting LiDAR — DDA grid traversal (fast)
# ---------------------------------------------------------------------------

class LiDAR2D:
    """Fast 2D raycaster using DDA (Digital Differential Analyzer) on the grid.

    Instead of testing every wall segment, we step through grid cells along
    each ray direction and stop at the first wall cell.  O(rays * cells_per_ray)
    which is much faster than O(rays * total_segments).
    """

    def __init__(self, grid: list[list[bool]], max_range: float = LIDAR_MAX_RANGE):
        self.max_range = max_range
        self._grid = grid
        self._h = len(grid)
        self._w = len(grid[0])
        # Pre-compute grid bounds in world coordinates
        # Grid cell (row, col) has center at world:
        #   wx = (col - _CX_OFF) * CELL_SIZE
        #   wy = (_CY_OFF - row) * CELL_SIZE
        # Grid cell edges span ±CELL_SIZE/2 from center
        # Grid origin (row=0, col=0) top-left corner in world:
        self._grid_x_min = (0 - _CX_OFF) * CELL_SIZE - CELL_SIZE / 2
        self._grid_y_max = (_CY_OFF - 0) * CELL_SIZE + CELL_SIZE / 2

    def _world_to_grid_continuous(self, x: float, y: float) -> tuple[float, float]:
        """Convert world (x, y) to continuous grid (col_f, row_f)."""
        col_f = (x - self._grid_x_min) / CELL_SIZE
        row_f = (self._grid_y_max - y) / CELL_SIZE
        return col_f, row_f

    def _cast_ray_dda(self, x: float, y: float, dx: float, dy: float) -> float:
        """Cast a single ray using DDA grid traversal. Returns distance to wall."""
        col_f, row_f = self._world_to_grid_continuous(x, y)

        # Current grid cell
        col_i = int(math.floor(col_f))
        row_i = int(math.floor(row_f))

        # Ray direction in grid space: +col = +x, +row = -y
        d_col = dx / CELL_SIZE
        d_row = -dy / CELL_SIZE

        # Step direction
        step_col = 1 if d_col >= 0 else -1
        step_row = 1 if d_row >= 0 else -1

        # Distance in world units to cross one full grid cell
        if abs(dx) > 1e-12:
            t_delta_col = CELL_SIZE / abs(dx)
        else:
            t_delta_col = 1e18

        if abs(dy) > 1e-12:
            t_delta_row = CELL_SIZE / abs(dy)
        else:
            t_delta_row = 1e18

        # Distance to next grid boundary
        if d_col >= 0:
            t_max_col = ((col_i + 1) - col_f) * CELL_SIZE / abs(dx) if abs(dx) > 1e-12 else 1e18
        else:
            t_max_col = (col_f - col_i) * CELL_SIZE / abs(dx) if abs(dx) > 1e-12 else 1e18

        if d_row >= 0:
            t_max_row = ((row_i + 1) - row_f) * CELL_SIZE / abs(dy) if abs(dy) > 1e-12 else 1e18
        else:
            t_max_row = (row_f - row_i) * CELL_SIZE / abs(dy) if abs(dy) > 1e-12 else 1e18

        t = 0.0
        max_steps = int(self.max_range / CELL_SIZE) + 5

        for _ in range(max_steps):
            # Check current cell
            if 0 <= row_i < self._h and 0 <= col_i < self._w:
                if self._grid[row_i][col_i]:
                    # Hit a wall — return distance to the entry point of this cell
                    return max(t, 0.0)
            else:
                # Out of bounds
                return self.max_range

            # Advance to next cell boundary
            if t_max_col < t_max_row:
                t = t_max_col
                t_max_col += t_delta_col
                col_i += step_col
            else:
                t = t_max_row
                t_max_row += t_delta_row
                row_i += step_row

            if t > self.max_range:
                return self.max_range

        return self.max_range

    def scan(self, x: float, y: float, yaw: float, num_rays: int = LIDAR_NUM_RAYS) -> np.ndarray:
        """Cast rays from (x, y, yaw) and return distances array."""
        distances = np.full(num_rays, self.max_range, dtype=np.float64)
        angle_step = 2 * math.pi / num_rays

        for i in range(num_rays):
            ray_angle = yaw + i * angle_step
            dx = math.cos(ray_angle)
            dy = math.sin(ray_angle)
            distances[i] = self._cast_ray_dda(x, y, dx, dy)

        return distances


# ---------------------------------------------------------------------------
# Collision checker
# ---------------------------------------------------------------------------

def is_collision(grid: list[list[bool]], x: float, y: float, radius: float = ROBOT_RADIUS) -> bool:
    """Check if a circle at (x,y) with given radius overlaps any wall cell."""
    h = len(grid)
    w = len(grid[0])

    # Check grid cells that could overlap with the robot
    # Convert robot bounds to grid indices
    for check_x in [x - radius, x, x + radius]:
        for check_y in [y - radius, y, y + radius]:
            grow, gcol = world_to_grid(check_x, check_y)
            if 0 <= grow < h and 0 <= gcol < w:
                if grid[grow][gcol]:
                    # Wall cell — check actual distance
                    cx = (gcol - _CX_OFF) * CELL_SIZE
                    cy = (_CY_OFF - grow) * CELL_SIZE
                    half = CELL_SIZE / 2.0
                    # Closest point on AABB to robot center
                    closest_x = max(cx - half, min(x, cx + half))
                    closest_y = max(cy - half, min(y, cy + half))
                    dist = math.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)
                    if dist < radius:
                        return True
    return False


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Return value of MazeSim.step()."""
    x: float
    y: float
    yaw: float
    lidar_36: np.ndarray       # 36-dim normalized [0, 1]
    exit_dir_body: np.ndarray  # (sin(θ), cos(θ))
    exit_dist_norm: float      # dist / maze_diagonal
    v_linear: float
    omega: float
    time: float
    done: bool
    escaped: bool
    collision: bool


@dataclass
class Metrics:
    """Episode metrics."""
    escaped: bool = False
    total_time: float = 0.0
    path_length: float = 0.0
    num_collisions: int = 0
    cells_visited: int = 0
    steps: int = 0
    trajectory: list[tuple[float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

class MazeSim:
    """Lightweight 2D maze simulator matching Isaac Lab MazeEscapeEnv kinematics."""

    def __init__(self, dt: float = SIM_DT, start_pos: tuple[float, float] | None = None,
                 start_yaw: float | None = None):
        self.dt = dt
        self.grid = build_maze_grid()
        self.bfs_field = compute_bfs_distance_field(self.grid)
        self.lidar = LiDAR2D(self.grid)

        self._start_pos = start_pos or (0.0, 0.0)  # center cell
        self._start_yaw = start_yaw if start_yaw is not None else 0.0

        self.reset()

    def reset(self, start_pos: tuple[float, float] | None = None,
              start_yaw: float | None = None) -> StepResult:
        """Reset robot to start position."""
        pos = start_pos or self._start_pos
        self.x, self.y = pos
        self.yaw = start_yaw if start_yaw is not None else self._start_yaw
        self.time = 0.0
        self.metrics = Metrics()
        self._visited_cells: set[tuple[int, int]] = set()
        self._visited_cells.add(world_to_cell(self.x, self.y))
        self.metrics.trajectory.append((self.x, self.y))
        return self._observe()

    def step(self, linear_vel: float, angular_vel: float) -> StepResult:
        """Advance one timestep with differential-drive kinematics.

        Args:
            linear_vel: desired forward velocity in m/s (clamped to ±MAX_LINEAR_VEL)
            angular_vel: desired angular velocity in rad/s (clamped to ±MAX_ANGULAR_VEL)
        """
        # Clamp velocities
        v = max(-MAX_LINEAR_VEL, min(MAX_LINEAR_VEL, linear_vel))
        w = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, angular_vel))

        # Kinematics update
        new_yaw = self.yaw + w * self.dt
        new_x = self.x + v * math.cos(new_yaw) * self.dt
        new_y = self.y + v * math.sin(new_yaw) * self.dt

        # Collision check — if collision, don't move but record it
        collision = is_collision(self.grid, new_x, new_y)
        if collision:
            self.metrics.num_collisions += 1
            # Slide along: try only rotation
            self.yaw = new_yaw
        else:
            dx = new_x - self.x
            dy = new_y - self.y
            self.metrics.path_length += math.sqrt(dx * dx + dy * dy)
            self.x = new_x
            self.y = new_y
            self.yaw = new_yaw

        self.time += self.dt
        self.metrics.total_time = self.time
        self.metrics.steps += 1

        # Track visited cells
        cell = world_to_cell(self.x, self.y)
        self._visited_cells.add(cell)
        self.metrics.cells_visited = len(self._visited_cells)
        self.metrics.trajectory.append((self.x, self.y))

        return self._observe()

    def _observe(self) -> StepResult:
        """Build observation matching the 41-dim RL observation space."""
        # LiDAR
        raw_lidar = self.lidar.scan(self.x, self.y, self.yaw)
        lidar_36 = raw_lidar[::10]  # downsample 360 -> 36
        lidar_36 = np.clip(lidar_36, 0.0, LIDAR_MAX_RANGE) / LIDAR_MAX_RANGE

        # Exit direction in body frame
        dx_w = EXIT_POS[0] - self.x
        dy_w = EXIT_POS[1] - self.y
        dx_b = dx_w * math.cos(self.yaw) + dy_w * math.sin(self.yaw)
        dy_b = -dx_w * math.sin(self.yaw) + dy_w * math.cos(self.yaw)
        angle = math.atan2(dy_b, dx_b)
        exit_dir = np.array([math.sin(angle), math.cos(angle)])

        dist = math.sqrt(dx_w ** 2 + dy_w ** 2)
        exit_dist_norm = dist / MAZE_DIAGONAL

        # Check termination
        escaped = dist < EXIT_THRESHOLD
        done = escaped or self.time >= MAX_EPISODE_TIME

        if escaped:
            self.metrics.escaped = True

        collision = is_collision(self.grid, self.x, self.y)

        return StepResult(
            x=self.x, y=self.y, yaw=self.yaw,
            lidar_36=lidar_36,
            exit_dir_body=exit_dir,
            exit_dist_norm=exit_dist_norm,
            v_linear=0.0, omega=0.0,
            time=self.time,
            done=done, escaped=escaped, collision=collision,
        )

    def get_obs_vector(self, result: StepResult, v: float = 0.0, w: float = 0.0) -> np.ndarray:
        """Build the full 41-dim observation vector matching RL env format."""
        obs = np.zeros(41, dtype=np.float32)
        obs[0] = v / MAX_LINEAR_VEL
        obs[1] = w / MAX_ANGULAR_VEL
        obs[2:38] = result.lidar_36.astype(np.float32)
        obs[38:40] = result.exit_dir_body.astype(np.float32)
        obs[40] = result.exit_dist_norm
        return obs
