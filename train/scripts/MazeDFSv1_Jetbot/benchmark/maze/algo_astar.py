"""A* path planner + reactive waypoint follower for maze navigation.

This algorithm has **full knowledge of the maze map** (global planner) and
uses a pure-pursuit controller with LiDAR-based wall avoidance to follow
the planned path safely through narrow passages.

It represents the **optimal baseline with perfect information**.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from maze_sim import (
    CELL_SIZE,
    COLS,
    EXIT_CELL,
    ROWS,
    build_maze_grid,
    cell_to_world,
    world_to_cell,
)


# ---------------------------------------------------------------------------
# A* on the logical cell graph
# ---------------------------------------------------------------------------

def astar_path(
    grid: list[list[bool]],
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int] = EXIT_CELL,
) -> list[tuple[int, int]]:
    """Find shortest cell-path from start to goal using A*.

    Returns a list of (r, c) cells from start to goal (inclusive).
    """

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0
    heapq.heappush(open_set, (heuristic(start_cell, goal_cell), counter, start_cell))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start_cell: 0.0}

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == goal_cell:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        r, c = current
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            wall_r = 2 * r + 1 + dr
            wall_c = 2 * c + 1 + dc
            if grid[wall_r][wall_c]:
                continue

            tentative_g = g_score[current] + 1.0
            if tentative_g < g_score.get((nr, nc), float("inf")):
                came_from[(nr, nc)] = current
                g_score[(nr, nc)] = tentative_g
                f_score = tentative_g + heuristic((nr, nc), goal_cell)
                counter += 1
                heapq.heappush(open_set, (f_score, counter, (nr, nc)))

    return []


# ---------------------------------------------------------------------------
# A* policy with reactive obstacle avoidance
# ---------------------------------------------------------------------------

class AStarPolicy:
    """A* global planner + waypoint follower with LiDAR collision avoidance.

    The robot plans an optimal cell path, converts to waypoints, then uses:
    1. Pure pursuit to steer toward the next waypoint
    2. LiDAR-based repulsive field to stay away from walls
    This hybrid approach ensures smooth navigation through narrow passages.
    """

    def __init__(
        self,
        forward_speed: float = 0.25,
        kp_angular: float = 3.5,
        goal_tolerance: float = 0.15,
        wall_avoid_dist: float = 0.15,  # start avoiding at this LiDAR distance (normalized)
        wall_avoid_gain: float = 1.5,
    ):
        self._fwd = forward_speed
        self._kp = kp_angular
        self._goal_tol = goal_tolerance
        self._wall_avoid_dist = wall_avoid_dist
        self._wall_avoid_gain = wall_avoid_gain

        self._grid = build_maze_grid()
        self._waypoints: list[tuple[float, float]] = []
        self._wp_idx = 0

    def plan(self, start_x: float, start_y: float) -> list[tuple[float, float]]:
        """Compute A* path and convert to world-coordinate waypoints."""
        start_cell = world_to_cell(start_x, start_y)
        cell_path = astar_path(self._grid, start_cell, EXIT_CELL)

        # Convert to world coordinates (skip start cell to avoid self-targeting)
        self._waypoints = [cell_to_world(r, c) for r, c in cell_path]

        # Add exit position as final waypoint
        self._waypoints.append((0.0, -3.15))

        self._wp_idx = 0
        return self._waypoints

    def _compute_wall_avoidance(self, lidar_36: np.ndarray) -> float:
        """Compute angular correction from LiDAR to avoid walls.

        Looks at front-left and front-right sectors. If a wall is close on one
        side, steer away from it.
        """
        # Ray angles: idx * 10 degrees CCW from front
        # Front-right: indices 34, 35, 0, 1, 2  (~-20 to 20 deg, right-biased)
        # Front-left:  indices 2, 3, 4, 5, 6    (~20 to 60 deg)
        # Right side:  indices 30, 31, 32, 33, 34
        # Left side:   indices 6, 7, 8, 9, 10

        angular_correction = 0.0

        for idx in range(36):
            d = lidar_36[idx]
            if d < self._wall_avoid_dist:
                # This ray is close to a wall — push away
                ray_angle = idx * (2 * math.pi / 36)  # angle from front, CCW
                # Repulsive force perpendicular to ray direction (push away)
                strength = (self._wall_avoid_dist - d) / self._wall_avoid_dist
                # If wall is on the left (0 < angle < pi), steer right (negative angular)
                # If wall is on the right (pi < angle < 2pi), steer left (positive angular)
                if 0 < ray_angle < math.pi:
                    angular_correction -= strength * self._wall_avoid_gain
                elif math.pi < ray_angle < 2 * math.pi:
                    angular_correction += strength * self._wall_avoid_gain

        return angular_correction

    def compute_action(self, x: float, y: float, yaw: float,
                       lidar_normalized: np.ndarray | None = None) -> tuple[float, float]:
        """Return (linear_vel, angular_vel) to follow the planned path.

        Combines pure pursuit with reactive obstacle avoidance.
        """
        if not self._waypoints:
            return 0.0, 0.0

        # Advance waypoint index if close enough
        while self._wp_idx < len(self._waypoints) - 1:
            wx, wy = self._waypoints[self._wp_idx]
            dist = math.sqrt((x - wx) ** 2 + (y - wy) ** 2)
            if dist < self._goal_tol:
                self._wp_idx += 1
            else:
                break

        # Target the current waypoint
        tx, ty = self._waypoints[self._wp_idx]
        dx_w = tx - x
        dy_w = ty - y
        dist_to_target = math.sqrt(dx_w ** 2 + dy_w ** 2)

        if dist_to_target < 0.05 and self._wp_idx >= len(self._waypoints) - 1:
            return 0.0, 0.0

        # Pure pursuit steering
        target_angle = math.atan2(dy_w, dx_w)
        angle_err = math.atan2(math.sin(target_angle - yaw), math.cos(target_angle - yaw))

        angular = self._kp * angle_err

        # Add reactive wall avoidance from LiDAR
        if lidar_normalized is not None:
            wall_correction = self._compute_wall_avoidance(lidar_normalized)
            angular += wall_correction

        angular = max(-2.0, min(2.0, angular))

        # Speed control
        if abs(angle_err) > math.radians(45):
            # Large heading error — rotate more, move less
            linear = 0.03
        elif abs(angle_err) > math.radians(20):
            linear = self._fwd * 0.5
        else:
            linear = self._fwd

        # Slow down if front is blocked
        if lidar_normalized is not None:
            front_min = min(lidar_normalized[0], lidar_normalized[1],
                            lidar_normalized[35], lidar_normalized[34])
            if front_min < 0.06:  # ~0.21m
                linear = 0.0
            elif front_min < 0.12:
                linear *= 0.3

        return float(linear), float(angular)

    def reset(self) -> None:
        self._waypoints = []
        self._wp_idx = 0
