"""Maze escape RL environment.

The robot (JetBot) must navigate a procedural DFS maze and reach the
exit in minimum time.  This task is entirely separate from the coverage
/cleaning task and serves as an RL learning exercise.

Observations (41 dims, **body frame**):
    velocity(2) + LiDAR(36) + exit_dir_body(2) + exit_distance(1)

Rewards:
    - Dense:  BFS-distance reduction toward exit each step
    - Sparse: large bonus on reaching the exit
    - Penalties: collision, time
    - Small exploration bonus for visiting new cells
    - Forward motion bonus to discourage spinning in place
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import RayCaster
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from smartcleaningrobot.robot_assets.jetbot import diff_drive_forward
from smartcleaningrobot.scenes import AVAILABLE_SCENES, build_scene
from smartcleaningrobot.scenes.builders.maze import (
    CELL_SIZE,
    COLS,
    ROWS,
    compute_bfs_distance_field,
    get_valid_cell_positions,
)
from smartcleaningrobot.utils.coverage_tracker import CoverageTracker

from .env_cfg import MazeEscapeEnvCfg


class MazeEscapeEnv(DirectRLEnv):
    """RL environment: escape a DFS maze as fast as possible."""

    cfg: MazeEscapeEnvCfg

    def __init__(self, cfg: MazeEscapeEnvCfg, render_mode: str | None = None, **kwargs):
        self.scene_config = AVAILABLE_SCENES[cfg.scene_name]
        self._valid_spawn_positions = self._precompute_spawn_positions()
        super().__init__(cfg, render_mode, **kwargs)

        self._wheel_joint_ids, _ = self.robot.find_joints(
            ["left_wheel_joint", "right_wheel_joint"]
        )

        self.coverage_tracker = CoverageTracker(
            grid_resolution=self.cfg.coverage_grid_resolution,
            scene_bounds=self.scene_config.scene_bounds,
            robot_radius=self.cfg.robot_cleaning_radius,
            num_envs=self.num_envs,
            device=self.device,
        )
        self.newly_covered = torch.zeros(self.num_envs, device=self.device)

        self._exit_pos = torch.tensor(
            [self.cfg.exit_pos[0], self.cfg.exit_pos[1]],
            dtype=torch.float32,
            device=self.device,
        )

        # BFS distance field for reward shaping
        bfs_field = compute_bfs_distance_field()
        self._bfs_grid = torch.tensor(bfs_field, dtype=torch.float32, device=self.device)
        self._bfs_max = self._bfs_grid.max().clamp(min=1.0)
        self._prev_bfs_dist = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------
    # Spawn position pre-computation
    # ------------------------------------------------------------------

    @staticmethod
    def _precompute_spawn_positions() -> list[tuple[float, float]]:
        """Return world-coordinate positions of every passable maze cell."""
        return get_valid_cell_positions()

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        build_scene(self.scene_config)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.lidar = RayCaster(self.cfg.lidar_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/Scene"])

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["lidar"] = self.lidar

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Actions (identical to coverage env)
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        linear_vel = self.actions[:, 0] * self.cfg.max_linear_vel
        angular_vel = self.actions[:, 1] * self.cfg.max_angular_vel
        left_vel, right_vel = diff_drive_forward(linear_vel, angular_vel)
        self.wheel_velocities = torch.stack([left_vel, right_vel], dim=-1)

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(
            self.wheel_velocities, joint_ids=self._wheel_joint_ids
        )

    # ------------------------------------------------------------------
    # Observations  (41 dims, body frame)
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w

        yaw = 2.0 * torch.atan2(root_quat[:, 3], root_quat[:, 0])

        # Body-frame velocity
        lin_vel = self.robot.data.root_com_lin_vel_b[:, 0:1]
        ang_vel = self.robot.data.root_com_ang_vel_b[:, 2:3]
        vel = torch.cat([lin_vel, ang_vel], dim=-1)  # (N, 2)

        # LiDAR: 360 -> 36 rays, normalized to [0, 1]
        sensor_pos = self.lidar.data.pos_w.unsqueeze(1)
        ray_hits = self.lidar.data.ray_hits_w
        lidar_distances = torch.norm(ray_hits - sensor_pos, dim=-1)
        lidar_obs = lidar_distances[:, ::10]
        lidar_obs = lidar_obs.clamp(0.0, self.cfg.lidar_cfg.max_distance) / self.cfg.lidar_cfg.max_distance
        # (N, 36)

        # Exit direction and distance in BODY FRAME
        robot_xy = root_pos[:, :2] - self.scene.env_origins[:, :2]
        exit_rel_world = self._exit_pos.unsqueeze(0) - robot_xy

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        exit_x_body = exit_rel_world[:, 0] * cos_yaw + exit_rel_world[:, 1] * sin_yaw
        exit_y_body = -exit_rel_world[:, 0] * sin_yaw + exit_rel_world[:, 1] * cos_yaw

        exit_angle = torch.atan2(exit_y_body, exit_x_body)
        exit_dir_body = torch.stack(
            [torch.sin(exit_angle), torch.cos(exit_angle)], dim=-1
        )  # (N, 2)

        exit_dist = torch.norm(exit_rel_world, dim=-1, keepdim=True).clamp(min=1e-4)
        exit_dist_norm = exit_dist / self.cfg.maze_diagonal  # (N, 1)

        # Update coverage tracker (for exploration bonus in reward)
        self.newly_covered = self.coverage_tracker.update(robot_xy)

        obs = torch.cat([vel, lidar_obs, exit_dir_body, exit_dist_norm], dim=-1)
        return {"policy": obs}

    # ------------------------------------------------------------------
    # BFS cell lookup helper
    # ------------------------------------------------------------------

    def _xy_to_bfs_dist(self, xy: torch.Tensor) -> torch.Tensor:
        """Convert local (x, y) positions to BFS distances via grid lookup."""
        cell_c = torch.round(xy[:, 0] / (2 * CELL_SIZE) + COLS // 2).long()
        cell_r = torch.round(ROWS // 2 - xy[:, 1] / (2 * CELL_SIZE)).long()
        cell_c = cell_c.clamp(0, COLS - 1)
        cell_r = cell_r.clamp(0, ROWS - 1)
        return self._bfs_grid[cell_r, cell_c]

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        root_pos = self.robot.data.root_pos_w
        robot_xy = root_pos[:, :2] - self.scene.env_origins[:, :2]

        # 1) Dense: BFS distance reduction (positive when getting closer via maze path)
        curr_bfs = self._xy_to_bfs_dist(robot_xy)
        bfs_reduction = (self._prev_bfs_dist - curr_bfs) / self._bfs_max
        dist_reward = self.cfg.rew_dist_reduction * bfs_reduction

        # 2) Sparse: exit reached bonus
        curr_eucl = torch.norm(robot_xy - self._exit_pos.unsqueeze(0), dim=-1)
        reached = (curr_eucl < self.cfg.exit_threshold).float()
        exit_bonus = self.cfg.rew_exit_bonus * reached

        # 3) Collision penalty
        sensor_pos = self.lidar.data.pos_w.unsqueeze(1)
        ray_hits = self.lidar.data.ray_hits_w
        lidar_distances = torch.norm(ray_hits - sensor_pos, dim=-1)
        lidar_min = lidar_distances.min(dim=-1).values
        collision = (lidar_min < self.cfg.collision_threshold).float()
        collision_penalty = self.cfg.rew_collision_penalty * collision

        # 4) Time penalty (per step)
        time_penalty = self.cfg.rew_time_penalty * torch.ones(
            self.num_envs, device=self.device
        )

        # 5) Small exploration bonus
        explore_bonus = self.cfg.rew_exploration * self.newly_covered

        # 6) Forward motion bonus (discourage spinning in place)
        forward_vel = self.robot.data.root_com_lin_vel_b[:, 0].clamp(min=0.0)
        forward_bonus = self.cfg.rew_forward_bonus * forward_vel

        self._prev_bfs_dist = curr_bfs.clone()

        return (
            dist_reward
            + exit_bonus
            + collision_penalty
            + time_penalty
            + explore_bonus
            + forward_bonus
        )

    # ------------------------------------------------------------------
    # Done conditions
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        root_pos = self.robot.data.root_pos_w
        robot_xy = root_pos[:, :2] - self.scene.env_origins[:, :2]
        dist_to_exit = torch.norm(robot_xy - self._exit_pos.unsqueeze(0), dim=-1)
        escaped = dist_to_exit < self.cfg.exit_threshold

        return escaped, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()

        n = len(env_ids)
        if self.cfg.randomize_spawn:
            indices = torch.randint(
                0, len(self._valid_spawn_positions), (n,)
            )
            for i, idx in enumerate(indices):
                sx, sy = self._valid_spawn_positions[idx.item()]
                default_root_state[i, 0] = sx
                default_root_state[i, 1] = sy
        else:
            default_root_state[:, 0] = 0.0
            default_root_state[:, 1] = 0.0

        default_root_state[:, 2] = 0.05

        yaw = sample_uniform(-math.pi, math.pi, (n,), self.device)
        default_root_state[:, 3] = torch.cos(yaw / 2.0)
        default_root_state[:, 4] = 0.0
        default_root_state[:, 5] = 0.0
        default_root_state[:, 6] = torch.sin(yaw / 2.0)

        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.coverage_tracker.reset(env_ids)

        # Initialize prev BFS distance for distance-reduction reward
        robot_xy = default_root_state[:, :2] - self.scene.env_origins[env_ids, :2]
        self._prev_bfs_dist[env_ids] = self._xy_to_bfs_dist(robot_xy)
