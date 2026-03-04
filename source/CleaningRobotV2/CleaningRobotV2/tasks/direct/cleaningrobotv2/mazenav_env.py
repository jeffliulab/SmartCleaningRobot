"""Maze escape RL environment.

The robot (JetBot) must navigate a procedural DFS maze and reach the
exit in minimum time.  This task is entirely separate from the coverage
/cleaning task and serves as an RL learning exercise.

Observations (44 dims):
    pose(3) + velocity(2) + LiDAR(36) + exit_direction(2) + exit_distance(1)

Rewards:
    - Dense:  distance-reduction toward exit each step
    - Sparse: large bonus on reaching the exit
    - Penalties: collision, time
    - Small exploration bonus for visiting new cells
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

from CleaningRobotV2.robots.jetbot import diff_drive_forward
from CleaningRobotV2.scenes import AVAILABLE_SCENES, build_scene
from CleaningRobotV2.scenes.builders.maze import get_valid_cell_positions
from CleaningRobotV2.utils.coverage_tracker import CoverageTracker

from .mazenav_env_cfg import MazeEscapeEnvCfg


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
        self._prev_exit_dist = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------
    # Spawn position pre-computation
    # ------------------------------------------------------------------

    @staticmethod
    def _precompute_spawn_positions() -> list[tuple[float, float]]:
        """Return world-coordinate positions of every passable maze cell."""
        return get_valid_cell_positions()

    # ------------------------------------------------------------------
    # Scene setup (shared with parent env)
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
    # Observations  (44 dims)
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w

        yaw = 2.0 * torch.atan2(root_quat[:, 3], root_quat[:, 0])
        pose = torch.stack([root_pos[:, 0], root_pos[:, 1], yaw], dim=-1)

        lin_vel = self.robot.data.root_com_lin_vel_b[:, 0:1]
        ang_vel = self.robot.data.root_com_ang_vel_b[:, 2:3]
        vel = torch.cat([lin_vel, ang_vel], dim=-1)

        # LiDAR: 360 -> 36 rays, normalized to [0, 1]
        sensor_pos = self.lidar.data.pos_w.unsqueeze(1)
        ray_hits = self.lidar.data.ray_hits_w
        lidar_distances = torch.norm(ray_hits - sensor_pos, dim=-1)
        lidar_obs = lidar_distances[:, ::10]
        lidar_obs = lidar_obs.clamp(0.0, self.cfg.lidar_cfg.max_distance) / self.cfg.lidar_cfg.max_distance

        # Exit direction and distance (relative to robot, in world frame)
        robot_xy = root_pos[:, :2] - self.scene.env_origins[:, :2]
        exit_rel = self._exit_pos.unsqueeze(0) - robot_xy
        exit_dist = torch.norm(exit_rel, dim=-1, keepdim=True).clamp(min=1e-4)
        exit_dir = exit_rel / exit_dist
        exit_dist_norm = exit_dist / self.cfg.maze_diagonal

        # Update coverage tracker (for exploration bonus in reward)
        self.newly_covered = self.coverage_tracker.update(robot_xy)

        obs = torch.cat([pose, vel, lidar_obs, exit_dir, exit_dist_norm], dim=-1)
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        root_pos = self.robot.data.root_pos_w
        robot_xy = root_pos[:, :2] - self.scene.env_origins[:, :2]

        # Current distance to exit
        curr_dist = torch.norm(robot_xy - self._exit_pos.unsqueeze(0), dim=-1)

        # 1) Dense: distance reduction (positive when getting closer)
        dist_reduction = (self._prev_exit_dist - curr_dist) / self.cfg.maze_diagonal
        dist_reward = self.cfg.rew_dist_reduction * dist_reduction

        # 2) Sparse: exit reached bonus
        reached = (curr_dist < self.cfg.exit_threshold).float()
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

        self._prev_exit_dist = curr_dist.clone()

        return dist_reward + exit_bonus + collision_penalty + time_penalty + explore_bonus

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

        # Initialize prev distance for distance-reduction reward
        robot_xy = default_root_state[:, :2] - self.scene.env_origins[env_ids, :2]
        self._prev_exit_dist[env_ids] = torch.norm(
            robot_xy - self._exit_pos.unsqueeze(0), dim=-1
        )
