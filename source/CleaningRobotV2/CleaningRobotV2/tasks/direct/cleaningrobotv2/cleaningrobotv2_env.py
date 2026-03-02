"""Cleaning robot coverage RL environment.

The robot (JetBot) navigates an indoor scene attempting to maximize floor
coverage. Observations include robot pose, velocity, downsampled LiDAR
readings, and a local coverage map patch. The camera captures RGB data
each step but does not contribute to the policy observation (reserved for
the future small-object detection module).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera, RayCaster
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform

from CleaningRobotV2.robots.jetbot import diff_drive_forward
from CleaningRobotV2.scenes import AVAILABLE_SCENES
from CleaningRobotV2.utils.coverage_tracker import CoverageTracker

from .cleaningrobotv2_env_cfg import Cleaningrobotv2EnvCfg


class Cleaningrobotv2Env(DirectRLEnv):
    cfg: Cleaningrobotv2EnvCfg

    def __init__(self, cfg: Cleaningrobotv2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._wheel_joint_ids, _ = self.robot.find_joints(
            ["left_wheel_joint", "right_wheel_joint"]
        )

        self.scene_config = AVAILABLE_SCENES[self.cfg.scene_name]

        self.coverage_tracker = CoverageTracker(
            grid_resolution=self.cfg.coverage_grid_resolution,
            scene_bounds=self.scene_config.scene_bounds,
            robot_radius=self.cfg.robot_cleaning_radius,
            num_envs=self.num_envs,
            device=self.device,
        )

        self.newly_covered = torch.zeros(self.num_envs, device=self.device)

        # Milestone tracking: record which milestones each env has reached
        self._milestone_thresholds = torch.tensor([0.50, 0.75, 0.90, 0.95], device=self.device)
        self._milestones_reached = torch.zeros(
            self.num_envs, len(self._milestone_thresholds), dtype=torch.bool, device=self.device
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        # Spawn scene USD
        scene_usd = f"{ISAAC_NUCLEUS_DIR}{self.scene_config.usd_path}"
        scene_spawn_cfg = sim_utils.UsdFileCfg(usd_path=scene_usd)
        scene_spawn_cfg.func("/World/envs/env_.*/Scene", scene_spawn_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.lidar = RayCaster(self.cfg.lidar_cfg)
        self.camera = Camera(self.cfg.camera_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["lidar"] = self.lidar
        self.scene.sensors["camera"] = self.camera

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

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

    def _get_observations(self) -> dict:
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w

        # Extract yaw from quaternion (assuming mostly flat motion)
        yaw = 2.0 * torch.atan2(root_quat[:, 3], root_quat[:, 0])
        pose = torch.stack([root_pos[:, 0], root_pos[:, 1], yaw], dim=-1)

        lin_vel = self.robot.data.root_com_lin_vel_b[:, 0:1]
        ang_vel = self.robot.data.root_com_ang_vel_b[:, 2:3]
        vel = torch.cat([lin_vel, ang_vel], dim=-1)

        # LiDAR: compute distances from ray hits
        sensor_pos = self.lidar.data.pos_w.unsqueeze(1)
        ray_hits = self.lidar.data.ray_hits_w
        lidar_distances = torch.norm(ray_hits - sensor_pos, dim=-1)
        # Downsample 360 -> 36 rays, normalize to [0, 1]
        lidar_obs = lidar_distances[:, ::10]
        lidar_obs = lidar_obs.clamp(0.0, self.cfg.lidar_cfg.max_distance) / self.cfg.lidar_cfg.max_distance

        # Coverage tracking
        robot_xy = root_pos[:, :2] - self.scene.env_origins[:, :2]
        self.newly_covered = self.coverage_tracker.update(robot_xy)
        coverage_ratio = self.coverage_tracker.get_coverage_ratio()
        local_cov = self.coverage_tracker.get_local_view(
            robot_xy, self.cfg.coverage_local_view_size
        )
        local_cov_flat = local_cov.reshape(self.num_envs, -1)

        obs = torch.cat(
            [
                pose,
                vel,
                lidar_obs,
                local_cov_flat,
                coverage_ratio.unsqueeze(-1),
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        coverage_reward = self.cfg.rew_coverage_weight * self.newly_covered

        sensor_pos = self.lidar.data.pos_w.unsqueeze(1)
        ray_hits = self.lidar.data.ray_hits_w
        lidar_distances = torch.norm(ray_hits - sensor_pos, dim=-1)
        lidar_min = lidar_distances.min(dim=-1).values
        collision = (lidar_min < self.cfg.collision_threshold).float()
        collision_penalty = self.cfg.rew_collision_penalty * collision

        time_penalty = self.cfg.rew_time_penalty * torch.ones(self.num_envs, device=self.device)

        # Milestone bonus: awarded once per milestone per episode
        coverage_ratio = self.coverage_tracker.get_coverage_ratio()
        current_milestones = coverage_ratio.unsqueeze(-1) >= self._milestone_thresholds.unsqueeze(0)
        new_milestones = current_milestones & ~self._milestones_reached
        milestone_bonus = self.cfg.rew_milestone_bonus * new_milestones.any(dim=-1).float()
        self._milestones_reached = current_milestones

        return coverage_reward + collision_penalty + time_penalty + milestone_bonus

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        coverage_ratio = self.coverage_tracker.get_coverage_ratio()
        full_coverage = coverage_ratio >= self.cfg.coverage_done_threshold
        return full_coverage, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Random start position within a safe fraction of scene bounds
        x_min, y_min, x_max, y_max = self.scene_config.scene_bounds
        safe = 0.4
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 0] = sample_uniform(
            x_min * safe, x_max * safe, (len(env_ids),), self.device
        )
        default_root_state[:, 1] = sample_uniform(
            y_min * safe, y_max * safe, (len(env_ids),), self.device
        )
        default_root_state[:, 2] = 0.05

        # Random yaw orientation
        yaw = sample_uniform(-math.pi, math.pi, (len(env_ids),), self.device)
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
        self._milestones_reached[env_ids] = False
