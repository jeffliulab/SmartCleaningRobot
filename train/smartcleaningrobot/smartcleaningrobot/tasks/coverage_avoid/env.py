"""CoverageAvoid RL environment (T6).

Chassis-only coverage task with small object avoidance.
The robot must clean as much floor area as possible while avoiding small
objects that would tangle in its wheels.

Observations (81-dim):
    vel(2)       chassis body-frame velocities
    lidar(72)    72-ray LiDAR distances, normalised
    nearest_3(6) OracleDetector: [r_norm, sin_θ] × 3 nearest objects
    coverage(1)  current episode coverage ratio
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import RayCaster
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from smartcleaningrobot.robot_assets.jetbot import diff_drive_forward
from smartcleaningrobot.scenes import AVAILABLE_SCENES, build_scene
from smartcleaningrobot.small_objects.detector import OracleDetector
from smartcleaningrobot.small_objects.object_assets import make_object_cfg
from smartcleaningrobot.utils.coverage_tracker import CoverageTracker

from .._shared import (
    apply_diff_drive,
    compute_wall_collision,
    detect_objects,
    process_lidar,
    randomize_object_positions,
)
from .env_cfg import CoverageAvoidEnvCfg


class CoverageAvoidEnv(DirectRLEnv):
    cfg: CoverageAvoidEnvCfg

    def __init__(self, cfg: CoverageAvoidEnvCfg, render_mode: str | None = None, **kwargs):
        self.scene_config = AVAILABLE_SCENES[cfg.scene_name]
        super().__init__(cfg, render_mode, **kwargs)

        self._wheel_ids, _ = self.robot.find_joints(["left_wheel_joint", "right_wheel_joint"])

        self.coverage_tracker = CoverageTracker(
            grid_resolution=self.cfg.coverage_grid_resolution,
            scene_bounds=self.scene_config.scene_bounds,
            robot_radius=self.cfg.robot_cleaning_radius,
            num_envs=self.num_envs,
            device=self.device,
        )

        self._detector = OracleDetector(
            fov_range=self.cfg.oracle_fov_range,
            max_detections=self.cfg.oracle_max_detections,
        )

        self._newly_covered = torch.zeros(self.num_envs, device=self.device)
        self._actions = torch.zeros(self.num_envs, 2, device=self.device)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        build_scene(self.scene_config)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.lidar = RayCaster(self.cfg.lidar_cfg)

        self._objects: list[RigidObject] = []
        for i, obj_type in enumerate(self.cfg.object_types):
            cfg_i = make_object_cfg(obj_type, i)
            obj = RigidObject(cfg_i)
            self._objects.append(obj)
            self.scene.rigid_objects[f"obj_{i}"] = obj

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/Scene"])

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["lidar"] = self.lidar

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        apply_diff_drive(
            self._actions, self.cfg.max_linear_vel, self.cfg.max_angular_vel,
            self.robot, self._wheel_ids, diff_drive_forward,
        )

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w

        # Chassis velocity
        lin_vel = self.robot.data.root_com_lin_vel_b[:, 0:1]
        ang_vel = self.robot.data.root_com_ang_vel_b[:, 2:3]
        vel = torch.cat([lin_vel, ang_vel], dim=-1)   # (N, 2)

        # LiDAR
        lidar_obs, lidar_dist = process_lidar(self.lidar, self.cfg.lidar_cfg.max_distance)

        # OracleDetector
        detect_obs, all_pos = detect_objects(
            self._detector, root_pos_w, root_quat_w, self._objects,
        )

        # Coverage
        robot_xy = root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        self._newly_covered = self.coverage_tracker.update(robot_xy)
        coverage_ratio = self.coverage_tracker.get_coverage_ratio().unsqueeze(-1)  # (N, 1)

        # Cache for reward
        self._root_pos_w = root_pos_w
        self._all_obj_pos = all_pos
        self._lidar_dist = lidar_dist

        obs = torch.cat([vel, lidar_obs, detect_obs, coverage_ratio], dim=-1)  # 81
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        # Coverage increase reward
        coverage_reward = self.cfg.rew_coverage * self._newly_covered

        # LiDAR-based wall collision
        wall_col = compute_wall_collision(self._lidar_dist, self.cfg.wall_collision_threshold)
        wall_penalty = self.cfg.rew_wall_collision * wall_col

        # Object collision: any object within threshold of robot base
        robot_xy = self._root_pos_w[:, :2].unsqueeze(1)   # (N, 1, 2)
        obj_xy = self._all_obj_pos[:, :, :2]               # (N, M, 2)
        obj_dist = torch.norm(obj_xy - robot_xy, dim=-1)   # (N, M)
        obj_col = (obj_dist < self.cfg.obj_collision_threshold).any(dim=-1).float()
        obj_penalty = self.cfg.rew_obj_collision * obj_col

        # Forward bonus
        forward_vel = self.robot.data.root_com_lin_vel_b[:, 0]
        forward_bonus = self.cfg.rew_forward_bonus * forward_vel.clamp(min=0.0)

        # Time penalty
        time_penalty = self.cfg.rew_time_penalty * torch.ones(self.num_envs, device=self.device)

        return coverage_reward + wall_penalty + obj_penalty + forward_bonus + time_penalty

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        coverage_ratio = self.coverage_tracker.get_coverage_ratio()
        done = coverage_ratio >= self.cfg.coverage_done_threshold
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return done, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)
        x_min, y_min, x_max, y_max = self.scene_config.scene_bounds
        safe = 0.5

        # Robot: random spawn position and yaw
        default_state = self.robot.data.default_root_state[env_ids].clone()
        default_state[:, 0] = self.scene.env_origins[env_ids, 0] + sample_uniform(
            x_min * safe, x_max * safe, (n,), self.device
        )
        default_state[:, 1] = self.scene.env_origins[env_ids, 1] + sample_uniform(
            y_min * safe, y_max * safe, (n,), self.device
        )
        default_state[:, 2] = 0.05
        yaw = sample_uniform(-math.pi, math.pi, (n,), self.device)
        default_state[:, 3] = torch.cos(yaw / 2.0)
        default_state[:, 4] = 0.0
        default_state[:, 5] = 0.0
        default_state[:, 6] = torch.sin(yaw / 2.0)
        default_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_state[:, 7:], env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Objects: random non-overlapping positions
        randomize_object_positions(
            self._objects, env_ids, self.scene_config.scene_bounds,
            self.scene.env_origins, min_separation=0.20, device=self.device,
        )

        self.coverage_tracker.reset(torch.as_tensor(env_ids, device=self.device))
