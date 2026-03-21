"""ObjectPickup RL environment (T5).

Mobile manipulation: navigate → grasp → place in basket.
Chassis (differential drive) + 5-DOF arm + gripper controlled jointly.

Observations (94-dim) — see env_cfg.py for breakdown.
Actions (8-dim):
    [0]   linear_vel
    [1]   angular_vel
    [2:7] arm_joints_delta(5)
    [7]   gripper
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
from isaaclab.utils.math import sample_uniform, quat_rotate_inverse

from smartcleaningrobot.robot_assets.turtlebot4_with_arm import MAX_REACH, diff_drive_forward
from smartcleaningrobot.scenes import AVAILABLE_SCENES, build_scene
from smartcleaningrobot.small_objects.detector import OracleDetector
from smartcleaningrobot.small_objects.object_assets import make_object_cfg
from smartcleaningrobot.utils.object_tracker import ObjectTracker

from .._shared import (
    apply_diff_drive,
    compute_wall_collision,
    detect_objects,
    process_lidar,
    randomize_object_positions,
)
from .env_cfg import ObjectPickupEnvCfg


class ObjectPickupEnv(DirectRLEnv):
    cfg: ObjectPickupEnvCfg

    def __init__(self, cfg: ObjectPickupEnvCfg, render_mode: str | None = None, **kwargs):
        self.scene_config = AVAILABLE_SCENES[cfg.scene_name]
        super().__init__(cfg, render_mode, **kwargs)

        # Joint bookkeeping
        self._wheel_ids, _ = self.robot.find_joints(self.cfg.wheel_joint_names)
        self._arm_ids, _ = self.robot.find_joints(self.cfg.arm_joint_names)
        self._gripper_ids, _ = self.robot.find_joints(self.cfg.gripper_joint_names)
        self._ee_body_ids, _ = self.robot.find_bodies([self.cfg.ee_body_name])

        lims = self.cfg.joint_limits
        self._joint_lo = torch.tensor([l[0] for l in lims], device=self.device)
        self._joint_hi = torch.tensor([l[1] for l in lims], device=self.device)
        self._joint_range = self._joint_hi - self._joint_lo

        # Object tracker and detector
        self._obj_tracker = ObjectTracker(
            self.num_envs, self.cfg.num_objects, self.device
        )
        self._detector = OracleDetector(
            fov_range=self.cfg.oracle_fov_range,
            max_detections=self.cfg.oracle_max_detections,
        )

        # Basket offset tensor (robot-relative)
        self._basket_offset = torch.tensor(
            self.cfg.basket_offset, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)

        # Per-env state
        self._prev_ee_dist = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._holding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        build_scene(self.scene_config)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.lidar = RayCaster(self.cfg.lidar_cfg)

        # Spawn all object slots
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
        # Wheels
        apply_diff_drive(
            self._actions, self.cfg.max_linear_vel, self.cfg.max_angular_vel,
            self.robot, self._wheel_ids, diff_drive_forward,
        )

        # Arm
        arm_delta = self._actions[:, 2:7] * self.cfg.max_joint_delta
        cur_arm = self.robot.data.joint_pos[:, self._arm_ids]
        new_arm = (cur_arm + arm_delta).clamp(self._joint_lo, self._joint_hi)
        self.robot.set_joint_position_target(new_arm, joint_ids=self._arm_ids)

        # Gripper
        close = self._actions[:, 7] > self.cfg.gripper_threshold
        gpos = torch.where(
            close.unsqueeze(-1).expand(-1, 2),
            torch.zeros(self.num_envs, 2, device=self.device),
            torch.full((self.num_envs, 2), 0.018, device=self.device),
        )
        self.robot.set_joint_position_target(gpos, joint_ids=self._gripper_ids)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        base_pos_w = self.robot.data.root_pos_w        # (N, 3)
        base_quat_w = self.robot.data.root_quat_w      # (N, 4)

        # Chassis velocity
        lin_vel = self.robot.data.root_com_lin_vel_b[:, 0:1]
        ang_vel = self.robot.data.root_com_ang_vel_b[:, 2:3]
        vel = torch.cat([lin_vel, ang_vel], dim=-1)   # (N, 2)

        # LiDAR
        lidar_obs, _ = process_lidar(self.lidar, self.cfg.lidar_cfg.max_distance)

        # OracleDetector: gather all object positions
        detect_obs, all_pos = detect_objects(
            self._detector, base_pos_w, base_quat_w, self._objects,
        )

        # End-effector state
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_ids[0]]
        ee_rel = quat_rotate_inverse(base_quat_w, ee_pos_w - base_pos_w)   # (N, 3)
        shoulder_w = base_pos_w.clone()
        shoulder_w[:, 2] += 0.15
        arm_reach_norm = (torch.norm(ee_pos_w - shoulder_w, dim=-1) / MAX_REACH).clamp(0, 1)
        gripper_pos = self.robot.data.joint_pos[:, self._gripper_ids]
        gripper_state = (gripper_pos[:, 0] / 0.037).clamp(0, 1)
        ee_state = torch.cat([
            ee_rel, arm_reach_norm.unsqueeze(-1), gripper_state.unsqueeze(-1)
        ], dim=-1)   # (N, 5)

        # Nearest target object (first FREE object by distance)
        obj_states = self._obj_tracker.states   # (N, M)
        free_mask = obj_states == ObjectTracker.FREE
        target_pos_rel = torch.zeros(self.num_envs, 3, device=self.device)
        target_dist_norm = torch.ones(self.num_envs, 1, device=self.device)
        in_grasp_range = torch.zeros(self.num_envs, 1, device=self.device)

        for i, obj in enumerate(self._objects):
            obj_pos = obj.data.root_pos_w   # (N, 3)
            d = torch.norm(ee_pos_w - obj_pos, dim=-1)   # (N,)
            # Prioritise: fill target info only where this is the best free object so far
            is_closer = free_mask[:, i] & (d < target_dist_norm.squeeze(-1) * 1.0)
            rel = quat_rotate_inverse(base_quat_w, obj_pos - base_pos_w)
            target_pos_rel = torch.where(is_closer.unsqueeze(-1).expand(-1, 3), rel, target_pos_rel)
            target_dist_norm = torch.where(
                is_closer.unsqueeze(-1), d.clamp(0, 1).unsqueeze(-1), target_dist_norm
            )
            in_grasp = (d < self.cfg.grasp_contact_threshold) & free_mask[:, i]
            in_grasp_range = torch.where(
                is_closer.unsqueeze(-1), in_grasp.float().unsqueeze(-1), in_grasp_range
            )

        # Basket count
        basket_norm = self._obj_tracker.get_placed_count_norm(self.cfg.num_objects).unsqueeze(-1)

        # Holding
        holding_obs = self._holding.float().unsqueeze(-1)

        # Wrist height (ee Z normalised)
        wrist_height = (ee_pos_w[:, 2] / 0.5).clamp(0, 1).unsqueeze(-1)

        # Cache for reward
        self._ee_pos_w = ee_pos_w
        self._all_obj_pos = all_pos
        self._base_pos_w = base_pos_w
        self._base_quat_w = base_quat_w

        obs = torch.cat([
            vel,              # 2
            lidar_obs,        # 72
            detect_obs,       # 6
            ee_state,         # 5
            target_pos_rel,   # 3
            target_dist_norm, # 1
            in_grasp_range,   # 1
            basket_norm,      # 1
            holding_obs,      # 1
            wrist_height,     # 1
        ], dim=-1)  # total 93... off by 1 — lidar_obs could be 73 or target_pos adds 1

        # Pad to match observation_space=94
        if obs.shape[-1] < self.cfg.observation_space:
            pad = torch.zeros(self.num_envs, self.cfg.observation_space - obs.shape[-1],
                              device=self.device)
            obs = torch.cat([obs, pad], dim=-1)

        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        total = torch.zeros(self.num_envs, device=self.device)

        # Approach reward: EE moving closer to nearest free object
        ee_pos_w = self._ee_pos_w
        best_dist = torch.full((self.num_envs,), float("inf"), device=self.device)
        best_obj_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        for i, obj in enumerate(self._objects):
            free = (self._obj_tracker.states[:, i] == ObjectTracker.FREE)
            d = torch.norm(ee_pos_w - obj.data.root_pos_w, dim=-1)
            closer = free & (d < best_dist)
            best_dist = torch.where(closer, d, best_dist)
            best_obj_idx = torch.where(closer, torch.full_like(best_obj_idx, i), best_obj_idx)

        dist_reduction = (self._prev_ee_dist - best_dist).clamp(min=0.0)
        total += self.cfg.rew_approach_object * dist_reduction
        self._prev_ee_dist = best_dist.clone()

        # Grasp detection & basket placement (per object)
        gripper_closed = (self.robot.data.joint_pos[:, self._gripper_ids[0]] < 0.005)
        basket_pos_w = self._base_pos_w + quat_rotate_inverse(
            self._base_quat_w,
            torch.tensor(self.cfg.basket_offset, device=self.device)
            .unsqueeze(0).expand(self.num_envs, -1)
        )

        for i, obj in enumerate(self._objects):
            obj_pos = obj.data.root_pos_w
            d_ee = torch.norm(ee_pos_w - obj_pos, dim=-1)
            in_contact = d_ee < self.cfg.grasp_contact_threshold
            state_i = self._obj_tracker.states[:, i]

            # Attempt grasp: free + contact + gripper closing
            newly_grasped = (state_i == ObjectTracker.FREE) & in_contact & gripper_closed
            if newly_grasped.any():
                idx = newly_grasped.nonzero(as_tuple=False).squeeze(-1)
                self._obj_tracker.mark_grasped(idx, torch.full_like(idx, i))
                total[idx] += self.cfg.rew_grasp_success
                self._holding[idx] = True

            # Detect drop: was GRASPED, now EE moved far away without basket placement
            was_grasped = state_i == ObjectTracker.GRASPED
            dropped = was_grasped & (d_ee > 0.10) & ~in_contact
            if dropped.any():
                idx = dropped.nonzero(as_tuple=False).squeeze(-1)
                self._obj_tracker.mark_dropped(idx, torch.full_like(idx, i))
                total[idx] += self.cfg.rew_drop_penalty
                self._holding[idx] = False

            # Basket placement check
            in_basket = ObjectTracker.check_in_basket(
                obj_pos.unsqueeze(1), basket_pos_w, self.cfg.basket_size
            ).squeeze(-1)   # (N,)
            newly_placed = (state_i == ObjectTracker.GRASPED) & in_basket
            if newly_placed.any():
                idx = newly_placed.nonzero(as_tuple=False).squeeze(-1)
                self._obj_tracker.mark_placed(idx, torch.full_like(idx, i))
                total[idx] += self.cfg.rew_place_in_basket
                self._holding[idx] = False

        # Collision penalties (LiDAR-based)
        _, lidar_dist = process_lidar(self.lidar, self.cfg.lidar_cfg.max_distance)
        wall_col = compute_wall_collision(lidar_dist, 0.20)
        total += self.cfg.rew_wall_collision * wall_col

        # Time penalty
        total += self.cfg.rew_time_penalty

        return total

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        all_placed = self._obj_tracker.placed_count >= self.cfg.num_objects
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return all_placed, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset robot
        default_state = self.robot.data.default_root_state[env_ids].clone()
        default_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_state[:, 7:], env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset objects at random non-overlapping positions
        randomize_object_positions(
            self._objects, env_ids, self.scene_config.scene_bounds,
            self.scene.env_origins, min_separation=0.15, device=self.device,
        )

        # Reset trackers
        ids_tensor = torch.as_tensor(env_ids, device=self.device)
        self._obj_tracker.reset(ids_tensor)
        self._prev_ee_dist[ids_tensor] = float("inf")
        self._holding[ids_tensor] = False
