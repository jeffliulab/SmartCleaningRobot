"""CoveragePickup RL environment (T7).

End-to-end cleaning + object pickup task.
A rule-based FSM activates sub-behaviours; the policy outputs all 8 actions
at every step and the FSM decides which components have effect.

FSM states (encoded as 3D one-hot in observations):
    CLEAN    (0) — coverage navigation; arm stowed; wheels active
    APPROACH (1) — navigate toward nearest pickable object; arm prep
    PICKUP   (2) — fine approach + arm grasp + basket placement

The FSM transitions are deterministic rules operating on sensor data.
The RL policy does NOT control the FSM transitions directly; it learns to
produce good wheel + arm actions for the current FSM state.

Observations (112-dim) — see env_cfg.py for full breakdown.
Actions (8-dim): chassis(2) + arm_delta(5) + gripper(1)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum

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
from smartcleaningrobot.utils.coverage_tracker import CoverageTracker
from smartcleaningrobot.utils.object_tracker import ObjectTracker

from .._shared import (
    apply_diff_drive,
    compute_wall_collision,
    detect_objects,
    process_lidar,
    randomize_object_positions,
)
from .env_cfg import CoveragePickupEnvCfg


class FSMState(IntEnum):
    CLEAN = 0
    APPROACH = 1
    PICKUP = 2


class CoveragePickupEnv(DirectRLEnv):
    cfg: CoveragePickupEnvCfg

    def __init__(self, cfg: CoveragePickupEnvCfg, render_mode: str | None = None, **kwargs):
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

        self._pickable_set = set(self.cfg.pickable_indices)

        # Trackers
        self.coverage_tracker = CoverageTracker(
            grid_resolution=self.cfg.coverage_grid_resolution,
            scene_bounds=self.scene_config.scene_bounds,
            robot_radius=self.cfg.robot_cleaning_radius,
            num_envs=self.num_envs,
            device=self.device,
        )
        self._obj_tracker = ObjectTracker(
            self.num_envs, self.cfg.num_objects, self.device
        )
        self._detector = OracleDetector(
            fov_range=self.cfg.oracle_fov_range,
            max_detections=self.cfg.oracle_max_detections,
        )

        # FSM state per env
        self._fsm = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Per-env caches
        self._newly_covered = torch.zeros(self.num_envs, device=self.device)
        self._holding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasp_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

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
            obj = RigidObject(make_object_cfg(obj_type, i))
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
        # Wheels always active
        apply_diff_drive(
            self._actions, self.cfg.max_linear_vel, self.cfg.max_angular_vel,
            self.robot, self._wheel_ids, diff_drive_forward,
        )

        # Arm active in APPROACH and PICKUP states
        arm_active = (self._fsm != FSMState.CLEAN).unsqueeze(-1).float()
        arm_delta = self._actions[:, 2:7] * self.cfg.max_joint_delta * arm_active
        cur_arm = self.robot.data.joint_pos[:, self._arm_ids]
        new_arm = (cur_arm + arm_delta).clamp(self._joint_lo, self._joint_hi)
        self.robot.set_joint_position_target(new_arm, joint_ids=self._arm_ids)

        # Gripper active in PICKUP state only
        gripper_active = (self._fsm == FSMState.PICKUP)
        close = (self._actions[:, 7] > self.cfg.gripper_threshold) & gripper_active
        gpos = torch.where(
            close.unsqueeze(-1).expand(-1, 2),
            torch.zeros(self.num_envs, 2, device=self.device),
            torch.full((self.num_envs, 2), 0.018, device=self.device),
        )
        self.robot.set_joint_position_target(gpos, joint_ids=self._gripper_ids)

    # ------------------------------------------------------------------
    # FSM transition (called before obs assembly)
    # ------------------------------------------------------------------

    def _update_fsm(
        self,
        base_pos_w: torch.Tensor,
        ee_pos_w: torch.Tensor,
        all_obj_pos: torch.Tensor,
    ) -> None:
        """Rule-based FSM transitions.

        CLEAN → APPROACH: nearest pickable object within approach_trigger_dist
        APPROACH → PICKUP: EE within grasp_contact_threshold of target
        PICKUP → CLEAN: object placed (obj_tracker) OR holding dropped
        """
        for i in self._pickable_set:
            obj_pos = all_obj_pos[:, i, :2]
            free = (self._obj_tracker.states[:, i] == ObjectTracker.FREE)
            d_robot = torch.norm(base_pos_w[:, :2] - obj_pos, dim=-1)

            # CLEAN → APPROACH
            trigger = free & (d_robot < self.cfg.approach_trigger_dist)
            in_clean = self._fsm == FSMState.CLEAN
            self._fsm = torch.where(in_clean & trigger,
                                    torch.full_like(self._fsm, FSMState.APPROACH),
                                    self._fsm)

        # APPROACH → PICKUP: EE within grasp range of any pickable free object
        ee_to_obj_min = torch.full((self.num_envs,), float("inf"), device=self.device)
        for i in self._pickable_set:
            free = (self._obj_tracker.states[:, i] == ObjectTracker.FREE)
            d_ee = torch.norm(ee_pos_w - all_obj_pos[:, i], dim=-1)
            ee_to_obj_min = torch.where(free, torch.minimum(ee_to_obj_min, d_ee), ee_to_obj_min)

        in_range = ee_to_obj_min < self.cfg.grasp_contact_threshold
        in_approach = self._fsm == FSMState.APPROACH
        self._fsm = torch.where(in_approach & in_range,
                                torch.full_like(self._fsm, FSMState.PICKUP),
                                self._fsm)

        # PICKUP → CLEAN: object was just placed (obj_tracker placed_count increased)
        # Also transition back if all pickable objects are handled
        all_done = torch.tensor(
            [self._obj_tracker.placed_count[e].item() >= self.cfg.num_pickable
             for e in range(self.num_envs)],
            dtype=torch.bool, device=self.device,
        )
        in_pickup = self._fsm == FSMState.PICKUP
        no_holding = ~self._holding
        self._fsm = torch.where(in_pickup & no_holding & ~in_range,
                                torch.full_like(self._fsm, FSMState.CLEAN),
                                self._fsm)
        self._fsm[all_done] = FSMState.CLEAN

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        base_pos_w = self.robot.data.root_pos_w
        base_quat_w = self.robot.data.root_quat_w

        # Chassis velocity
        lin_vel = self.robot.data.root_com_lin_vel_b[:, 0:1]
        ang_vel = self.robot.data.root_com_ang_vel_b[:, 2:3]
        vel = torch.cat([lin_vel, ang_vel], dim=-1)   # (N, 2)

        # LiDAR
        lidar_obs, lidar_dist = process_lidar(self.lidar, self.cfg.lidar_cfg.max_distance)

        # Object positions
        _, all_pos = detect_objects(
            self._detector, base_pos_w, base_quat_w, self._objects,
        )

        # EE pose
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_ids[0]]

        # Update FSM
        self._update_fsm(base_pos_w, ee_pos_w, all_pos)

        # Detector (only on pickable objects)
        detect_obs, _ = detect_objects(
            self._detector, base_pos_w, base_quat_w, self._objects,
            object_indices=list(self._pickable_set),
        )

        # Coverage
        robot_xy = base_pos_w[:, :2] - self.scene.env_origins[:, :2]
        self._newly_covered = self.coverage_tracker.update(robot_xy)
        coverage_ratio = self.coverage_tracker.get_coverage_ratio().unsqueeze(-1)  # (N, 1)

        # EE state
        ee_rel = quat_rotate_inverse(base_quat_w, ee_pos_w - base_pos_w)
        shoulder_w = base_pos_w.clone(); shoulder_w[:, 2] += 0.15
        arm_reach_norm = (torch.norm(ee_pos_w - shoulder_w, dim=-1) / MAX_REACH).clamp(0, 1)
        gripper_pos = self.robot.data.joint_pos[:, self._gripper_ids]
        gripper_state = (gripper_pos[:, 0] / 0.037).clamp(0, 1)
        ee_state = torch.cat([ee_rel, arm_reach_norm.unsqueeze(-1), gripper_state.unsqueeze(-1)], dim=-1)  # (N, 5)

        # Nearest pickable target
        target_pos_rel = torch.zeros(self.num_envs, 3, device=self.device)
        target_dist_norm = torch.ones(self.num_envs, 1, device=self.device)
        in_grasp_range = torch.zeros(self.num_envs, 1, device=self.device)

        for i in self._pickable_set:
            free = (self._obj_tracker.states[:, i] == ObjectTracker.FREE)
            obj_pos = all_pos[:, i, :]
            d = torch.norm(ee_pos_w - obj_pos, dim=-1)
            closer = free & (d < target_dist_norm.squeeze(-1))
            rel = quat_rotate_inverse(base_quat_w, obj_pos - base_pos_w)
            target_pos_rel = torch.where(closer.unsqueeze(-1).expand(-1, 3), rel, target_pos_rel)
            target_dist_norm = torch.where(closer.unsqueeze(-1), d.clamp(0, 1).unsqueeze(-1), target_dist_norm)
            in_grasp = (d < self.cfg.grasp_contact_threshold) & free
            in_grasp_range = torch.where(closer.unsqueeze(-1), in_grasp.float().unsqueeze(-1), in_grasp_range)

        basket_norm = (self._obj_tracker.placed_count.float() / self.cfg.num_pickable).unsqueeze(-1)
        holding_obs = self._holding.float().unsqueeze(-1)
        wrist_height = (ee_pos_w[:, 2] / 0.5).clamp(0, 1).unsqueeze(-1)

        # FSM one-hot
        fsm_onehot = torch.zeros(self.num_envs, 3, device=self.device)
        fsm_onehot.scatter_(1, self._fsm.unsqueeze(-1), 1.0)   # (N, 3)

        # Cache
        self._base_pos_w = base_pos_w
        self._base_quat_w = base_quat_w
        self._ee_pos_w = ee_pos_w
        self._all_obj_pos = all_pos
        self._lidar_dist = lidar_dist

        core = torch.cat([
            vel,              # 2
            lidar_obs,        # 72
            detect_obs,       # 6
            coverage_ratio,   # 1
            ee_state,         # 5
            target_pos_rel,   # 3
            target_dist_norm, # 1
            in_grasp_range,   # 1
            basket_norm,      # 1
            holding_obs,      # 1
            wrist_height,     # 1
            fsm_onehot,       # 3
        ], dim=-1)  # = 97

        # Pad to observation_space=112
        if core.shape[-1] < self.cfg.observation_space:
            pad = torch.zeros(self.num_envs, self.cfg.observation_space - core.shape[-1],
                              device=self.device)
            obs = torch.cat([core, pad], dim=-1)
        else:
            obs = core[:, :self.cfg.observation_space]

        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        total = torch.zeros(self.num_envs, device=self.device)

        # Coverage
        total += self.cfg.rew_coverage * self._newly_covered

        # LiDAR wall collision
        total += self.cfg.rew_wall_collision * compute_wall_collision(self._lidar_dist, 0.20)

        # Object interactions
        ee_pos_w = self._ee_pos_w
        gripper_closed = (self.robot.data.joint_pos[:, self._gripper_ids[0]] < 0.005)
        basket_pos_w = self._base_pos_w + quat_rotate_inverse(
            self._base_quat_w,
            torch.tensor(self.cfg.basket_offset, device=self.device)
            .unsqueeze(0).expand(self.num_envs, -1),
        )
        robot_xy = self._base_pos_w[:, :2].unsqueeze(1)

        for i, obj in enumerate(self._objects):
            obj_pos = obj.data.root_pos_w
            d_ee = torch.norm(ee_pos_w - obj_pos, dim=-1)
            d_robot = torch.norm(robot_xy - obj_pos[:, :2].unsqueeze(1), dim=-1).squeeze(1)
            state_i = self._obj_tracker.states[:, i]
            is_pickable = i in self._pickable_set

            if is_pickable:
                # Grasp attempt
                contact = d_ee < self.cfg.grasp_contact_threshold
                newly_grasped = (state_i == ObjectTracker.FREE) & contact & gripper_closed
                if newly_grasped.any():
                    idx = newly_grasped.nonzero(as_tuple=False).squeeze(-1)
                    self._obj_tracker.mark_grasped(idx, torch.full_like(idx, i))
                    total[idx] += self.cfg.rew_grasp_success
                    self._holding[idx] = True

                # Basket placement
                in_basket = ObjectTracker.check_in_basket(
                    obj_pos.unsqueeze(1), basket_pos_w, self.cfg.basket_size
                ).squeeze(-1)
                newly_placed = (state_i == ObjectTracker.GRASPED) & in_basket
                if newly_placed.any():
                    idx = newly_placed.nonzero(as_tuple=False).squeeze(-1)
                    self._obj_tracker.mark_placed(idx, torch.full_like(idx, i))
                    total[idx] += self.cfg.rew_place_in_basket
                    self._holding[idx] = False

                # Bump penalty
                bumped = (state_i == ObjectTracker.FREE) & (d_robot < 0.10)
                total += self.cfg.rew_obj_collision_pick * bumped.float()

            else:
                # Avoid-type object: penalty for getting too close
                too_close = d_robot < 0.12
                total += self.cfg.rew_obj_collision_avoid * too_close.float()

        # Time penalty
        total += self.cfg.rew_time_penalty

        return total

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        coverage = self.coverage_tracker.get_coverage_ratio()
        all_placed = self._obj_tracker.placed_count >= self.cfg.num_pickable
        success = (coverage >= self.cfg.coverage_done_threshold) | all_placed
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return success, time_out

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

        # Robot random spawn
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
        default_state[:, 4:6] = 0.0
        default_state[:, 6] = torch.sin(yaw / 2.0)
        default_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_state[:, 7:], env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Objects random non-overlapping
        randomize_object_positions(
            self._objects, env_ids, self.scene_config.scene_bounds,
            self.scene.env_origins, min_separation=0.20, device=self.device,
        )

        # Reset trackers and FSM
        ids_t = torch.as_tensor(env_ids, device=self.device)
        self.coverage_tracker.reset(ids_t)
        self._obj_tracker.reset(ids_t)
        self._fsm[ids_t] = FSMState.CLEAN
        self._holding[ids_t] = False
        self._grasp_success[ids_t] = False
