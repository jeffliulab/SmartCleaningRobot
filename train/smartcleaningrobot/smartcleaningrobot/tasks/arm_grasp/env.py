"""ArmGrasp RL environment (T4).

Fixed-base arm manipulation task.  The chassis never moves; only the
5-DOF arm and gripper are controlled.  Curriculum progresses from a
fixed target (Stage 1) through random targets (Stage 2) to fully
randomised arm starts (Stage 3).

Observations (21-dim):
    ee_pos_rel(3)     end-effector XYZ relative to robot base, body frame
    ee_quat(4)        end-effector quaternion (w, x, y, z)
    joint_pos_norm(5) arm joint angles normalised to [-1, 1]
    gripper_state(1)  gripper opening [0=closed, 1=open]
    target_pos_rel(3) target object XYZ relative to robot base
    target_dist(1)    EE-to-target Euclidean distance, clamped [0, 1]
    contact(1)        1 if EE within grasp threshold of target
    holding(1)        1 if gripper closed + object co-moving with EE
    arm_reach_norm(1) EE distance from shoulder, normalised by MAX_REACH
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_rotate_inverse

from smartcleaningrobot.robot_assets.turtlebot4_with_arm import MAX_REACH
from smartcleaningrobot.small_objects.object_assets import make_object_cfg

from .env_cfg import ArmGraspEnvCfg


class ArmGraspEnv(DirectRLEnv):
    cfg: ArmGraspEnvCfg

    def __init__(self, cfg: ArmGraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint index bookkeeping
        self._arm_joint_ids, _ = self.robot.find_joints(self.cfg.arm_joint_names)
        self._gripper_joint_ids, _ = self.robot.find_joints(self.cfg.gripper_joint_names)
        self._ee_body_ids, _ = self.robot.find_bodies([self.cfg.ee_body_name])

        # Joint limits tensor: (5, 2)
        lims = self.cfg.joint_limits
        self._joint_lo = torch.tensor([l[0] for l in lims], device=self.device)
        self._joint_hi = torch.tensor([l[1] for l in lims], device=self.device)
        self._joint_range = self._joint_hi - self._joint_lo

        # Per-env grasp state
        self._grasp_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lift_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_holding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_ee_dist = torch.full((self.num_envs,), float("inf"), device=self.device)

        # Store actions for apply step
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Spawn ONE target object per environment (coin is easiest to grasp)
        target_cfg = make_object_cfg(object_type="coin", slot_index=0)
        self.target_object = RigidObject(target_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["target"] = self.target_object

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        # Arm: add scaled delta to current joint positions
        arm_delta = self._actions[:, :5] * self.cfg.max_joint_delta
        current_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        new_pos = (current_pos + arm_delta).clamp(
            self._joint_lo.unsqueeze(0), self._joint_hi.unsqueeze(0)
        )
        self.robot.set_joint_position_target(new_pos, joint_ids=self._arm_joint_ids)

        # Gripper: binarise action > threshold → close
        gripper_cmd = self._actions[:, 5]
        close = gripper_cmd > self.cfg.gripper_threshold
        gripper_pos = torch.where(
            close.unsqueeze(-1).expand(-1, 2),
            torch.zeros(self.num_envs, 2, device=self.device),        # closed
            torch.full((self.num_envs, 2), self.cfg.gripper_joint_names.__len__() * 0.018,
                       device=self.device),                            # open (≈ 18 mm each)
        )
        self.robot.set_joint_position_target(gripper_pos, joint_ids=self._gripper_joint_ids)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        # Robot base pose
        base_pos_w = self.robot.data.root_pos_w          # (N, 3)
        base_quat_w = self.robot.data.root_quat_w        # (N, 4)

        # End-effector world pose
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_ids[0]]   # (N, 3)
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_ids[0]] # (N, 4)

        # EE position in robot base frame
        ee_rel_w = ee_pos_w - base_pos_w
        ee_pos_rel = quat_rotate_inverse(base_quat_w, ee_rel_w)  # (N, 3)

        # Arm joint angles normalised to [-1, 1]
        joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]  # (N, 5)
        joint_pos_norm = 2.0 * (joint_pos - self._joint_lo) / self._joint_range - 1.0

        # Gripper opening ratio [0, 1]
        gripper_pos = self.robot.data.joint_pos[:, self._gripper_joint_ids]  # (N, 2)
        gripper_state = (gripper_pos[:, 0] / 0.037).clamp(0.0, 1.0).unsqueeze(-1)  # (N, 1)

        # Target object relative position
        target_pos_w = self.target_object.data.root_pos_w  # (N, 3)
        target_rel_w = target_pos_w - base_pos_w
        target_pos_rel = quat_rotate_inverse(base_quat_w, target_rel_w)  # (N, 3)

        # EE → target distance
        ee_to_target = torch.norm(ee_pos_w - target_pos_w, dim=-1)  # (N,)
        target_dist = ee_to_target.clamp(0.0, 1.0).unsqueeze(-1)    # (N, 1)

        # Contact: EE within threshold of target
        contact = (ee_to_target < self.cfg.grasp_contact_threshold).float().unsqueeze(-1)  # (N, 1)

        # Holding: gripper closed (< 5 mm open) + contact
        gripper_closed = (gripper_pos[:, 0] < 0.005).float()
        holding = (gripper_closed * contact.squeeze(-1)).unsqueeze(-1)  # (N, 1)

        # Arm reach normalised
        # Approximate shoulder world position = base_pos + (0, 0, BASE_HEIGHT)
        # Use ee distance from (base_pos_xy + offset_z) as reach proxy
        shoulder_pos_w = base_pos_w.clone()
        shoulder_pos_w[:, 2] += 0.15   # approximate shoulder height
        arm_reach = torch.norm(ee_pos_w - shoulder_pos_w, dim=-1)
        arm_reach_norm = (arm_reach / MAX_REACH).clamp(0.0, 1.0).unsqueeze(-1)  # (N, 1)

        obs = torch.cat([
            ee_pos_rel,       # 3
            ee_quat_w,        # 4
            joint_pos_norm,   # 5
            gripper_state,    # 1
            target_pos_rel,   # 3
            target_dist,      # 1
            contact,          # 1
            holding,          # 1
            arm_reach_norm,   # 1
        ], dim=-1)  # total: 21

        # Cache for reward computation
        self._ee_pos_w = ee_pos_w
        self._ee_to_target = ee_to_target
        self._contact = contact.squeeze(-1).bool()
        self._holding = holding.squeeze(-1).bool()
        self._target_pos_w = target_pos_w

        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        # Dense reach reward: reward when EE gets closer
        dist_reduction = (self._prev_ee_dist - self._ee_to_target).clamp(min=0.0)
        reach_reward = self.cfg.rew_reach * dist_reduction
        self._prev_ee_dist = self._ee_to_target.clone()

        # Contact reward: small bonus per step when touching
        contact_reward = self.cfg.rew_contact * self._contact.float()

        # Time penalty
        time_penalty = self.cfg.rew_time_penalty * torch.ones(self.num_envs, device=self.device)

        # One-shot grasp success bonus
        new_grasp = self._holding & ~self._grasp_success
        grasp_bonus = self.cfg.rew_grasp_success * new_grasp.float()
        self._grasp_success |= new_grasp

        # Lift success: held object rises above grasp_lift_height relative to start
        lifted = self._holding & (self._target_pos_w[:, 2] > self.cfg.object_spawn_height
                                  + self.cfg.grasp_lift_height)
        new_lift = lifted & ~self._lift_success
        lift_bonus = self.cfg.rew_lift_success * new_lift.float()
        self._lift_success |= new_lift

        # Drop penalty: was holding, now dropped
        dropped = self._prev_holding & ~self._holding & ~new_grasp
        drop_penalty = self.cfg.rew_drop_penalty * dropped.float()
        self._prev_holding = self._holding.clone()

        return reach_reward + contact_reward + time_penalty + grasp_bonus + lift_bonus + drop_penalty

    # ------------------------------------------------------------------
    # Done conditions
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Terminate early on successful lift
        success = self._lift_success
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return success, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # --- Reset robot to default pose (base locked, arm in ready pose) ---
        default_state = self.robot.data.default_root_state[env_ids].clone()
        default_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_state[:, 7:], env_ids)

        # Arm joints
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        if self.cfg.curriculum_stage >= 3:
            # Stage 3: randomise arm start pose ± 20% of joint range
            noise = sample_uniform(-0.2, 0.2, (len(env_ids), 5), self.device)
            arm_range = self._joint_hi - self._joint_lo
            arm_noise = noise * arm_range.unsqueeze(0)
            joint_pos[torch.arange(len(env_ids)), :][:, self._arm_joint_ids] += arm_noise
            joint_pos[:, self._arm_joint_ids] = joint_pos[:, self._arm_joint_ids].clamp(
                self._joint_lo, self._joint_hi
            )

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # --- Spawn target object ---
        n = len(env_ids)
        target_state = self.target_object.data.default_root_state[env_ids].clone()

        if self.cfg.curriculum_stage == 1:
            # Fixed position: directly forward at target_fixed_x
            target_state[:, 0] = self.scene.env_origins[env_ids, 0] + self.cfg.target_fixed_x
            target_state[:, 1] = self.scene.env_origins[env_ids, 1]
        else:
            # Random forward distance + lateral jitter
            dist = sample_uniform(
                self.cfg.target_range_min, self.cfg.target_range_max, (n,), self.device
            )
            lateral = sample_uniform(-0.10, 0.10, (n,), self.device)
            target_state[:, 0] = self.scene.env_origins[env_ids, 0] + dist
            target_state[:, 1] = self.scene.env_origins[env_ids, 1] + lateral

        target_state[:, 2] = self.scene.env_origins[env_ids, 2] + self.cfg.object_spawn_height
        target_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # identity quat
        target_state[:, 7:] = 0.0  # zero velocity

        self.target_object.write_root_pose_to_sim(target_state[:, :7], env_ids)
        self.target_object.write_root_velocity_to_sim(target_state[:, 7:], env_ids)

        # --- Reset per-env state ---
        ids = torch.as_tensor(env_ids, device=self.device)
        self._grasp_success[ids] = False
        self._lift_success[ids] = False
        self._prev_holding[ids] = False
        self._prev_ee_dist[ids] = float("inf")
