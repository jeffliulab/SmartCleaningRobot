"""Configuration for the ArmGrasp RL environment (T4).

The robot base is FIXED — only the 5-DOF arm and gripper move.
Curriculum is controlled by setting `curriculum_stage` (1, 2, or 3).

Observation space (21-dim)
--------------------------
ee_pos_rel(3)    : end-effector position relative to robot base, body frame
ee_quat(4)       : end-effector orientation (w, x, y, z)
joint_pos_norm(5): arm joint angles normalised to [-1, 1]
gripper_state(1) : gripper opening ratio in [0, 1]
target_pos_rel(3): target object XYZ relative to robot base
target_dist(1)   : Euclidean distance EE → target (m), clamped to [0, 1]
contact(1)       : 1 if EE within grasp_contact_threshold of target
holding(1)       : 1 if object is being held (contact + gripper closed)
arm_reach_norm(1): current EE distance from shoulder, normalised by MAX_REACH

Action space (6-dim)
---------------------
[0:5] arm_joints_delta  : joint angle increments, scaled by max_joint_delta (rad/step)
[5]   gripper           : target open/close in [0, 1] → binarised at 0.5

Curriculum stages
-----------------
1 : target fixed at (target_fixed_x, 0, 0)    — learn reach only
2 : target random in [approach_min, approach_max] forward range
3 : random target + random initial arm pose    — robustness
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from smartcleaningrobot.robot_assets.turtlebot4_with_arm import TURTLEBOT4_WITH_ARM_CFG


@configclass
class ArmGraspEnvCfg(DirectRLEnvCfg):
    # -- env --
    decimation: int = 2
    episode_length_s: float = 15.0       # shorter episodes for dense arm task

    # -- spaces --
    # ee_pos_rel(3) + ee_quat(4) + joint_pos_norm(5) + gripper(1)
    # + target_pos_rel(3) + target_dist(1) + contact(1) + holding(1) + arm_reach_norm(1)
    action_space: int = 6
    observation_space: int = 21
    state_space: int = 0

    # -- simulation --
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # -- robot --
    robot_cfg: ArticulationCfg = TURTLEBOT4_WITH_ARM_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # -- scene --
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=3.0, replicate_physics=True
    )

    # -- body / joint names (must match the USD model) --
    ee_body_name: str = "ee_gripper_link"
    arm_joint_names: list = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle"]
    gripper_joint_names: list = ["left_finger_link", "right_finger_link"]

    # -- arm limits (radians) --
    joint_limits: list = [
        (-3.14,  3.14),   # waist
        (-1.88,  1.99),   # shoulder
        (-2.15,  1.57),   # elbow
        (-3.14,  3.14),   # forearm_roll
        (-1.75,  2.15),   # wrist_angle
    ]

    # -- action scaling --
    max_joint_delta: float = 0.05   # rad per physics step (≈ 3°/step)
    gripper_threshold: float = 0.5  # action > this → close gripper

    # -- curriculum stage (1, 2, or 3) --
    curriculum_stage: int = 1

    # Stage 1: fixed target forward distance (m) in body frame
    target_fixed_x: float = 0.15

    # Stage 2/3: random target range in body-frame X (forward distance)
    target_range_min: float = 0.05
    target_range_max: float = 0.30

    # Target object sits on the ground (z = object_spawn_height)
    object_spawn_height: float = 0.01   # m above ground

    # -- grasp detection thresholds --
    grasp_contact_threshold: float = 0.04   # m — EE within this → contact
    grasp_lift_height: float = 0.05         # m — object must rise this much to count as lifted

    # -- reward weights --
    rew_reach: float = 2.0          # per step, proportional to -Δdist
    rew_contact: float = 1.0        # per step when contact = True
    rew_grasp_success: float = 20.0 # one-shot on first successful grasp
    rew_lift_success: float = 10.0  # one-shot on lift above grasp_lift_height
    rew_time_penalty: float = -0.01
    rew_drop_penalty: float = -5.0  # penalty if held object falls
