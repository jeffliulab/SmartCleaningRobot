"""Configuration for the CoveragePickup RL environment (T7).

End-to-end task: clean the floor while deciding per object whether to
avoid (large/heavy) or pick up and basket-place (small/light).

A High-Level FSM switches between two sub-policies:
    CLEAN    — coverage navigation (from T6 checkpoint)
    APPROACH — navigate toward detected pickable object
    PICKUP   — arm grasp + place in basket (from T5 checkpoint)

Observation space (~112-dim)
-----------------------------
vel(2)              chassis linear + angular velocity
lidar(72)           72-ray LiDAR, normalised [0,1]
nearest_3(6)        OracleDetector: [r_norm, sin_θ, pickable] × 3   (9 would be cleaner,
                    but kept at 6 for T6 compatibility — pickable merged into class_id)
coverage(1)         coverage ratio
ee_state(5)         EE pos_rel(3) + arm_reach_norm(1) + gripper_state(1)
target_rel_pos(3)   nearest pickable object relative to robot base
target_dist_norm(1) EE-to-target distance, clamped [0,1]
in_grasp_range(1)   1 if target within grasp threshold
basket_count_norm(1)placed count / num_pickable
holding(1)          1 if gripper holding
wrist_height(1)     wrist Z, normalised
fsm_state_onehot(3) one-hot FSM state: CLEAN / APPROACH / PICKUP

Action space (8-dim) — same as T5
-------------------------------------
[0]   linear_vel
[1]   angular_vel
[2:7] arm_joints_delta(5)
[7]   gripper
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from smartcleaningrobot.robot_assets.turtlebot4_with_arm import TURTLEBOT4_WITH_ARM_CFG
from smartcleaningrobot.sensors.lidar import LIDAR_2D_CFG


@configclass
class CoveragePickupEnvCfg(DirectRLEnvCfg):
    # -- env --
    decimation: int = 2
    episode_length_s: float = 300.0

    # -- spaces --
    # vel(2) + lidar(72) + detect(6) + coverage(1) + ee_state(5)
    # + target_rel(3) + target_dist(1) + in_grasp(1) + basket_norm(1)
    # + holding(1) + wrist_h(1) + fsm_onehot(3)  =  97
    # Pad to 112 to leave room for detector upgrade (YOLOv8 may add class info)
    action_space: int = 8
    observation_space: int = 112
    state_space: int = 0

    # -- simulation --
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # -- robot --
    robot_cfg: ArticulationCfg = TURTLEBOT4_WITH_ARM_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # -- sensors --
    lidar_cfg: RayCasterCfg = LIDAR_2D_CFG

    # -- scene --
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32, env_spacing=15.0, replicate_physics=True
    )
    scene_name: str = "room_with_objects"

    # -- coverage tracking --
    coverage_grid_resolution: float = 0.05
    robot_cleaning_radius: float = 0.06

    # -- object population: 6 pickable + 4 avoid (large obstacles) --
    num_objects: int = 10
    object_types: list = [
        "coin", "coin", "hairpin", "eraser", "cable", "coin",   # pickable
        "eraser", "eraser", "cable", "cable",                    # avoid (reuse eraser/cable as large proxy)
    ]
    # Which indices in object_types are pickable
    pickable_indices: list = [0, 1, 2, 3, 4, 5]
    num_pickable: int = 6

    # -- body / joint names --
    ee_body_name: str = "ee_gripper_link"
    arm_joint_names: list = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle"]
    gripper_joint_names: list = ["left_finger_link", "right_finger_link"]
    wheel_joint_names: list = ["wheel_left_joint", "wheel_right_joint"]

    # -- arm limits --
    joint_limits: list = [
        (-3.14,  3.14),
        (-1.88,  1.99),
        (-2.15,  1.57),
        (-3.14,  3.14),
        (-1.75,  2.15),
    ]

    # -- FSM thresholds --
    approach_trigger_dist: float = 1.5    # m — object within this → switch to APPROACH
    grasp_contact_threshold: float = 0.04 # m — EE within this → contact
    grasp_lift_height: float = 0.05       # m
    basket_offset: tuple = (0.0, -0.20, 0.15)
    basket_size: tuple = (0.18, 0.18, 0.12)

    # -- detection --
    oracle_fov_range: float = 2.0
    oracle_max_detections: int = 3

    # -- action scaling --
    max_linear_vel: float = 0.3
    max_angular_vel: float = 2.0
    max_joint_delta: float = 0.05
    gripper_threshold: float = 0.5

    # -- reward weights --
    rew_place_in_basket: float = 50.0
    rew_grasp_success: float = 10.0
    rew_coverage: float = 2.0
    rew_avoid_collision: float = 0.5       # bonus for passing within 0.3m of avoidable object
    rew_wall_collision: float = -3.0
    rew_obj_collision_avoid: float = -2.0  # hit a non-pickable object
    rew_obj_collision_pick: float = -0.5   # bump a pickable object (not during grasp)
    rew_drop_penalty: float = -5.0
    rew_time_penalty: float = -0.02

    # -- done condition --
    coverage_done_threshold: float = 0.70  # 70% clean OR all pickable placed
