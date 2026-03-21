"""Configuration for the ObjectPickup RL environment (T5).

Mobile manipulation: the robot navigates to a target object, grasps it
with the 5-DOF arm, and drops it into the rear basket.

Observation space (94-dim)
--------------------------
vel(2)             linear + angular velocity of robot chassis
lidar(72)          72-ray downsampled LiDAR, normalised [0,1]
nearest_3(6)       OracleDetector: 3 × [r_norm, sin_θ] for nearest objects
ee_state(5)        EE pos_rel(3) + arm_reach_norm(1) + gripper_state(1)
target_rel_pos(3)  target XYZ relative to robot base
target_dist_norm(1)EE-to-target distance, clamped [0,1]
in_grasp_range(1)  1 if target within grasp_contact_threshold
basket_count_norm(1) placed count / num_objects
holding(1)         1 if gripper holding an object
wrist_height(1)    wrist Z above ground, normalised

Action space (8-dim)
---------------------
[0]    linear velocity     scaled by max_linear_vel
[1]    angular velocity    scaled by max_angular_vel
[2:7]  arm_joints_delta(5) scaled by max_joint_delta (rad/step)
[7]    gripper             binarised at 0.5

Curriculum stages
-----------------
1 : Fine-tune from T4 checkpoint — chassis moves, arm frozen
    Target at fixed forward distance, just navigate to it
2 : Joint navigation + grasping, single object in random position
3 : Multiple objects, full loop (navigate → grasp → place → repeat)
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
class ObjectPickupEnvCfg(DirectRLEnvCfg):
    # -- env --
    decimation: int = 2
    episode_length_s: float = 180.0

    # -- spaces --
    # vel(2) + lidar(72) + nearest_3(6) + ee_state(5) + target_rel_pos(3)
    # + target_dist_norm(1) + in_grasp_range(1) + basket_count_norm(1) + holding(1) + wrist_height(1)
    action_space: int = 8
    observation_space: int = 94
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
        num_envs=64, env_spacing=15.0, replicate_physics=True
    )
    scene_name: str = "room_with_objects"

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

    # -- action scaling --
    max_linear_vel: float = 0.3
    max_angular_vel: float = 2.0
    max_joint_delta: float = 0.05
    gripper_threshold: float = 0.5

    # -- curriculum stage --
    curriculum_stage: int = 1

    # -- objects --
    num_objects: int = 5
    object_types: list = ["coin", "coin", "hairpin", "eraser", "cable"]

    # -- basket (rear of robot, ~20 cm behind base) --
    basket_offset: tuple = (0.0, -0.20, 0.15)   # relative to robot base
    basket_size: tuple = (0.18, 0.18, 0.12)

    # -- detection --
    oracle_fov_range: float = 2.0
    oracle_max_detections: int = 3

    # -- thresholds --
    approach_done_threshold: float = 0.20   # m — switch from APPROACH to GRASP
    grasp_contact_threshold: float = 0.04   # m — EE within this → contact
    grasp_lift_height: float = 0.05         # m — required lift for success

    # -- reward weights --
    rew_place_in_basket: float = 50.0
    rew_grasp_success: float = 10.0
    rew_approach_object: float = 1.0    # per step, proportional to dist reduction
    rew_coverage: float = 0.5           # per newly covered grid cell
    rew_wall_collision: float = -3.0
    rew_obj_collision: float = -1.0
    rew_drop_penalty: float = -5.0
    rew_time_penalty: float = -0.02
