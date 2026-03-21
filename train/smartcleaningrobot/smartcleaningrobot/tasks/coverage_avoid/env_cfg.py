"""Configuration for the CoverageAvoid RL environment (T6).

The robot (JetBot, no arm) cleans the floor while actively avoiding small
floor objects.  Small objects detected via OracleDetector during training;
replaced by YOLOv8-nano for real deployment.

Observation space (81-dim)
--------------------------
vel(2)         chassis linear + angular velocity
lidar(72)      72-ray LiDAR, normalised [0,1]
nearest_3(6)   OracleDetector: 3 × [r_norm, sin_θ] for nearest objects
coverage(1)    current coverage ratio [0,1]

Action space (2-dim)
---------------------
[0] linear_vel   scaled by max_linear_vel
[1] angular_vel  scaled by max_angular_vel
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from smartcleaningrobot.robot_assets.jetbot import JETBOT_CFG
from smartcleaningrobot.sensors.lidar import LIDAR_2D_CFG


@configclass
class CoverageAvoidEnvCfg(DirectRLEnvCfg):
    # -- env --
    decimation: int = 2
    episode_length_s: float = 120.0

    # -- spaces --
    # vel(2) + lidar(72) + nearest_3_objects(6) + coverage_ratio(1)
    action_space: int = 2
    observation_space: int = 81
    state_space: int = 0

    # -- simulation --
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # -- robot --
    robot_cfg: ArticulationCfg = JETBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # -- sensors --
    lidar_cfg: RayCasterCfg = LIDAR_2D_CFG

    # -- scene --
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=15.0, replicate_physics=True
    )
    scene_name: str = "room_with_objects"

    # -- coverage tracking --
    coverage_grid_resolution: float = 0.05
    robot_cleaning_radius: float = 0.06

    # -- small objects --
    num_objects: int = 10
    object_types: list = [
        "coin", "coin", "coin",
        "hairpin", "hairpin",
        "cable", "cable",
        "eraser", "eraser", "eraser",
    ]

    # -- detection --
    oracle_fov_range: float = 2.0
    oracle_max_detections: int = 3

    # -- action scaling --
    max_linear_vel: float = 0.3
    max_angular_vel: float = 2.0

    # -- collision thresholds --
    wall_collision_threshold: float = 0.20   # m — same as fixed MazeEscape
    obj_collision_threshold: float = 0.08    # m — object bounding radius + safety

    # -- reward weights --
    rew_coverage: float = 2.0
    rew_wall_collision: float = -3.0
    rew_obj_collision: float = -2.0
    rew_forward_bonus: float = 0.3
    rew_time_penalty: float = -0.02

    # -- done condition --
    coverage_done_threshold: float = 0.80   # episode ends early at 80% coverage
