"""Configuration for the cleaning robot coverage RL environment.

Defines observation/action spaces, reward weights, simulation parameters,
and references to robot/sensor/scene configurations.
"""

from __future__ import annotations

from CleaningRobotV2.robots.jetbot import JETBOT_CAMERA_CFG, JETBOT_CFG, JETBOT_LIDAR_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, RayCasterCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class Cleaningrobotv2EnvCfg(DirectRLEnvCfg):
    # -- env --
    decimation = 2
    episode_length_s = 300.0

    # -- spaces --
    # pose(3) + vel(2) + lidar(36) + local_coverage(32*32=1024) + coverage_ratio(1)
    action_space = 2
    observation_space = 1066
    state_space = 0

    # -- simulation --
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # -- robot --
    robot_cfg: ArticulationCfg = JETBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # -- sensors --
    lidar_cfg: RayCasterCfg = JETBOT_LIDAR_CFG
    camera_cfg: CameraCfg = JETBOT_CAMERA_CFG

    # -- scene --
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=25.0, replicate_physics=True
    )
    scene_name: str = "hospital"

    # -- coverage tracking --
    coverage_grid_resolution: float = 0.05
    robot_cleaning_radius: float = 0.06
    coverage_local_view_size: int = 32

    # -- action scaling --
    max_linear_vel: float = 0.3
    max_angular_vel: float = 2.0

    # -- reward weights --
    rew_coverage_weight: float = 10.0
    rew_collision_penalty: float = -5.0
    rew_time_penalty: float = -0.01
    rew_milestone_bonus: float = 50.0
    collision_threshold: float = 0.15

    # -- done conditions --
    coverage_done_threshold: float = 0.98
