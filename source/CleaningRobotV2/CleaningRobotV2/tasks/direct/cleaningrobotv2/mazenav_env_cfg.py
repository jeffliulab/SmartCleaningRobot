"""Configuration for the maze escape RL environment.

This task is SEPARATE from the cleaning/coverage task.  The agent must
navigate a procedural DFS maze and reach the exit as fast as possible.

Observation (44 dims):
    pose(3) + vel(2) + lidar(36) + exit_dir(2) + exit_dist(1)

Action (2 dims):
    linear velocity, angular velocity  (same as coverage task)
"""

from __future__ import annotations

from CleaningRobotV2.robots.jetbot import JETBOT_CFG, JETBOT_LIDAR_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class MazeEscapeEnvCfg(DirectRLEnvCfg):
    # -- env --
    decimation = 2
    episode_length_s = 120.0

    # -- spaces --
    # pose(3) + vel(2) + lidar(36) + exit_dir(2) + exit_dist(1) = 44
    action_space = 2
    observation_space = 44
    state_space = 0

    # -- simulation --
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # -- robot --
    robot_cfg: ArticulationCfg = JETBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # -- sensors (camera disabled — not needed for maze escape) --
    lidar_cfg: RayCasterCfg = JETBOT_LIDAR_CFG

    # -- scene --
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=15.0, replicate_physics=True
    )
    scene_name: str = "maze"

    # -- coverage tracking (used only for exploration bonus) --
    coverage_grid_resolution: float = 0.05
    robot_cleaning_radius: float = 0.06

    # -- action scaling --
    max_linear_vel: float = 0.3
    max_angular_vel: float = 2.0

    # -- maze exit --
    exit_pos: tuple[float, float] = (0.0, -3.15)
    exit_threshold: float = 0.3
    maze_diagonal: float = 5.7

    # -- reward weights --
    rew_dist_reduction: float = 5.0
    rew_exit_bonus: float = 200.0
    rew_collision_penalty: float = -1.0
    rew_time_penalty: float = -0.05
    rew_exploration: float = 0.5
    collision_threshold: float = 0.15

    # -- spawn --
    randomize_spawn: bool = True
