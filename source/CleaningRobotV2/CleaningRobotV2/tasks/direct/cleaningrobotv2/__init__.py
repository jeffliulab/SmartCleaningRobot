# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="CleaningRobot-Coverage-Direct-v0",
    entry_point=f"{__name__}.cleaningrobotv2_env:Cleaningrobotv2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cleaningrobotv2_env_cfg:Cleaningrobotv2EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="CleaningRobot-MazeEscape-Direct-v0",
    entry_point=f"{__name__}.mazenav_env:MazeEscapeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mazenav_env_cfg:MazeEscapeEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_maze_ppo_cfg.yaml",
    },
)
