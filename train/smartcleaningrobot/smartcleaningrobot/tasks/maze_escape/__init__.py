import gymnasium as gym

from . import agents

gym.register(
    id="SmartCleaningRobot-MazeEscape-v0",
    entry_point=f"{__name__}.env:MazeEscapeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:MazeEscapeEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:sac.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:sac.yaml",
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:ppo.yaml",
    },
)
