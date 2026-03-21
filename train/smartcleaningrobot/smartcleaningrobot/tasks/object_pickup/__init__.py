import gymnasium as gym

from . import agents

gym.register(
    id="SmartCleaningRobot-ObjectPickup-v0",
    entry_point=f"{__name__}.env:ObjectPickupEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ObjectPickupEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo.yaml",
    },
)
