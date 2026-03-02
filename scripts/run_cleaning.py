"""Run the cleaning robot in Isaac Sim for visualization and testing.

Usage:
    isaaclab -p scripts/run_cleaning.py --scene office
    isaaclab -p scripts/run_cleaning.py --scene hospital --policy explorer
    isaaclab -p scripts/run_cleaning.py --scene office --policy random
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run cleaning robot demo")
parser.add_argument("--scene", type=str, default="hospital", choices=["office", "hospital", "warehouse"])
parser.add_argument(
    "--policy",
    type=str,
    default="zero",
    choices=["zero", "random", "explorer"],
    help="zero: no motion, random: random actions, explorer: bump-and-turn policy",
)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import CleaningRobotV2  # noqa: F401

from CleaningRobotV2.traditional import RandomExplorerPolicy
from CleaningRobotV2.utils.map_viewer import render_coverage_map


def main():
    env_cfg = gym.spec("CleaningRobot-Coverage-Direct-v0").kwargs["env_cfg_entry_point"]()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene_name = args_cli.scene

    env = gym.make("CleaningRobot-Coverage-Direct-v0", cfg=env_cfg)
    obs, info = env.reset()

    explorer = None
    if args_cli.policy == "explorer":
        explorer = RandomExplorerPolicy(
            num_envs=args_cli.num_envs, device=env.unwrapped.device
        )

    step = 0
    episode = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            if args_cli.policy == "explorer":
                action = explorer.compute_action(obs["policy"])
            elif args_cli.policy == "random":
                action = torch.randn(args_cli.num_envs, 2, device=env.unwrapped.device) * 0.5
            else:
                action = torch.zeros(args_cli.num_envs, 2, device=env.unwrapped.device)

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        if terminated.any() or truncated.any():
            coverage = env.unwrapped.coverage_tracker.get_coverage_ratio()
            print(f"[Episode {episode}] Step {step} | Coverage: {coverage[0]:.1%}")

            grid = env.unwrapped.coverage_tracker.get_full_map()[0].cpu().numpy()
            render_coverage_map(
                grid,
                save_path=f"coverage_maps/episode_{episode}_{args_cli.scene}.png",
            )

            obs, info = env.reset()
            if explorer is not None:
                explorer.reset(list(range(args_cli.num_envs)))
            step = 0
            episode += 1

    env.close()


if __name__ == "__main__":
    main()
