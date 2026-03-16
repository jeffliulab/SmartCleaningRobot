"""MazeDFSv1 + Jetbot: escape DFS maze demo in Isaac Sim.

Scene:  DFS-generated 8×8 maze (seed=42)
Robot:  Jetbot (differential drive)
Task:   Navigate to the exit as fast as possible

Usage:
    python train/scripts/MazeDFSv1_Jetbot/MazeDFSv1_Jetbot_EscapeMaze.py --policy random --num_envs 1
    python train/scripts/MazeDFSv1_Jetbot/MazeDFSv1_Jetbot_EscapeMaze.py --policy zero --num_envs 1 --headless --max_steps 200
"""

from __future__ import annotations

import argparse
import importlib

from isaaclab.app import AppLauncher

ENV_ID = "SmartCleaningRobot-MazeEscape-v0"

parser = argparse.ArgumentParser(description="MazeDFSv1 Jetbot escape-maze demo")
parser.add_argument("--scene", type=str, default="maze",
                    choices=["maze"])
parser.add_argument(
    "--policy",
    type=str,
    default="random",
    choices=["zero", "random"],
    help="zero: no motion, random: random actions",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=0, help="Stop after N steps (0=unlimited)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import smartcleaningrobot  # noqa: F401


def _resolve_entry_point(entry_point: str):
    """Resolve 'module.path:ClassName' string to the actual class."""
    module_path, class_name = entry_point.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    cfg_cls = _resolve_entry_point(
        gym.spec(ENV_ID).kwargs["env_cfg_entry_point"]
    )
    env_cfg = cfg_cls()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene_name = args_cli.scene

    print(f"[run] Task: MazeEscape | Scene: {args_cli.scene}")
    print("[run] Creating environment ...")
    env = gym.make(ENV_ID, cfg=env_cfg)
    print("[run] Environment created. Resetting ...")
    obs, info = env.reset()
    print(f"[run] Reset done. Obs shape: {obs['policy'].shape}")

    step = 0
    episode = 0
    max_test_steps = args_cli.max_steps

    while simulation_app.is_running():
        with torch.inference_mode():
            if args_cli.policy == "random":
                action = torch.randn(args_cli.num_envs, 2, device=env.unwrapped.device) * 0.5
            else:
                action = torch.zeros(args_cli.num_envs, 2, device=env.unwrapped.device)

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        if step % 100 == 0:
            pos = env.unwrapped.robot.data.root_pos_w[0].cpu()
            exit_dist = env.unwrapped._prev_exit_dist[0].item()
            print(
                f"  [step {step}] reward={reward[0].item():.4f}"
                f" exit_dist={exit_dist:.3f}m"
                f" pos=({pos[0]:.3f}, {pos[1]:.3f})"
            )

        if terminated.any() or truncated.any():
            escaped = terminated.any().item()
            status = "ESCAPED" if escaped else "TIMEOUT"
            print(f"[Episode {episode}] Step {step} | {status}")
            obs, info = env.reset()
            step = 0
            episode += 1

        if max_test_steps > 0 and step >= max_test_steps:
            print(f"[Test complete] {max_test_steps} steps")
            break

    env.close()


if __name__ == "__main__":
    main()
