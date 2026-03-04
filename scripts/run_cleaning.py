"""Run the cleaning robot in Isaac Sim for visualization and testing.

Supports two tasks:
    - coverage:     cleaning coverage (default)
    - maze_escape:  escape a DFS maze as fast as possible

Usage:
    python scripts/run_cleaning.py --scene hospital --policy zero --num_envs 1
    python scripts/run_cleaning.py --scene hospital --policy explorer --num_envs 1
    python scripts/run_cleaning.py --task maze_escape --policy random --num_envs 1
"""

from __future__ import annotations

import argparse
import importlib

from isaaclab.app import AppLauncher

_TASK_MAP = {
    "coverage": "CleaningRobot-Coverage-Direct-v0",
    "maze_escape": "CleaningRobot-MazeEscape-Direct-v0",
}

parser = argparse.ArgumentParser(description="Run cleaning robot demo")
parser.add_argument("--task", type=str, default="coverage", choices=list(_TASK_MAP.keys()),
                    help="Task to run: coverage (cleaning) or maze_escape")
parser.add_argument("--scene", type=str, default=None,
                    choices=["simple", "maze", "office", "hospital", "warehouse"])
parser.add_argument(
    "--policy",
    type=str,
    default="zero",
    choices=["zero", "random", "explorer"],
    help="zero: no motion, random: random actions, explorer: bump-and-turn policy",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=0, help="Stop after N steps (0=unlimited)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.scene is None:
    args_cli.scene = "maze" if args_cli.task == "maze_escape" else "simple"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import CleaningRobotV2  # noqa: F401

from CleaningRobotV2.traditional import RandomExplorerPolicy
from CleaningRobotV2.utils.map_viewer import render_coverage_map


def _resolve_entry_point(entry_point: str):
    """Resolve 'module.path:ClassName' string to the actual class."""
    module_path, class_name = entry_point.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    env_id = _TASK_MAP[args_cli.task]
    is_maze_escape = args_cli.task == "maze_escape"

    cfg_cls = _resolve_entry_point(
        gym.spec(env_id).kwargs["env_cfg_entry_point"]
    )
    env_cfg = cfg_cls()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene_name = args_cli.scene

    print(f"[run] Task: {args_cli.task} | Scene: {args_cli.scene}")
    print("[run] Creating environment ...")
    env = gym.make(env_id, cfg=env_cfg)
    print("[run] Environment created. Resetting ...")
    obs, info = env.reset()
    print(f"[run] Reset done. Obs shape: {obs['policy'].shape}")

    explorer = None
    if args_cli.policy == "explorer":
        explorer = RandomExplorerPolicy(
            num_envs=args_cli.num_envs, device=env.unwrapped.device
        )

    step = 0
    episode = 0
    max_test_steps = args_cli.max_steps

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

        if step % 100 == 0:
            pos = env.unwrapped.robot.data.root_pos_w[0].cpu()
            if is_maze_escape:
                exit_dist = env.unwrapped._prev_exit_dist[0].item()
                print(
                    f"  [step {step}] reward={reward[0].item():.4f}"
                    f" exit_dist={exit_dist:.3f}m"
                    f" pos=({pos[0]:.3f}, {pos[1]:.3f})"
                )
            else:
                coverage = env.unwrapped.coverage_tracker.get_coverage_ratio()
                print(
                    f"  [step {step}] reward={reward[0].item():.4f}"
                    f" coverage={coverage[0]:.2%}"
                    f" pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                )

        if terminated.any() or truncated.any():
            if is_maze_escape:
                escaped = terminated.any().item()
                status = "ESCAPED" if escaped else "TIMEOUT"
                print(f"[Episode {episode}] Step {step} | {status}")
            else:
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

        if max_test_steps > 0 and step >= max_test_steps:
            if is_maze_escape:
                print(f"[Test complete] {max_test_steps} steps")
            else:
                coverage = env.unwrapped.coverage_tracker.get_coverage_ratio()
                print(f"[Test complete] {max_test_steps} steps | Coverage: {coverage[0]:.1%}")
            break

    env.close()


if __name__ == "__main__":
    main()
