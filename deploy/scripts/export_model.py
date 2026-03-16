"""Export a trained skrl checkpoint to a standalone TorchScript model.

After training completes, run this script to convert the skrl checkpoint
into a self-contained .pt file that policy_node.py can load directly.

Usage (cleaning task — PPO, 1066-dim obs):
    python deploy/scripts/export_model.py \
        --checkpoint logs/skrl/cleaning_coverage/.../checkpoints/best_agent.pt \
        --output deploy/models/best_agent_scripted.pt

Usage (maze task — SAC, 41-dim obs):
    python deploy/scripts/export_model.py \
        --task maze \
        --checkpoint logs/skrl/maze_escape/.../checkpoints/best_agent.pt \
        --output deploy/models/maze_agent_scripted.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

# ---- Task-specific configurations ----------------------------------------

TASK_CONFIGS = {
    "cleaning": {
        "obs_dim": 1066,
        "act_dim": 2,
        "hidden": [256, 128, 64],
    },
    "maze": {
        "obs_dim": 41,
        "act_dim": 2,
        "hidden": [256, 128, 64],
    },
}


class PolicyNet(nn.Module):
    """Recreates the policy network architecture used by skrl.

    The layer sizes MUST match the YAML training config for the chosen task.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ELU())
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def export(checkpoint_path: str, output_path: str, task: str) -> None:
    cfg = TASK_CONFIGS[task]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    policy = PolicyNet(cfg["obs_dim"], cfg["act_dim"], cfg["hidden"])

    state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint
    if "policy" in state_dict:
        state_dict = state_dict["policy"]

    try:
        policy.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {checkpoint_path} (task={task})")
    except Exception as e:
        print(f"Warning: could not auto-load weights: {e}")
        print("Listing checkpoint keys for manual mapping:")
        for k, v in state_dict.items():
            print(f"  {k:50s}  {tuple(v.shape)}")
        print("\nYou may need to adjust the key mapping in this script.")
        return

    policy.eval()
    dummy = torch.zeros(1, cfg["obs_dim"])
    scripted = torch.jit.trace(policy, dummy)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    scripted.save(output_path)
    print(f"Exported TorchScript model -> {output_path}")

    loaded = torch.jit.load(output_path)
    out = loaded(dummy)
    print(f"Verification: input {tuple(dummy.shape)} -> output {tuple(out.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Export skrl checkpoint -> TorchScript")
    parser.add_argument("--checkpoint", required=True, help="Path to skrl best_agent.pt")
    parser.add_argument(
        "--output",
        default="deploy/models/best_agent_scripted.pt",
        help="Output TorchScript path",
    )
    parser.add_argument(
        "--task",
        default="cleaning",
        choices=list(TASK_CONFIGS.keys()),
        help="Task type: 'cleaning' (1066-dim) or 'maze' (41-dim)",
    )
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.task)


if __name__ == "__main__":
    main()
