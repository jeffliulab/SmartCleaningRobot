"""Export a trained skrl PPO checkpoint to a standalone TorchScript model.

After training completes, run this script to convert the skrl checkpoint
into a self-contained .pt file that policy_node.py can load directly.

Usage:
    python deployment/scripts/export_model.py \
        --checkpoint logs/skrl/cleaning_coverage/.../checkpoints/best_agent.pt \
        --output deployment/models/best_agent_scripted.pt

The exported model takes a (1, 1066) observation tensor and returns a (1, 2)
action tensor: [normalized_linear_vel, normalized_angular_vel] in [-1, 1].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


OBS_DIM = 1066
ACT_DIM = 2
HIDDEN_LAYERS = [256, 128, 64]


class PolicyNet(nn.Module):
    """Recreates the policy network architecture used by skrl PPO.

    The layer sizes MUST match the skrl_ppo_cfg.yaml training config.
    """

    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = []
        prev = OBS_DIM
        for h in HIDDEN_LAYERS:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ELU())
            prev = h
        layers.append(nn.Linear(prev, ACT_DIM))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def export(checkpoint_path: str, output_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    policy = PolicyNet()

    state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint
    if "policy" in state_dict:
        state_dict = state_dict["policy"]

    try:
        policy.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {checkpoint_path}")
    except Exception as e:
        print(f"Warning: could not auto-load weights: {e}")
        print("Listing checkpoint keys for manual mapping:")
        for k, v in state_dict.items():
            print(f"  {k:50s}  {tuple(v.shape)}")
        print("\nYou may need to adjust the key mapping in this script.")
        return

    policy.eval()
    dummy = torch.zeros(1, OBS_DIM)
    scripted = torch.jit.trace(policy, dummy)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    scripted.save(output_path)
    print(f"Exported TorchScript model → {output_path}")

    loaded = torch.jit.load(output_path)
    out = loaded(dummy)
    print(f"Verification: input {tuple(dummy.shape)} → output {tuple(out.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Export skrl checkpoint → TorchScript")
    parser.add_argument("--checkpoint", required=True, help="Path to skrl best_agent.pt")
    parser.add_argument(
        "--output",
        default="deployment/models/best_agent_scripted.pt",
        help="Output TorchScript path",
    )
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
