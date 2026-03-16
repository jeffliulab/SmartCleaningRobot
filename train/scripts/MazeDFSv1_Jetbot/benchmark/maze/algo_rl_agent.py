"""RL agent wrapper for maze benchmark.

Loads a trained TorchScript model exported from skrl SAC training
and runs inference using the same 41-dim observation space.

This algorithm has **learned behavior** from millions of simulation steps.
It uses LiDAR + exit direction (body frame) as input, no map knowledge.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class RLAgentPolicy:
    """Wrapper for a trained RL policy (TorchScript or raw skrl checkpoint).

    Accepts the 41-dim observation vector and outputs (linear_vel, angular_vel).
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for RL agent inference.")

        self._device = torch.device(device)

        # Try loading as TorchScript first, then as raw checkpoint
        try:
            self._model = torch.jit.load(model_path, map_location=self._device)
            self._model.eval()
            self._mode = "torchscript"
            print(f"[RL Agent] Loaded TorchScript model from {model_path}")
        except Exception:
            # Try loading as skrl checkpoint (dict with 'policy' key)
            ckpt = torch.load(model_path, map_location=self._device, weights_only=False)
            if isinstance(ckpt, dict) and "policy" in ckpt:
                self._state_dict = ckpt["policy"]
                self._mode = "state_dict"
                print(f"[RL Agent] Loaded state dict from {model_path}")
                print("  NOTE: State dict mode requires manual network construction.")
                print("  For best results, export to TorchScript first:")
                print(f"    python deploy/scripts/export_model.py --task maze "
                      f"--checkpoint {model_path} --output maze_agent_scripted.pt")
            else:
                raise ValueError(
                    f"Cannot load model from {model_path}. "
                    "Expected TorchScript (.pt) or skrl checkpoint with 'policy' key."
                )

        # Running statistics for observation normalization (if available)
        self._obs_mean = None
        self._obs_std = None
        if isinstance(ckpt, dict) if self._mode == "state_dict" else False:
            if "state_preprocessor" in ckpt:
                sp = ckpt["state_preprocessor"]
                if "running_mean" in sp:
                    self._obs_mean = sp["running_mean"].to(self._device)
                    self._obs_std = sp["running_var"].sqrt().clamp(min=1e-6).to(self._device)

    def compute_action(self, obs_41: np.ndarray) -> tuple[float, float]:
        """Given 41-dim observation, return (linear_vel, angular_vel) in m/s and rad/s.

        Output is already scaled to physical units:
            linear:  action[0] * 0.3  -> [-0.3, 0.3] m/s
            angular: action[1] * 2.0  -> [-2.0, 2.0] rad/s
        """
        with torch.no_grad():
            obs_t = torch.tensor(obs_41, dtype=torch.float32, device=self._device).unsqueeze(0)

            # Apply normalization if available
            if self._obs_mean is not None:
                obs_t = (obs_t - self._obs_mean) / self._obs_std

            if self._mode == "torchscript":
                action = self._model(obs_t)
                if isinstance(action, dict):
                    action = action.get("action", action.get("actions", list(action.values())[0]))
                elif isinstance(action, (tuple, list)):
                    action = action[0]
            else:
                raise NotImplementedError(
                    "Raw state dict inference not implemented. "
                    "Please export to TorchScript first."
                )

            action = action.squeeze(0).cpu().numpy()

        # Clamp to [-1, 1] and scale to physical units
        action = np.clip(action, -1.0, 1.0)
        linear_vel = float(action[0]) * 0.3   # max_linear_vel
        angular_vel = float(action[1]) * 2.0  # max_angular_vel

        return linear_vel, angular_vel

    def reset(self) -> None:
        """No internal state to reset for the RL agent."""
        pass


class DummyRLPolicy:
    """Placeholder policy when no trained model is available.

    Uses a reactive heuristic: steer toward exit direction with LiDAR-based
    obstacle avoidance. When blocked, scan for the best open direction
    biased toward the exit.

    Replace with a real trained model for meaningful RL benchmark results:
        python run_benchmark.py --rl_model path/to/maze_agent_scripted.pt
    """

    def __init__(self):
        self._fwd = 0.12
        self._kp = 3.0
        self._step_count = 0

    def compute_action(self, obs_41: np.ndarray) -> tuple[float, float]:
        """Heuristic: steer toward exit, avoid walls reactively."""
        lidar = obs_41[2:38]  # 36 rays, normalized [0,1]
        exit_sin = obs_41[38]
        exit_cos = obs_41[39]
        self._step_count += 1

        # Exit bearing in body frame
        exit_angle = float(np.arctan2(exit_sin, exit_cos))

        # Compute repulsive force from nearby walls
        repulsive_angular = 0.0
        min_front = 1.0
        for i in range(36):
            d = float(lidar[i])
            ray_angle = i * (2 * math.pi / 36)
            if ray_angle > math.pi:
                ray_angle -= 2 * math.pi

            # Track front minimum (±30 deg)
            if abs(ray_angle) < math.radians(30):
                min_front = min(min_front, d)

            # Repulsive force from close walls
            if d < 0.12:  # ~0.42m
                strength = (0.12 - d) / 0.12
                if ray_angle > 0:
                    repulsive_angular -= strength * 2.0  # wall on left -> steer right
                else:
                    repulsive_angular += strength * 2.0  # wall on right -> steer left

        # Attractive force toward exit
        attractive_angular = self._kp * exit_angle

        # Blend: when walls are close, give more weight to avoidance
        wall_weight = max(0.0, 1.0 - min_front / 0.15)
        angular = (1.0 - wall_weight) * attractive_angular + wall_weight * repulsive_angular

        # Add small noise to escape local minima
        if self._step_count % 120 == 0:
            angular += (1.0 if (self._step_count // 120) % 2 == 0 else -1.0) * 0.5

        angular = max(-2.0, min(2.0, angular))

        # Speed: slow when turning or near walls
        if min_front < 0.06:
            linear = 0.0
        elif min_front < 0.10:
            linear = self._fwd * 0.3
        elif abs(angular) > 1.5:
            linear = self._fwd * 0.3
        else:
            linear = self._fwd

        return float(linear), float(angular)

    def reset(self) -> None:
        self._step_count = 0


# Convenience: auto-select based on whether model path is provided
def make_rl_policy(model_path: str | None = None, device: str = "cpu"):
    """Create RL policy — real model if path given, dummy otherwise."""
    if model_path is not None:
        return RLAgentPolicy(model_path, device)
    else:
        print("[RL Agent] No model path provided, using DummyRLPolicy (heuristic)")
        return DummyRLPolicy()


# Need math for DummyRLPolicy
import math
