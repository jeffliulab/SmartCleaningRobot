"""Shared utility functions for task environments.

Extracted from T5/T6/T7 env.py to eliminate code duplication.
This is an internal module — not registered as a Gym task.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import sample_uniform


def process_lidar(
    lidar: RayCaster,
    max_distance: float,
    downsample: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute normalised LiDAR distances from a RayCaster sensor.

    Returns:
        lidar_obs: (N, num_rays // downsample) normalised distances in [0, 1].
        lidar_dist: (N, num_rays) raw distances (useful for reward computation).
    """
    sensor_pos = lidar.data.pos_w.unsqueeze(1)
    lidar_dist = torch.norm(lidar.data.ray_hits_w - sensor_pos, dim=-1)
    lidar_obs = (lidar_dist[:, ::downsample] / max_distance).clamp(0.0, 1.0)
    return lidar_obs, lidar_dist


def detect_objects(
    detector,
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
    objects: list[RigidObject],
    *,
    object_indices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run OracleDetector on a set of rigid objects.

    Args:
        detector: OracleDetector instance.
        base_pos_w: (N, 3) robot root position.
        base_quat_w: (N, 4) robot root quaternion.
        objects: list of RigidObject instances.
        object_indices: optional subset of indices to detect (e.g. pickable only).

    Returns:
        detect_obs: (N, K*2) detector output.
        all_pos: (N, M, 3) stacked object positions (M = len(objects) or len(object_indices)).
    """
    all_pos = torch.stack([o.data.root_pos_w for o in objects], dim=1)  # (N, M, 3)
    yaw = 2.0 * torch.atan2(base_quat_w[:, 3], base_quat_w[:, 0])

    if object_indices is not None:
        detect_pos = all_pos[:, object_indices, :]
    else:
        detect_pos = all_pos

    detect_obs = detector.detect_batch(base_pos_w[:, :2], detect_pos, yaw)
    return detect_obs, all_pos


def randomize_object_positions(
    objects: list[RigidObject],
    env_ids: Sequence[int],
    scene_bounds: tuple[float, float, float, float],
    env_origins: torch.Tensor,
    min_separation: float,
    device: torch.device | str,
) -> None:
    """Place objects at random non-overlapping positions via rejection sampling.

    Args:
        objects: list of RigidObject to reposition.
        env_ids: environment indices to reset.
        scene_bounds: (x_min, y_min, x_max, y_max).
        env_origins: (total_envs, 3) world-frame origins per env.
        min_separation: minimum distance between any two objects.
        device: torch device.
    """
    n = len(env_ids)
    x_min, y_min, x_max, y_max = scene_bounds
    safe = 0.5

    placed: list[torch.Tensor] = []
    for i, obj in enumerate(objects):
        state = obj.data.default_root_state[env_ids].clone()
        xy = torch.zeros(n, 2, device=device)
        for b in range(n):
            for _ in range(50):
                x = float(sample_uniform(x_min * safe, x_max * safe, (1,), "cpu"))
                y = float(sample_uniform(y_min * safe, y_max * safe, (1,), "cpu"))
                candidate = torch.tensor([x, y], device=device)
                ok = True
                for prev in placed:
                    if torch.norm(candidate - prev[b]).item() < min_separation:
                        ok = False
                        break
                if ok:
                    break
            xy[b] = candidate
        placed.append(xy)

        state[:, 0] = env_origins[env_ids, 0] + xy[:, 0]
        state[:, 1] = env_origins[env_ids, 1] + xy[:, 1]
        state[:, 2] = env_origins[env_ids, 2] + 0.01
        state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        state[:, 7:] = 0.0
        obj.write_root_pose_to_sim(state[:, :7], env_ids)
        obj.write_root_velocity_to_sim(state[:, 7:], env_ids)


def apply_diff_drive(
    actions: torch.Tensor,
    max_linear_vel: float,
    max_angular_vel: float,
    robot: Articulation,
    wheel_ids: list[int],
    diff_drive_fn,
) -> None:
    """Convert policy actions[0:2] to wheel velocities and apply.

    Args:
        actions: (N, A) full action tensor (only columns 0 and 1 used).
        max_linear_vel: scaling factor for linear velocity.
        max_angular_vel: scaling factor for angular velocity.
        robot: Articulation to command.
        wheel_ids: joint IDs for [left_wheel, right_wheel].
        diff_drive_fn: callable(lin, ang) -> (left, right) wheel angular velocities.
    """
    lin = actions[:, 0] * max_linear_vel
    ang = actions[:, 1] * max_angular_vel
    left, right = diff_drive_fn(lin, ang)
    robot.set_joint_velocity_target(
        torch.stack([left, right], dim=-1), joint_ids=wheel_ids
    )


def compute_wall_collision(
    lidar_dist: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Return binary collision flag from LiDAR distances.

    Args:
        lidar_dist: (N, num_rays) raw LiDAR distances.
        threshold: distance below which collision is detected.

    Returns:
        (N,) float tensor: 1.0 where min distance < threshold, else 0.0.
    """
    lidar_min = lidar_dist.min(dim=-1).values
    return (lidar_min < threshold).float()
