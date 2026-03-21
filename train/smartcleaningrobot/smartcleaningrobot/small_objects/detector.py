"""Small object detectors for the SmartCleaningRobot project.

Provides a unified Detection interface and three concrete implementations:

    OracleDetector     — Phase A: reads object positions directly from the
                         simulator (perfect, zero-latency, simulation-only).
    ColorBlobDetector  — Phase B placeholder: HSV colour segmentation from RGB.
    CameraDetector     — Phase C placeholder: YOLOv8-nano from RGB image.

All detectors share the BaseDetector ABC and the Detection dataclass so that
RL policy code only depends on the interface, not the implementation.
Phase A → C replacement requires changing only the detector_class in config.

Batched RL interface
--------------------
OracleDetector.detect_batch() accepts (N, 2) robot positions + (N, M, 3)
object world positions and returns an (N, max_detections * 2) tensor of
[r_norm, sin_θ] pairs, compatible with the ObjectAvoidEnv observation vector.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# Unified data container
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single detected object from the robot's perspective."""

    class_id: int
    class_name: str
    confidence: float
    rel_x: float      # body-frame forward axis (positive = ahead)
    rel_y: float      # body-frame left axis (positive = left)
    distance: float   # Euclidean distance (m)
    pickable: bool    # True if the arm can attempt to grasp this object


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseDetector(ABC):
    """Common interface implemented by all detector phases."""

    @abstractmethod
    def detect(self, *args, **kwargs) -> list[Detection]:
        """Return detections for a single environment (deployment interface)."""
        ...


# ---------------------------------------------------------------------------
# OracleDetector — Phase A
# ---------------------------------------------------------------------------

class OracleDetector(BaseDetector):
    """Simulation-only detector: reads object world positions from the sim.

    Zero noise, zero latency.  Used during Phase A training to validate RL
    strategies without vision-processing overhead.
    Replace with CameraDetector(YOLOv8) for Phase C deployment.

    Args:
        fov_range:       Maximum detection distance in metres (default 2.0 m).
        max_detections:  Number of objects returned per environment.
                         Slots beyond detected count are zero-padded.
    """

    def __init__(self, fov_range: float = 2.0, max_detections: int = 3) -> None:
        self.fov_range = fov_range
        self.max_detections = max_detections

    # ------------------------------------------------------------------
    # Batched RL interface — returns tensor for obs vector assembly
    # ------------------------------------------------------------------

    def detect_batch(
        self,
        robot_positions: torch.Tensor,   # (N, 2)  world XY
        object_positions: torch.Tensor,  # (N, M, 3) world XYZ
        robot_yaw: torch.Tensor,         # (N,)  radians
    ) -> torch.Tensor:                   # (N, max_detections * 2)
        """Compute nearest-K object detections for all N environments.

        Returns:
            Tensor of shape ``(N, max_detections * 2)`` where each pair
            ``[r_norm, sin_θ]`` encodes::

                r_norm  = distance / fov_range  ∈ [0, 1]
                sin_θ   = sin(bearing)           ∈ [-1, 1]
                          (positive = object to the left)

            Pairs beyond detected objects within fov_range are **zero**.
        """
        N = robot_positions.shape[0]
        M = object_positions.shape[1]
        device = robot_positions.device

        # Relative XY displacement in world frame: (N, M, 2)
        robot_xy = robot_positions.unsqueeze(1)                     # (N, 1, 2)
        obj_xy = object_positions[:, :, :2]                          # (N, M, 2)
        delta = obj_xy - robot_xy                                    # (N, M, 2)

        # Distance: (N, M)
        distances = torch.norm(delta, dim=-1)

        # Bearing in robot body frame: angle from forward axis, CCW positive
        # world_angle = atan2(dy, dx) — angle of object in world frame
        world_angle = torch.atan2(delta[:, :, 1], delta[:, :, 0])  # (N, M)
        bearing = world_angle - robot_yaw.unsqueeze(1)              # (N, M)
        # Wrap to [-π, π]
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi

        # Mask objects beyond fov_range — treat as infinitely far
        beyond = distances > self.fov_range
        distances_masked = distances.clone()
        distances_masked[beyond] = float("inf")

        # Sort by distance (closest first): (N, M)
        sorted_idx = torch.argsort(distances_masked, dim=-1)
        sorted_dist = torch.gather(distances_masked, 1, sorted_idx)
        sorted_bear = torch.gather(bearing, 1, sorted_idx)

        K = self.max_detections
        out = torch.zeros(N, K * 2, device=device)

        n_fill = min(K, M)
        valid = sorted_dist[:, :n_fill] < float("inf")   # (N, n_fill)

        # r_norm = distance / fov_range, clamped to [0, 1]
        r_norm = (sorted_dist[:, :n_fill] / self.fov_range).clamp(0.0, 1.0)
        sin_theta = torch.sin(sorted_bear[:, :n_fill])

        out[:, 0:n_fill * 2:2] = torch.where(valid, r_norm, torch.zeros_like(r_norm))
        out[:, 1:n_fill * 2:2] = torch.where(valid, sin_theta, torch.zeros_like(sin_theta))

        return out

    # ------------------------------------------------------------------
    # Single-env deployment interface (wraps detect_batch)
    # ------------------------------------------------------------------

    def detect(
        self,
        robot_position: torch.Tensor,   # (2,) or (1, 2)
        object_positions: torch.Tensor, # (M, 3) or (1, M, 3)
        robot_yaw: torch.Tensor,        # scalar or (1,)
        class_names: list[str] | None = None,
    ) -> list[Detection]:
        """Single-environment wrapper for use in deployment / evaluation."""
        rp = robot_position.view(1, 2)
        op = object_positions.view(1, -1, 3)
        ry = robot_yaw.view(1)

        batch_out = self.detect_batch(rp, op, ry)  # (1, K*2)
        M = op.shape[1]
        delta = op[0, :, :2] - rp[0]
        distances = torch.norm(delta, dim=-1)

        detections: list[Detection] = []
        for k in range(self.max_detections):
            r_norm = batch_out[0, k * 2].item()
            if r_norm == 0.0:
                break
            dist = r_norm * self.fov_range
            # Re-find which object this corresponds to (by distance match)
            closest = int(torch.argmin(torch.abs(distances - dist)).item())
            world_angle = torch.atan2(delta[closest, 1], delta[closest, 0]).item()
            bearing = world_angle - robot_yaw.item()
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
            rel_x = dist * math.cos(bearing)
            rel_y = dist * math.sin(bearing)
            name = class_names[closest] if class_names else "unknown"
            detections.append(
                Detection(
                    class_id=closest,
                    class_name=name,
                    confidence=1.0,
                    rel_x=float(rel_x),
                    rel_y=float(rel_y),
                    distance=float(dist),
                    pickable=True,
                )
            )
        return detections


# ---------------------------------------------------------------------------
# ColorBlobDetector — Phase B placeholder
# ---------------------------------------------------------------------------

class ColorBlobDetector(BaseDetector):
    """HSV colour-segmentation detector (Phase B transition).

    Not implemented — placeholder for the colour-based detector that bridges
    OracleDetector (Phase A) and CameraDetector/YOLOv8 (Phase C).
    """

    def detect(self, *args, **kwargs) -> list[Detection]:
        raise NotImplementedError(
            "ColorBlobDetector is not yet implemented. "
            "Use OracleDetector for Phase A or CameraDetector for Phase C."
        )


# ---------------------------------------------------------------------------
# CameraDetector — Phase C placeholder
# ---------------------------------------------------------------------------

class CameraDetector(BaseDetector):
    """YOLOv8-nano camera detector (Phase C).

    Not implemented — placeholder for the real-deployment detector that runs
    YOLOv8-nano inference on RGB images and converts 2D bounding boxes to
    3D world positions via monocular depth estimation.

    Args:
        model_path: Path to the YOLOv8 .pt or .onnx weights file.
        conf_threshold: Minimum confidence to report a detection.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.5) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold

    def detect(self, *args, **kwargs) -> list[Detection]:
        raise NotImplementedError(
            "CameraDetector (YOLOv8) is not yet implemented. "
            "Use OracleDetector for Phase A training."
        )
