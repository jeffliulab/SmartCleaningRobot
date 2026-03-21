"""Small object detection and handling sub-project.

Phase A  — OracleDetector: reads object positions directly from simulation.
Phase B  — ColorBlobDetector: HSV colour segmentation (placeholder).
Phase C  — CameraDetector: YOLOv8-nano inference on RGB image (placeholder).

All detectors implement BaseDetector and return Detection instances so that
RL policy code is independent of the detection backend.
"""

from .detector import (
    BaseDetector,
    CameraDetector,
    ColorBlobDetector,
    Detection,
    OracleDetector,
)
from .object_assets import (
    OBJECT_HALF_HEIGHTS,
    OBJECT_RADII,
    PICKABLE_TYPES,
    SMALL_OBJECT_CFGS,
    make_object_cfg,
)

__all__ = [
    # Detector interface
    "BaseDetector",
    "Detection",
    "OracleDetector",
    "ColorBlobDetector",
    "CameraDetector",
    # Object asset helpers
    "SMALL_OBJECT_CFGS",
    "OBJECT_RADII",
    "OBJECT_HALF_HEIGHTS",
    "PICKABLE_TYPES",
    "make_object_cfg",
]
