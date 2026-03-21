"""Generic 2D LiDAR sensor configuration for Isaac Lab.

This RayCasterCfg is robot-agnostic. Task env_cfg files should reference
``LIDAR_2D_CFG`` instead of importing from a specific robot asset module.

The prim_path uses a generic ``/World/envs/env_.*/Robot/chassis`` pattern —
each task env_cfg can override this via ``.replace(prim_path=...)`` if the
robot's chassis link has a different name.
"""

from __future__ import annotations

from isaaclab.sensors import RayCasterCfg
from isaaclab.sensors.ray_caster import patterns

LIDAR_2D_CFG = RayCasterCfg(
    prim_path="/World/envs/env_.*/Robot/chassis",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.05)),
    ray_alignment="yaw",
    pattern_cfg=patterns.LidarPatternCfg(
        channels=1,
        vertical_fov_range=(0.0, 0.0),
        horizontal_fov_range=(0.0, 360.0),
        horizontal_res=1.0,
    ),
    max_distance=3.5,
    mesh_prim_paths=["/World/Scene"],
)
