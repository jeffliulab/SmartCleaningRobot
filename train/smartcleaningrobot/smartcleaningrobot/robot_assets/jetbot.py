"""JetBot robot configuration with LiDAR and Camera sensors for Isaac Lab.

JetBot is a differential drive robot with two wheels and built-in cameras.
Physical specs:
    - Wheel radius: 0.0325m
    - Wheel separation: 0.118m
    - 2 built-in cameras in the USD model
    - LiDAR added via RayCaster sensor

Note: Joint names and prim paths assume the official NVIDIA JetBot USD from
Isaac Sim Nucleus. If using a different JetBot model, verify the joint/link
names match the USD hierarchy.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import CameraCfg, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

WHEEL_RADIUS = 0.0325
WHEEL_SEPARATION = 0.118

JETBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            velocity_limit=20.0,
            effort_limit=0.5,
            stiffness=0.0,
            damping=1.0e-2,
        ),
    },
)

JETBOT_LIDAR_CFG = RayCasterCfg(
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

JETBOT_CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/chassis/front_cam",
    update_period=0.1,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 1.0e5),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.05, 0.0, 0.05),
        rot=(0.5, -0.5, 0.5, -0.5),
        convention="ros",
    ),
)


def diff_drive_forward(
    linear_vel: torch.Tensor, angular_vel: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert (linear, angular) velocity to (left_wheel, right_wheel) angular velocities."""
    left = (linear_vel - angular_vel * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
    right = (linear_vel + angular_vel * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
    return left, right
