"""TurtleBot4 + WidowX 250 robot configuration for Isaac Lab.

Target hardware
---------------
Base:    TurtleBot4 (iRobot Create3, Ø 340 mm, 9 kg payload)
Arm:     Interbotix WidowX 250 (5-DOF, 650 mm reach, 250 g payload)
Compute: NVIDIA Orin NX 16 GB

Simulation proxy
----------------
Uses the NVIDIA LoCoBot USD from Isaac Nucleus as the simulation model.
LoCoBot has the same WidowX 200/250 5-DOF kinematic structure, so RL
policies trained on this proxy transfer to the target hardware with
minimal adaptation (link lengths differ but joint topology is identical).

LoCoBot USD path::

    {ISAAC_NUCLEUS_DIR}/Robots/LoCoBot/locobot.usd

Joint names (Interbotix WidowX standard)
------------------------------------------
Arm (5-DOF):
    waist  shoulder  elbow  forearm_roll  wrist_angle

Gripper (2 mirrored fingers):
    left_finger_link  right_finger_link

Drive wheels (Create2 / Create3 base):
    wheel_left_joint  wheel_right_joint

Kinematic constants
--------------------
MAX_REACH        = 0.650 m  (WidowX 250 fully extended)
BASE_HEIGHT      = 0.150 m  (approximate arm mount height above ground)
GRIPPER_MAX_OPEN = 0.037 m  (one-side travel)
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# ---------------------------------------------------------------------------
# Kinematic constants (exposed for use in env calculations)
# ---------------------------------------------------------------------------
MAX_REACH: float = 0.650          # m — WidowX 250 fully-extended reach
BASE_HEIGHT: float = 0.150        # m — arm-mount height above ground
GRIPPER_MAX_OPEN: float = 0.037   # m — max one-side gripper travel

# Wheel geometry (matches Create3 differential drive)
WHEEL_RADIUS: float = 0.036       # m
WHEEL_SEPARATION: float = 0.235   # m (Create3 track width)

# ---------------------------------------------------------------------------
# ArticulationCfg
# ---------------------------------------------------------------------------
TURTLEBOT4_WITH_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/LoCoBot/locobot.usd",
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
        joint_pos={
            # Arm starts in "ready" pose — forearm raised, arm bent
            "waist":        0.0,
            "shoulder":    -0.50,  # ~-28°
            "elbow":        0.80,  # ~+46°
            "forearm_roll": 0.0,
            "wrist_angle": -0.30,  # ~-17°
            # Gripper open
            "left_finger_link":  0.037,
            "right_finger_link": 0.037,
        },
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
            velocity_limit=20.0,
            effort_limit=2.0,
            stiffness=0.0,
            damping=0.01,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle"],
            velocity_limit=3.14,   # 180°/s
            effort_limit=5.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_finger_link", "right_finger_link"],
            velocity_limit=0.2,
            effort_limit=3.0,
            stiffness=200.0,
            damping=10.0,
        ),
    },
)


def diff_drive_forward(
    linear_vel: torch.Tensor, angular_vel: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert (linear, angular) to (left_wheel, right_wheel) angular velocities."""
    left = (linear_vel - angular_vel * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
    right = (linear_vel + angular_vel * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
    return left, right
