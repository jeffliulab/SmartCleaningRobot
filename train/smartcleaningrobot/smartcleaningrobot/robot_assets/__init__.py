"""SmartCleaningRobot robot assets — hardware configurations (ArticulationCfg + sensors)."""

from .jetbot import (
    JETBOT_CAMERA_CFG,
    JETBOT_CFG,
    JETBOT_LIDAR_CFG,
    WHEEL_RADIUS,
    WHEEL_SEPARATION,
    diff_drive_forward,
)
from .turtlebot4_with_arm import (
    TURTLEBOT4_WITH_ARM_CFG,
    MAX_REACH,
    BASE_HEIGHT,
    GRIPPER_MAX_OPEN,
)
