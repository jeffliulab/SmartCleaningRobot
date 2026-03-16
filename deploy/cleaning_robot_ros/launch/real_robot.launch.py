"""Launch cleaning policy for real JetBot hardware.

Assumes the following driver nodes are already running
(or launched separately by the robot's bringup package):
    - LiDAR driver  → publishes /scan   (sensor_msgs/LaserScan)
    - Motor driver   → subscribes /cmd_vel (geometry_msgs/Twist)
    - Odometry node  → publishes /odom   (nav_msgs/Odometry)
    - Camera driver  → publishes /front_camera/image_raw (optional)

Usage:
    ros2 launch cleaning_robot_ros real_robot.launch.py
    ros2 launch cleaning_robot_ros real_robot.launch.py policy_type:=model
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import xacro


def generate_launch_description():
    pkg_share = get_package_share_directory("cleaning_robot_ros")
    urdf_path = os.path.join(pkg_share, "urdf", "jetbot.urdf.xacro")
    config_path = os.path.join(pkg_share, "config", "policy_params.yaml")

    robot_description = xacro.process_file(urdf_path).toxml()

    return LaunchDescription([
        DeclareLaunchArgument(
            "policy_type", default_value="simple",
            description="Policy type: simple (bump-and-turn) or model (RL)",
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
        ),

        Node(
            package="cleaning_robot_ros",
            executable="policy_node",
            parameters=[
                config_path,
                {"policy_type": LaunchConfiguration("policy_type")},
            ],
            output="screen",
        ),
    ])
