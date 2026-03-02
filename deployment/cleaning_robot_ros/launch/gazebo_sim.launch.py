"""Launch Gazebo simulation with JetBot and cleaning policy.

Usage:
    ros2 launch cleaning_robot_ros gazebo_sim.launch.py
    ros2 launch cleaning_robot_ros gazebo_sim.launch.py scene:=office policy_type:=model
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import xacro


def _setup(context: LaunchContext):
    pkg_share = get_package_share_directory("cleaning_robot_ros")

    scene = context.launch_configurations["scene"]
    policy_type = context.launch_configurations["policy_type"]

    world_path = os.path.join(pkg_share, "worlds", f"{scene}.world")
    urdf_path = os.path.join(pkg_share, "urdf", "jetbot.urdf.xacro")
    config_path = os.path.join(pkg_share, "config", "policy_params.yaml")

    robot_description = xacro.process_file(urdf_path).toxml()

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("gazebo_ros"),
                "launch",
                "gazebo.launch.py",
            )
        ),
        launch_arguments={"world": world_path}.items(),
    )

    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
    )

    spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "robot_description",
            "-entity", "jetbot",
            "-x", "0", "-y", "0", "-z", "0.05",
        ],
        output="screen",
    )

    policy = Node(
        package="cleaning_robot_ros",
        executable="policy_node",
        parameters=[config_path, {"policy_type": policy_type}],
        output="screen",
    )

    return [gazebo, robot_state_pub, spawn, policy]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "scene", default_value="hospital",
            description="Gazebo world to load (hospital / office)",
        ),
        DeclareLaunchArgument(
            "policy_type", default_value="simple",
            description="Policy type: simple (bump-and-turn) or model (RL)",
        ),
        OpaqueFunction(function=_setup),
    ])
