"""cr_bringup 仿真总入口：Gazebo + 机器人模型 + ros_gz 桥接。

用法：
  ros2 launch cr_bringup sim.launch.py              # 带 Gazebo 画面
  ros2 launch cr_bringup sim.launch.py headless:=true   # 无界面（CI / 无显示环境）

⚠️ 机器人 spawn 名固定为 oomwoo_one：cr_description/config/gz_bridge.yaml 里
bumper 的桥接 topic 绑死了这个名字（详见该文件注释）。
"""

import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# 出生点高度（米）：略高于地面，让仿真开局自然落稳
SPAWN_Z = 0.02


def generate_launch_description():
    desc_share = get_package_share_directory("cr_description")
    gazebo_share = get_package_share_directory("cr_gazebo")

    robot_xml = xacro.process_file(
        os.path.join(desc_share, "urdf", "robot.urdf.xacro")
    ).toxml()
    world = os.path.join(gazebo_share, "worlds", "empty.sdf")

    headless = LaunchConfiguration("headless")

    gz_gui = ExecuteProcess(
        cmd=["gz", "sim", "-r", world], condition=UnlessCondition(headless), output="screen"
    )
    gz_headless = ExecuteProcess(
        cmd=["gz", "sim", "-s", "-r", world], condition=IfCondition(headless), output="screen"
    )

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_xml, "use_sim_time": True}],
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-name", "oomwoo_one", "-topic", "robot_description", "-z", str(SPAWN_Z)],
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[
            {"config_file": os.path.join(desc_share, "config", "gz_bridge.yaml")}
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("headless", default_value="false"),
            gz_gui,
            gz_headless,
            rsp,
            spawn,
            bridge,
        ]
    )
