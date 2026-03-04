"""ROS 2 node that runs a cleaning policy on live sensor data.

Supports three policy modes:
  - "simple"        : reactive bump-and-turn (no model file required)
  - "wall_follower" : left-hand rule wall following (maze solving)
  - "model"         : trained RL policy loaded from TorchScript .pt file

Subscribes: /scan (LaserScan), /odom (Odometry)
Publishes:  /cmd_vel (Twist), /coverage_map (OccupancyGrid, optional)
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist
from std_msgs.msg import Header

from .obs_builder import ObsBuilder
from .coverage_tracker_cpu import CoverageTrackerCPU
from .simple_policy import SimplePolicy
from .wall_follower_policy import WallFollowerPolicy


class PolicyNode(Node):
    def __init__(self):
        super().__init__("cleaning_policy")

        # ---------- parameters ----------
        self.declare_parameter("policy_type", "simple")
        self.declare_parameter("model_path", "models/best_agent_scripted.pt")
        self.declare_parameter("max_linear_vel", 0.3)
        self.declare_parameter("max_angular_vel", 2.0)
        self.declare_parameter("lidar_num_rays", 360)
        self.declare_parameter("lidar_downsample_factor", 10)
        self.declare_parameter("lidar_max_distance", 3.5)
        self.declare_parameter("coverage_grid_resolution", 0.05)
        self.declare_parameter("coverage_local_view_size", 32)
        self.declare_parameter("robot_cleaning_radius", 0.06)
        self.declare_parameter("scene_bounds", [-15.0, -15.0, 15.0, 15.0])
        self.declare_parameter("inference_rate", 10.0)

        self._policy_type = self.get_parameter("policy_type").value
        self._max_v = self.get_parameter("max_linear_vel").value
        self._max_w = self.get_parameter("max_angular_vel").value
        bounds = self.get_parameter("scene_bounds").value

        # ---------- policy backend ----------
        self._model = None
        self._reactive_policy = None

        if self._policy_type == "model":
            import torch
            model_path = self.get_parameter("model_path").value
            self.get_logger().info(f"Loading RL model from: {model_path}")
            self._model = torch.jit.load(model_path, map_location="cpu")
            self._model.eval()
        elif self._policy_type == "wall_follower":
            self.get_logger().info("Using left-hand rule wall follower policy")
            self._reactive_policy = WallFollowerPolicy()
        else:
            self.get_logger().info("Using simple bump-and-turn policy")
            self._reactive_policy = SimplePolicy(
                forward_speed=self._max_v * 0.5,
                turn_speed=self._max_w * 0.6,
            )

        # ---------- obs builder ----------
        self._obs = ObsBuilder(
            lidar_num_rays=self.get_parameter("lidar_num_rays").value,
            lidar_downsample_factor=self.get_parameter("lidar_downsample_factor").value,
            lidar_max_distance=self.get_parameter("lidar_max_distance").value,
            coverage_local_size=self.get_parameter("coverage_local_view_size").value,
        )

        # ---------- coverage ----------
        self._coverage = CoverageTrackerCPU(
            x_min=bounds[0], y_min=bounds[1],
            x_max=bounds[2], y_max=bounds[3],
            resolution=self.get_parameter("coverage_grid_resolution").value,
            cleaning_radius=self.get_parameter("robot_cleaning_radius").value,
        )

        # ---------- ROS I/O ----------
        self.declare_parameter("lidar_topic", "/laser_scan/out")
        lidar_topic = self.get_parameter("lidar_topic").value
        self._sub_scan = self.create_subscription(LaserScan, lidar_topic, self._on_scan, 10)
        self._sub_odom = self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self._pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self._pub_map = self.create_publisher(OccupancyGrid, "/coverage_map", 1)

        rate = self.get_parameter("inference_rate").value
        self._timer = self.create_timer(1.0 / rate, self._step)
        self._map_timer = self.create_timer(1.0, self._publish_coverage_map)

        self._latest_x = 0.0
        self._latest_y = 0.0
        self._lidar_normalized: np.ndarray | None = None
        self._step_count = 0

        self.get_logger().info(
            f"PolicyNode ready — policy={self._policy_type}, "
            f"bounds={bounds}, rate={rate}Hz"
        )

    # ------------------------------------------------------------------ callbacks
    def _on_scan(self, msg: LaserScan) -> None:
        self._obs.update_lidar(msg.ranges)
        lidar_raw = np.array(msg.ranges, dtype=np.float32)
        lidar_max = self.get_parameter("lidar_max_distance").value
        factor = self.get_parameter("lidar_downsample_factor").value
        clipped = np.clip(lidar_raw, 0.0, lidar_max)
        self._lidar_normalized = clipped[::factor][:len(lidar_raw) // factor] / lidar_max

    def _on_odom(self, msg: Odometry) -> None:
        self._obs.update_pose(msg.pose.pose)
        self._obs.update_velocity(msg.twist.twist)
        self._latest_x = msg.pose.pose.position.x
        self._latest_y = msg.pose.pose.position.y

    # ------------------------------------------------------------------ control
    def _step(self) -> None:
        if not self._obs.ready():
            return

        self._coverage.update(self._latest_x, self._latest_y)
        local = self._coverage.get_local_view(self._latest_x, self._latest_y)
        ratio = self._coverage.get_coverage_ratio()
        self._obs.update_coverage(local, ratio)

        cmd = Twist()

        if self._policy_type == "model" and self._model is not None:
            import torch
            obs_tensor = self._obs.build()
            with torch.inference_mode():
                action = self._model(obs_tensor)
            cmd.linear.x = float(action[0, 0]) * self._max_v
            cmd.angular.z = float(action[0, 1]) * self._max_w
        elif self._reactive_policy is not None and self._lidar_normalized is not None:
            v, w = self._reactive_policy.compute_action(self._lidar_normalized)
            cmd.linear.x = v
            cmd.angular.z = w

        self._pub_cmd.publish(cmd)

        self._step_count += 1
        if self._step_count % 100 == 0:
            self.get_logger().info(
                f"step={self._step_count}  coverage={ratio:.1%}  "
                f"pos=({self._latest_x:.2f}, {self._latest_y:.2f})"
            )

    # ------------------------------------------------------------------ map viz
    def _publish_coverage_map(self) -> None:
        grid = self._coverage.grid
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.info.resolution = self._coverage.resolution
        msg.info.width = self._coverage.width
        msg.info.height = self._coverage.height
        msg.info.origin.position.x = self._coverage.x_min
        msg.info.origin.position.y = self._coverage.y_min
        msg.data = (grid.flatten() * 100).astype(np.int8).tolist()
        self._pub_map.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
