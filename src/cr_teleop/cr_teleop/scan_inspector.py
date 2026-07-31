"""E2 作业：读懂 LiDAR——最近的障碍物在哪个方向、多远？

/scan（sensor_msgs/msg/LaserScan）是 360° 单线激光：
  ranges[i]   = 第 i 条射线的测距（米），可能是 inf（没打到东西）或 nan
  angle_min   = 第 0 条射线的角度（弧度，以机器人正前方为 0，逆时针为正）
  angle_increment = 相邻射线的角度间隔
  range_min / range_max = 有效量程
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

# 日志节流：scan 以 5Hz 进来，每 REPORT_EVERY 帧报一次（= 每秒 1 次）
REPORT_EVERY = 5


class ScanInspector(Node):
    def __init__(self):
        super().__init__("scan_inspector")
        self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
        self.frame = 0
        self.get_logger().info("盯着 /scan 看…")

    def on_scan(self, msg):
        # TODO(1) 遍历 msg.ranges，找出"最近的有效测距"：
        #   - 跳过非有限值（math.isfinite）和低于 range_min 的值；
        #   - 记录最小值及其下标 best_i。
        # TODO(2) 用 angle_min + best_i * angle_increment 算它的方向角，
        #   并用 math.degrees 转成度。
        # TODO(3) 一条有效测距都没有时（空旷场地）该怎么报？别让它崩。

        # —— 下面这段日志节流已写好，把你的结果填进 status 即可 ——
        self.frame += 1
        if self.frame % REPORT_EVERY != 0:
            return
        status = "（还没实现）"
        self.get_logger().info(status)


def main(args=None):
    rclpy.init(args=args)
    node = ScanInspector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
