"""E3 作业 starter：让扫地机器人走一个 1m × 1m 的方块并停准（odom 闭环）。

控制回路：订阅 /odom 拿位置与朝向，发布 /cmd_vel 给速度指令。
状态机：forward（前进 1m）→ turn（原地转 90°）→ 重复 4 次 → 停。

你的任务：补全下面 on_odom 里的三段 TODO。两个工具函数已给出，
任务参数都在文件顶部——调参改常量，不要在代码里写死数字。
"""

import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

# ---- 任务参数（都给了名字与单位，不是魔法数字）----
SIDE_LENGTH = 1.0      # 每边前进距离（米）
LINEAR_SPEED = 0.2     # 前进速度（米/秒），oomwoo 底盘不宜过快
ANGULAR_SPEED = 0.5    # 原地旋转速度（弧度/秒）
DIST_TOL = 0.02        # 距离判定余量（米）：欠一点再转弯，防过冲
ANGLE_TOL = 0.03       # 角度判定余量（弧度），约 1.7°
SIDES = 4              # 方块边数


def yaw_from_quaternion(x, y, z, w):
    """从四元数提取偏航角（绕 z 轴的朝向，弧度）。"""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    """把角度归一化到 (-pi, pi]，跨 ±180° 时差值才算得对。"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class SquareDrive(Node):
    def __init__(self):
        super().__init__("square_drive")
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.state = "forward"          # forward / turn / done
        self.sides_done = 0
        self.start_x = None
        self.start_y = None
        self.start_yaw = None
        self.get_logger().info("等待第一帧里程计…")

    def on_odom(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        # 第一帧只记录起点
        if self.start_x is None:
            self.start_x, self.start_y, self.start_yaw = x, y, yaw
            self.get_logger().info(f"起点 ({x:.3f}, {y:.3f})，出发")
            return

        cmd = Twist()

        # TODO(1) forward 状态：算从起点出发已经走了多远（math.hypot）。
        #   - 没走够 SIDE_LENGTH（记得留 DIST_TOL 余量）→ 给 cmd.linear.x 一个前进速度；
        #   - 走够了 → sides_done += 1，并判断：
        #       · 已完成 SIDES 条边 → 进 "done" 状态，打日志报终点与回原点误差；
        #       · 否则 → 切到 "turn" 状态，记录当前 yaw 作为转弯起点。

        # TODO(2) turn 状态：算从转弯起点到现在转过了多少角度
        #   （用 normalize_angle 处理 ±180° 跨界）。
        #   - 没转够 90°（pi/2，留 ANGLE_TOL 余量）→ 给 cmd.angular.z 一个旋转速度；
        #   - 转够了 → 切回 "forward"，记录当前 x, y 作为下一条边的起点。

        # TODO(3) 想清楚：为什么 "done" 状态下这个函数不需要写任何分支，
        #   机器人就会停住？（提示：cmd 是怎么来的、每一帧都会发生什么。）
        #   把答案讲给同伴听，或在注释里写一句。

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = SquareDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
