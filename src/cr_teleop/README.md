# cr_teleop — S1 学员包：基础运动控制与数据

对应教程：**Cleaning Robot S1**（E2 解剖扫地机器人 / E3 走一个方块 / E4 录数据与回放）。
三个作业共用这一个包，各自独立；参考答案都在 `solutions` 分支，main 只放 starter。

## E3 · square_drive（走一个方块）

补全 `square_drive.py` 的 `on_odom`：订阅 `/odom`、发布 `/cmd_vel`，
让机器人走出 **1m × 1m 方块**并停车（forward 1m → 原地转 90° → ×4）。

```bash
ros2 launch cr_bringup sim.launch.py        # 终端 1
ros2 run cr_teleop square_drive             # 终端 2
```

验收：日志 4 次「边到位」+「方块完成」，回原点误差 < 0.3m（odom 闭环漂移的正常量级）。

## E2 · scan_inspector（读懂 LiDAR）

补全 `scan_inspector.py`：订阅 `/scan`，每秒报告**最近障碍物的方向与距离**
（注意过滤 inf/nan 与盲区，角度用 angle_min + i × angle_increment 算）。

测试：仿真里给正前方 1.5m 放个 0.5m 箱子（教程 E2 附 spawn 命令），
应报出 ≈1.25m、≈0°。

## E4 · bag_odom_report（回放 rosbag）

先把 E3 跑一遍并录下数据，再补全 `bag_odom_report.py`：
从 bag 的 `/odom` 算出**总里程**和**最高速度**。

```bash
ros2 bag record /odom -o square_run         # 终端 2（录完 Ctrl-C 让它干净收尾）
ros2 run cr_teleop square_drive             # 终端 3
ros2 run cr_teleop bag_odom_report -- square_run   # 分析（Jazzy 默认 mcap 格式）
```

预期：总里程 ≈ 4.0m（4 × 1m）、最高速度 ≈ 0.2m/s。
bag 目录损坏时也可以把 URI 直接指到 `.mcap` 文件。

⛔ 调参请改文件顶部的具名常量，不要在代码里写死数字（工作区红线）。
