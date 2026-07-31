# cr_teleop — S1 学员包：基础运动控制

对应教程：**Cleaning Robot S1 · E3（第一个节点：走一个方块）**。

## 作业

补全 `cr_teleop/square_drive.py` 里的 `on_odom`，让机器人走出一个 **1m × 1m 的方块**并停车：

- 订阅 `/odom`（`nav_msgs/msg/Odometry`）拿位置与朝向，发布 `/cmd_vel`（`geometry_msgs/msg/Twist`）给速度指令；
- 状态机：`forward`（前进 1m）→ `turn`（原地转 90°）→ 重复 4 次 → 停；
- 四元数转偏航角、角度归一化两个工具函数已给出，任务参数（边长/速度/余量）在文件顶部，
  **调参请改常量，不要写死数字**。

## 运行

```bash
# 终端 1（容器内）：起仿真
ros2 launch cr_bringup sim.launch.py            # 或 headless:=true

# 终端 2（容器内）：跑你的节点
ros2 run cr_teleop square_drive
```

## 验收

- 日志出现 4 次「边到位」+「方块完成」；
- 终点回原点误差（节点日志会打印）在 odom 闭环漂移的正常量级（< 0.3m）。

⛔ 参考答案在 `solutions` 分支，先自己写。main 分支只放 starter。
