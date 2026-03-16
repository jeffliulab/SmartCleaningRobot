# 05 — 部署与测试

## 部署流程概览

```
Isaac Lab 训练                    ROS 2 / Gazebo 部署
┌─────────────────┐              ┌─────────────────────┐
│  SAC 训练        │              │  policy_node.py      │
│  best_agent.pt  │──export──→   │  (maze_model 模式)    │
│  (skrl 格式)     │              │                      │
└─────────────────┘              │  输入: /scan + /odom  │
                                 │  输出: /cmd_vel       │
                                 └─────────────────────┘
```

## 步骤 1：导出模型

将 skrl 训练的 checkpoint 转为 TorchScript 格式：

```bash
python deploy/scripts/export_model.py \
    --task maze \
    --checkpoint train/logs/skrl/maze_escape/<run_dir>/checkpoints/best_agent.pt \
    --output deploy/models/maze_agent_scripted.pt
```

参数说明：
- `--task maze`：指定迷宫任务（41 维输入，`[256, 128, 64]` 网络）
- `--checkpoint`：skrl 保存的最佳 agent checkpoint 路径
- `--output`：输出的 TorchScript 模型路径

验证输出：
```
Loaded weights from .../best_agent.pt (task=maze)
Exported TorchScript model -> deploy/models/maze_agent_scripted.pt
Verification: input (1, 41) -> output (1, 2)
```

## 步骤 2：Gazebo 仿真测试

### 生成 Gazebo 迷宫世界

```bash
python deploy/scripts/gen_maze_world.py
# 输出: deploy/cleaning_robot_ros/worlds/maze.world
```

该脚本使用与训练完全相同的 DFS 算法和种子 (seed=42)，确保仿真和训练的迷宫布局一致。

### 启动 Gazebo 仿真

```bash
# 终端 1: 启动 Gazebo
ros2 launch cleaning_robot_ros gazebo.launch.py world:=maze.world

# 终端 2: 启动策略节点 (SAC 训练模型)
ros2 run cleaning_robot_ros policy_node --ros-args \
    -p policy_type:=maze_model \
    -p model_path:=deploy/models/maze_agent_scripted.pt \
    -p exit_x:=0.0 \
    -p exit_y:=-3.15 \
    -p maze_diagonal:=5.7
```

### 对比：使用左手法则

```bash
ros2 run cleaning_robot_ros policy_node --ros-args \
    -p policy_type:=wall_follower
```

## 步骤 3：观测验证

### maze_model 模式的数据流

```
/scan (LaserScan)  ──→  MazeObsBuilder  ──→  41 维 obs tensor
/odom (Odometry)   ──→  (body frame)    ──→  TorchScript model
                                         ──→  (v, ω) action
                                         ──→  /cmd_vel (Twist)
```

### 41 维观测的构成

```
[0:2]   v_forward, omega          ← 来自 /odom twist
[2:38]  36 条 LiDAR 射线          ← 来自 /scan ranges (每 10 条取 1)
[38:40] sin(θ), cos(θ)            ← 出口方向（机体坐标系，由 odom pose 计算）
[40]    dist / 5.7                 ← 出口距离（归一化）
```

### 机体坐标系计算验证

在 ROS 2 中，`MazeObsBuilder` 使用与训练环境完全相同的坐标转换：

```python
# 世界坐标差
dx_w = exit_x - robot_x    # 来自 /odom
dy_w = exit_y - robot_y

# 旋转到机体坐标系
dx_b =  dx_w * cos(yaw) + dy_w * sin(yaw)
dy_b = -dx_w * sin(yaw) + dy_w * cos(yaw)

# 极坐标
angle = atan2(dy_b, dx_b)
exit_dir = (sin(angle), cos(angle))
```

## 策略对比

| 指标 | 左手法则 | SAC 训练模型 |
|------|----------|-------------|
| 需要全局地图 | 否 | 否 |
| 需要出口方向 | 否 | 是（odom 估算） |
| 适用迷宫类型 | 仅简单连通 | 任意结构 |
| 动作平滑度 | 低（急转） | 高（连续控制） |
| 路径效率 | 依赖迷宫结构 | 学习近似最优 |
| 对 odom 漂移敏感 | 否 | 轻度敏感 |

## 真实机器人部署注意事项

### 里程计对齐

1. 将机器人放在已知位置（推荐迷宫中心 (0, 0)）
2. 确保初始朝向一致（与训练中相同，如面向北/Y+方向）
3. 启动节点后不要手动移动机器人

### LiDAR 对齐

- 确保真实 LiDAR 的射线数量为 360（或对应调整 `lidar_num_rays` 参数）
- 确保 `lidar_max_distance` 与真实传感器量程一致
- 射线 0 号方向必须是机器人正前方

### 参数调整

如果真实表现不佳，可以尝试：

```bash
# 降低速度（更保守的控制）
ros2 run cleaning_robot_ros policy_node --ros-args \
    -p policy_type:=maze_model \
    -p model_path:=deploy/models/maze_agent_scripted.pt \
    -p max_linear_vel:=0.2 \
    -p max_angular_vel:=1.5
```

### 里程计漂移补偿

120 秒内的漂移通常在 10cm 左右，对于 0.45m 宽的通道足够使用。如果发现漂移严重，可以：

1. 使用 EKF 融合 IMU + 轮式里程计（ROS `robot_localization` 包）
2. 减小 `episode_length_s`，缩短单次运行时间
3. 考虑添加视觉里程计辅助（未来扩展）
