# Deployment: ROS 2 + Gazebo 验证与实机部署

本目录包含完整的 ROS 2 部署骨架，可直接在 Gazebo 中运行 JetBot 清扫机器人。

## 快速开始

### 前提条件

- ROS 2 Humble (或更新版本)
- Gazebo Classic 11 + `gazebo_ros_pkgs`
- Python 3.8+, PyTorch (仅 model 模式需要)

### 构建

```bash
# 在 ROS 2 workspace 中构建
mkdir -p ~/cleaning_ws/src
ln -s /path/to/SmartCleaningRobot/deploy/cleaning_robot_ros ~/cleaning_ws/src/
cd ~/cleaning_ws
colcon build --packages-select cleaning_robot_ros
source install/setup.bash
```

### 运行 Gazebo 仿真 (simple 策略)

```bash
# Hospital 场景 + 简单巡逻策略 (无需 RL 模型)
ros2 launch cleaning_robot_ros gazebo_sim.launch.py scene:=hospital policy_type:=simple

# Office 场景
ros2 launch cleaning_robot_ros gazebo_sim.launch.py scene:=office policy_type:=simple
```

### 运行 RL 模型

```bash
# 先导出训练好的模型
python deploy/scripts/export_model.py \
    --checkpoint train/logs/skrl/cleaning_coverage/.../checkpoints/best_agent.pt \
    --output deploy/models/best_agent_scripted.pt

# 再用 model 策略启动
ros2 launch cleaning_robot_ros gazebo_sim.launch.py policy_type:=model
```

### 查看覆盖地图

```bash
# 在另一个终端
ros2 topic echo /coverage_map --no-arr   # 查看元数据
# 或在 RViz2 中添加 OccupancyGrid 显示 /coverage_map
rviz2
```

## 整体架构

```
Level 1 (Isaac Lab)               Level 2 (ROS 2 + Gazebo)         Level 3 (实机)
┌────────────────┐                ┌────────────────────┐           ┌──────────────┐
│ 64 envs, GPU   │  best_agent.pt │ 1 env, 实时速度    │  同一代码 │ JetBot 实机   │
│ 无 ROS         │ ──export──────→│ ROS 2 topic 通信   │ ────────→ │ ROS 2 驱动    │
│ 百万步训练     │                │ Gazebo 模拟传感器  │           │ 真实传感器    │
└────────────────┘                └────────────────────┘           └──────────────┘
                                   ↑ 你现在在这里 (simple 策略已可运行)
```

## 目录结构

```
deploy/
├── README.md                              ← 本文件
├── cleaning_robot_ros/                    ← ROS 2 Python 包
│   ├── package.xml                        ← 包依赖声明
│   ├── setup.py / setup.cfg               ← 构建配置
│   ├── cleaning_robot_ros/                ← Python 源码
│   │   ├── policy_node.py                 ← 核心节点：传感器 → 策略 → /cmd_vel
│   │   ├── baselines/                     ← 基线策略
│   │   │   ├── simple_policy.py           ← bump-and-turn
│   │   │   └── wall_follower.py           ← 左手法则
│   │   ├── obs_builders/                  ← 观测构建器 (每个任务一个)
│   │   │   ├── coverage.py                ← 覆盖任务 obs
│   │   │   └── maze.py                    ← 迷宫任务 obs
│   │   └── coverage_tracker_cpu.py        ← 覆盖追踪 (CPU/numpy 版)
│   ├── launch/
│   │   ├── gazebo_sim.launch.py           ← Gazebo + JetBot + 策略节点
│   │   └── real_robot.launch.py           ← 实机驱动 + 策略节点
│   ├── config/
│   │   └── policy_params.yaml             ← 参数配置 (和训练配置一一对应)
│   ├── urdf/
│   │   └── jetbot.urdf.xacro             ← JetBot URDF (差速+LiDAR+Camera)
│   └── worlds/
│       ├── hospital.world                 ← Hospital 场景 (30m×30m)
│       ├── office.world                   ← Office 场景 (20m×20m)
│       └── maze.world                     ← 迷宫场景
├── models/                                ← 训练产出模型
│   └── .gitkeep
├── scripts/
│   ├── export_model.py                    ← skrl checkpoint → TorchScript
│   └── gen_maze_world.py                  ← 生成 Gazebo 迷宫 world
├── gazebo_worlds/                         ← (legacy, see cleaning_robot_ros/worlds/)
└── urdf/                                  ← (legacy, see cleaning_robot_ros/urdf/)
```

## 机器人配置 (与训练一致)

| 参数 | 值 | 对应训练文件 |
|------|------|-------------|
| 机器人 | JetBot (差速驱动) | `robots/jetbot.py` |
| 轮半径 | 0.0325m | `WHEEL_RADIUS` |
| 轮距 | 0.118m | `WHEEL_SEPARATION` |
| LiDAR | 360 rays, 3.5m | `JETBOT_LIDAR_CFG` |
| Camera | 640×480 RGB, 前置 | `JETBOT_CAMERA_CFG` |
| 最大线速度 | 0.3 m/s | `max_linear_vel` |
| 最大角速度 | 2.0 rad/s | `max_angular_vel` |

## 通信拓扑

```
                    ┌─────────────────────────┐
                    │     policy_node.py      │
                    │                         │
  /scan ──────────→ │  LiDAR → 36-dim vector  │
  (LaserScan)       │                         │
                    │  选择策略:               │
  /odom ──────────→ │  simple → bump-and-turn  │ ──→ /cmd_vel (Twist)
  (Odometry)        │  model  → RL inference  │
                    │                         │
                    │  覆盖追踪 → 覆盖地图     │ ──→ /coverage_map (OccupancyGrid)
                    └─────────────────────────┘
```

| Topic | 消息类型 | 方向 | 频率 | 说明 |
|-------|---------|------|------|------|
| `/scan` | LaserScan | 输入 | 10 Hz | LiDAR 360° 扫描 |
| `/odom` | Odometry | 输入 | 50 Hz | 里程计 (位姿+速度) |
| `/cmd_vel` | Twist | 输出 | 10 Hz | 线速度 + 角速度 |
| `/coverage_map` | OccupancyGrid | 输出 | 1 Hz | 覆盖地图 (RViz 可视化) |
| `/front_camera/image_raw` | Image | 输出 | 10 Hz | 前置相机 (预留) |

## 两种策略模式

### Simple 模式 (当前可用)

反应式 bump-and-turn 策略，仅使用 LiDAR：
- 前方无障碍 → 直行 (带微小弧度以增加覆盖)
- 前方有障碍 → 停下，向空旷侧转弯
- 转弯完成且前方清空 → 恢复直行

用途：**验证整个部署流水线是否跑通** (Gazebo → 传感器 → ROS → policy → 运动)

### Model 模式 (训练完成后)

加载 TorchScript 格式的 RL 模型：
1. 构建 1066 维 obs 向量 (和训练时完全一致)
2. 模型推理 → 归一化 action
3. 缩放到实际速度发布

## 模型导出

训练完成后，使用 `scripts/export_model.py` 将 skrl checkpoint 转为 TorchScript：

```bash
python deploy/scripts/export_model.py \
    --checkpoint train/logs/skrl/cleaning_coverage/YYYY-MM-DD_HH-MM-SS/checkpoints/best_agent.pt \
    --output deploy/models/best_agent_scripted.pt
```

## Sim-to-Real 注意事项

### 传感器差异

| 差异 | Gazebo | 真实硬件 | 缓解方法 |
|------|--------|---------|---------|
| LiDAR 噪声 | 高斯噪声 σ=0.005 | 真实噪声 ±2-5cm | 训练时加噪声 |
| 里程计漂移 | 理想 | 累积误差 | 短周期 + 重置覆盖图 |
| 地面摩擦 | 固定参数 | 真实地面各异 | Domain Randomization |
| 延迟 | ~0ms | ~50-100ms | 降低推理频率 |

### Level 2 → Level 3 验证清单

- [ ] Gazebo 中覆盖率 > 70%
- [ ] `/cmd_vel` 输出在安全范围内
- [ ] 碰撞频率可接受
- [ ] 推理延迟 < 50ms
- [ ] 覆盖图和里程计保持同步
- [ ] 紧急停止机制就绪
