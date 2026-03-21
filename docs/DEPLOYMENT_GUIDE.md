# SmartCleaningRobot — ROS 2 + Gazebo 部署指南

> 本文档面向训练完成后的部署阶段。如果你还未完成训练，请先参阅 [GETTING_STARTED.md](GETTING_STARTED.md) 和 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)。
> 对应目录: `deploy/`
> 运行环境: Ubuntu 22.04 (WSL2), ROS 2 Humble

---

## 目录

1. [部署端概述](#1-部署端概述)
2. [系统安装](#2-系统安装)
3. [构建步骤](#3-构建步骤)
4. [使用方法](#4-使用方法)
5. [文件职责速查](#5-文件职责速查)
6. [ROS 2 话题一览](#6-ros-2-话题一览)
7. [参数对照表（训练 ↔ 部署）](#7-参数对照表训练--部署)
8. [通信拓扑图](#8-通信拓扑图)
9. [两种策略模式说明](#9-两种策略模式说明)
10. [场景说明](#10-场景说明)
11. [常见问题](#11-常见问题)
12. [训练端与部署端的关系](#12-训练端与部署端的关系)

---

## 1. 部署端概述

部署端运行在 **ROS 2 + Gazebo** 上，与训练端 (Isaac Lab) 的环境**完全独立**。训练在 Windows 原生的 Isaac Lab 里跑，部署在 WSL2 里跑，两边通过 `deploy/models/` 共享模型文件。

---

## 2. 系统安装

### 2.1 操作系统

| 选项 | 推荐度 | 说明 |
|------|--------|------|
| **Ubuntu 22.04 (WSL2)** | 强烈推荐 | ROS 2 Humble 官方支持，Gazebo 开箱即用 |
| Ubuntu 22.04 (原生 / 双系统) | 推荐 | 性能最好，GPU 直通无损 |
| Windows 原生 | 不推荐 | ROS 2 Windows 版缺少很多包，Gazebo 支持差 |

### 2.2 ROS 2 Humble

```bash
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install -y curl gnupg lsb-release

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop
```

安装完后在 `~/.bashrc` 中添加：
```bash
source /opt/ros/humble/setup.bash
```

### 2.3 Gazebo Classic 11 + ROS 2 桥接

```bash
sudo apt install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros \
    ros-humble-robot-state-publisher \
    ros-humble-xacro \
    ros-humble-tf2-ros
```

### 2.4 键盘遥控（可选，推荐）

```bash
sudo apt install -y ros-humble-teleop-twist-keyboard
```

### 2.5 Python 依赖

```bash
pip install numpy torch    # torch 仅在 policy_type=model 时需要
```

> simple 策略模式只需要 `numpy`，不需要 GPU。

### 2.6 colcon 构建工具

```bash
sudo apt install -y python3-colcon-common-extensions
```

### 2.7 环境检查一键脚本

```bash
echo "=== ROS 2 ===" && ros2 --help 2>&1 | head -1
echo "=== ROS_DISTRO ===" && echo $ROS_DISTRO
echo "=== Gazebo ===" && gazebo --version 2>&1 | head -1
echo "=== colcon ===" && colcon version-check 2>&1 | head -1
echo "=== Key packages ==="
for pkg in gazebo-ros robot-state-publisher xacro tf2-ros; do
  dpkg -s ros-humble-$pkg 2>/dev/null | grep "Status:" || echo "MISSING: ros-humble-$pkg"
done
echo "=== Python ==="
python3 -c "import rclpy; print('rclpy OK')" 2>&1
python3 -c "import numpy; print('numpy OK')" 2>&1
```

### 2.8 安装总结

| 软件 | 用途 | 是否必须 |
|------|------|---------|
| ROS 2 Humble | 节点通信框架 | 必须 |
| Gazebo 11 | 物理仿真 | 必须 |
| gazebo_ros_pkgs | Gazebo <-> ROS 2 桥接 | 必须 |
| robot_state_publisher | 发布机器人 TF | 必须 |
| xacro | 解析 URDF 模板 | 必须 |
| teleop_twist_keyboard | 键盘遥控 | 推荐 |
| numpy | 覆盖追踪、obs 构建 | 必须 |
| PyTorch | 加载 RL 模型推理 | 仅 model 模式 |
| colcon | 构建 ROS 2 包 | 必须 |

---

## 3. 构建步骤

```bash
# 1. 创建 ROS 2 workspace
mkdir -p ~/cleaning_ws/src

# 2. 软链接包目录（代码修改实时同步）
ln -s /mnt/d/Projects/SmartCleaningRobot/deploy/cleaning_robot_ros \
      ~/cleaning_ws/src/cleaning_robot_ros

# 3. 构建
cd ~/cleaning_ws
colcon build --packages-select cleaning_robot_ros --symlink-install

# 4. 加载环境（每次新开终端都要执行，或加入 ~/.bashrc）
source ~/cleaning_ws/install/setup.bash
```

> `--symlink-install` 让 Python 文件修改后不用重新 build，直接生效。
> 如果在 `/mnt/d/` 下编译很慢，可将整个 `deployment` 拷贝到 WSL2 home 目录下。

---

## 4. 使用方法

### 4.1 Gazebo 仿真 — Simple 策略（立即可用，无需训练）

```bash
# Hospital 场景
ros2 launch cleaning_robot_ros gazebo_sim.launch.py scene:=hospital policy_type:=simple

# Office 场景
ros2 launch cleaning_robot_ros gazebo_sim.launch.py scene:=office policy_type:=simple
```

启动后：
- Gazebo 窗口弹出，显示场景和 JetBot 机器人
- 机器人自动开始巡逻（前进、碰壁转弯）
- 终端每 100 步打印一次覆盖率和位置

> **注意**：Gazebo 的场景是用 SDF 格式手写的简化版，和 Isaac Sim 的高精度 USD
> 场景在视觉细节上不同，但物理布局概念性一致。

### 4.2 键盘遥控机器人

```bash
# 新开终端
source /opt/ros/humble/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

| 按键 | 功能 |
|------|------|
| `i` | 前进 |
| `,` | 后退 |
| `j` | 左转 |
| `l` | 右转 |
| `k` | 停止 |
| `q` / `z` | 增/减速度 |

### 4.3 RViz2 可视化（LiDAR + 覆盖地图 + 摄像头）

```bash
# 新开终端
source /opt/ros/humble/setup.bash
source ~/cleaning_ws/install/setup.bash
rviz2
```

**必须先做**：Global Options -> **Fixed Frame** 改为 **`odom`**（默认的 `map` 不可用）

添加显示项（推荐用 **By topic** 方式添加）：

| 点击 Add -> By topic | 选择类型 | 显示效果 |
|------|------|------|
| `/laser_scan/out` | **LaserScan** | 3D 视图中的红色激光线 |
| `/coverage_map` | **Map** | 覆盖进度热力图 |
| `/front_camera/image_raw` | **Image** | 左下角摄像头画面 |

**视角跟随机器人**：
1. 右侧 Views 面板 -> Type 选 **ThirdPersonFollower**
2. Target Frame 设为 **`base_link`**
3. Distance 设为 `2`

> LiDAR 的实际 topic 是 `/laser_scan/out`（Gazebo 插件默认命名），不是 `/scan`。

### 4.4 查看话题数据

```bash
# 查看所有活跃话题
ros2 topic list

# 查看 LiDAR 数据（单次）
ros2 topic echo /laser_scan/out --once

# 查看里程计
ros2 topic echo /odom --once

# 查看机器人发出的速度指令
ros2 topic echo /cmd_vel

# 查看覆盖地图元数据（不显示大数组）
ros2 topic echo /coverage_map --no-arr

# 查看各话题发布频率
ros2 topic hz /laser_scan/out
ros2 topic hz /front_camera/image_raw
```

### 4.5 Gazebo 仿真 — RL 模型（训练完成后）

```bash
# Step 1: 导出模型（在 Windows/Isaac Lab 端执行）
conda activate isaaclab
python deploy/scripts/export_model.py \
    --checkpoint train/logs/skrl/cleaning_coverage/YYYY-MM-DD_HH-MM-SS/checkpoints/best_agent.pt \
    --output deploy/models/best_agent_scripted.pt

# Step 2: 启动（在 WSL2/ROS 2 端执行）
ros2 launch cleaning_robot_ros gazebo_sim.launch.py policy_type:=model
```

### 4.6 实机部署（Level 3，未来）

```bash
# 假设硬件驱动已经由 JetBot 的 bringup 包启动
ros2 launch cleaning_robot_ros real_robot.launch.py policy_type:=model
```

> `real_robot.launch.py` 不启动 Gazebo，只启动 `robot_state_publisher` 和 `policy_node`。

---

## 5. 文件职责速查

| 文件 | 做什么 |
|------|--------|
| `policy_node.py` | 核心节点：订阅传感器，计算动作，发布 `/cmd_vel` + `/coverage_map` |
| `simple_policy.py` | 简单 bump-and-turn 策略，仅用 LiDAR 36 维向量做决策 |
| `obs_builder.py` | 把 ROS 消息转换成训练时的 obs 张量 |
| `coverage_tracker_cpu.py` | 覆盖追踪的 CPU/numpy 版本（训练端用 GPU/torch 版本） |
| `jetbot.urdf.xacro` | JetBot 的 Gazebo URDF（差速 + LiDAR + Camera + 物理属性） |
| `hospital.world` | 30m x 30m 医院场景（SDF 格式） |
| `office.world` | 20m x 20m 办公场景（SDF 格式） |
| `gazebo_sim.launch.py` | 一键启动 Gazebo + 机器人 + 策略节点 |
| `real_robot.launch.py` | 一键启动策略节点（不含 Gazebo，用于实机） |
| `policy_params.yaml` | 所有运行参数（和训练端 `env_cfg.py` 一一对应） |
| `export_model.py` | 训练完成后把 skrl checkpoint 转为 TorchScript `.pt` 文件 |

---

## 6. ROS 2 话题一览

| 话题 | 类型 | 方向 | 来源 |
|------|------|------|------|
| `/cmd_vel` | `geometry_msgs/Twist` | policy -> Gazebo | 策略输出的速度指令 |
| `/odom` | `nav_msgs/Odometry` | Gazebo -> policy | 差速驱动插件发布的里程计 |
| `/laser_scan/out` | `sensor_msgs/LaserScan` | Gazebo -> policy | LiDAR 360线扫描 |
| `/front_camera/image_raw` | `sensor_msgs/Image` | Gazebo -> (预留) | 前置摄像头 RGB |
| `/coverage_map` | `nav_msgs/OccupancyGrid` | policy -> RViz | 清扫覆盖进度地图 |
| `/tf` | `tf2_msgs/TFMessage` | 多个节点 | 坐标变换 |
| `/robot_description` | `std_msgs/String` | RSP | URDF 描述 |

---

## 7. 参数对照表（训练 <-> 部署）

> **关键原则：训练端改了任何参数，部署端的 `policy_params.yaml` + URDF 也必须同步修改！**

| 参数 | 训练端 (Isaac Lab) | 部署端 (ROS 2) | 不一致会怎样 |
|------|-------------------|----------------|-------------|
| 轮半径 | 0.0325m | URDF 0.0325m | 速度缩放错误 |
| 轮距 | 0.118m | URDF 0.118m | 转弯半径不对 |
| LiDAR 射线数 | 360 | URDF 360 | obs 维度不匹配 |
| LiDAR 最大距离 | 3.5m | URDF 3.5m + yaml 3.5 | 归一化错误 |
| 降采样因子 | 5 (360->72) | yaml 5 | obs 维度不匹配 |
| 覆盖图分辨率 | 0.05m | yaml 0.05 | 覆盖图不对齐 |
| 局部视图大小 | 32x32 | yaml 32 | obs 维度不匹配 |
| 清扫半径 | 0.06m | yaml 0.06 | 覆盖计算不同 |
| 场景边界 | 取决于场景 | yaml scene_bounds | 覆盖图越界 |
| max_linear_vel | 0.3 m/s | yaml 0.3 | 动作缩放错误 |
| max_angular_vel | 2.0 rad/s | yaml 2.0 | 动作缩放错误 |

> 切换场景时需同步修改 `scene_bounds`（Hospital: [-15,-15,15,15]，Office: [-10,-10,10,10]）。

---

## 8. 通信拓扑图

```
Gazebo (或真实硬件)                    policy_node.py
┌──────────────────┐                 ┌─────────────────────┐
│ diff_drive 插件   │ ── /odom ────→ │ obs_builder          │
│ LiDAR 插件       │ ── /scan ────→ │   ↓                  │
│ Camera 插件      │ ── /image ──→  │ (预留, 暂不使用)      │
│                  │                │                      │
│                  │ ← /cmd_vel ── │ simple_policy          │
│                  │                │   或 RL model         │
└──────────────────┘                │                      │
                                    │ coverage_tracker_cpu  │
                                    │   ↓                  │
                                    │ /coverage_map ──→ RViz│
                                    └─────────────────────┘
```

---

## 9. 两种策略模式说明

### Simple 模式 (`policy_type:=simple`)

- **不需要任何训练模型**，立即可用
- 用途：验证部署流水线是否跑通
- 逻辑：
  1. 用 LiDAR 前方 7 根射线检测障碍
  2. 前方空旷 -> 直行 (0.15 m/s)，带微小弧度增加覆盖
  3. 前方有障碍 (距离 < 0.25 归一化阈值) -> 停下，转向空旷侧
  4. 转够 15 步且前方清空 -> 恢复直行

### Model 模式 (`policy_type:=model`)

- **需要先完成 Isaac Lab 训练 + 导出模型**
- 流程：
  1. `obs_builder` 组装 obs 张量
  2. TorchScript 模型推理 -> 输出 `(v, omega)` 归一化值
  3. 乘以 `max_linear_vel` / `max_angular_vel` -> 发布 `/cmd_vel`

---

## 10. 场景说明

### Hospital (hospital.world)

- 尺寸：30m x 30m（对应训练端 `scene_bounds: (-15,-15,15,15)`）
- 布局：中央走廊 + 南北两翼各 3 间病房 + 病床/护士站/医疗推车
- 复杂度：中高（走廊狭窄、多房间、多障碍物）

### Office (office.world)

- 尺寸：20m x 20m（对应训练端 `scene_bounds: (-10,-10,10,10)`）
- 布局：会议室 + 茶水间 + 1.2m 高隔断区 + 办公桌/打印机/文件柜
- 复杂度：中（开放区域多、低矮隔断需绕行）

---

## 11. 常见问题

### Q1: Gazebo 启动后黑屏

```bash
# 检查 Gazebo 是否正常安装
gazebo --version    # 应输出 11.x.x

# WSL2 GUI 测试
sudo apt install -y mesa-utils
glxinfo | grep "OpenGL version"
```

> Windows 11 自带 WSLg，一般直接可用。Windows 10 需要额外装 X Server（VcXsrv）。

### Q2: 找不到包 "cleaning_robot_ros"

```bash
cd ~/cleaning_ws
colcon build --packages-select cleaning_robot_ros
source install/setup.bash
ros2 pkg list | grep cleaning    # 应输出: cleaning_robot_ros
```

### Q3: RViz2 中 LaserScan 不显示

1. **Fixed Frame** 必须是 `odom`（不是 `map`）
2. LaserScan Topic 应为 `/laser_scan/out`
3. 如果通过 By topic 添加会自动填好

### Q4: Coverage Map 显示 "No map received"

- 确认 Fixed Frame 已改为 `odom`
- 确认 `policy_node` 在运行：终端应有 `PolicyNode ready` 日志
- `ros2 topic echo /coverage_map --once` 确认有数据

### Q5: 机器人生成后直接掉下去 / 穿过地面

- 检查 spawn 时的 z 坐标是否 >= 0.05

### Q6: model 模式报错 "No such file"

- 先完成训练 + `export_model.py` 导出
- 检查 `policy_params.yaml` 中的 `model_path` 路径

### Q7: 键盘遥控不响应

- 确认 `teleop_twist_keyboard` 终端窗口是否有焦点
- 确认该终端已 source 了 ROS 2 环境

---

## 12. 训练端与部署端的关系

```
                     SmartCleaningRobot/
                     ├── train/smartcleaningrobot/         ← 训练端 (Isaac Lab, Windows)
                     │   ├── robot_assets/jetbot.py
                     │   ├── scenes/
                     │   ├── tasks/.../env_cfg.py
                     │   └── utils/
                     │
                     ├── deploy/                           ← 部署端 (ROS 2, WSL2)
                     │   ├── cleaning_robot_ros/
                     │   ├── models/
                     │   └── scripts/export_model.py
                     │
训练 ──best_agent.pt──→ export_model.py ──→ best_agent_scripted.pt ──→ 部署
```

- 训练端改参数 -> 部署端必须同步改 `policy_params.yaml` + URDF
- 两端共享 `deploy/models/` 目录
- 部署端**不依赖** Isaac Lab 或 Isaac Sim，可以在没有 GPU 的机器上跑 (simple 模式)
