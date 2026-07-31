# T1: MazeEscape — 完整指南

> 本文档是 T1 MazeEscape 任务的自包含指南，假设你已完成 [GETTING_STARTED.md](../../GETTING_STARTED.md) 的环境搭建。

**Gym ID:** `SmartCleaningRobot-MazeEscape-v0`
**机器人:** JetBot（差分驱动底盘，无机械臂）
**场景:** DFS 生成的 8×8 迷宫
**推荐算法:** SAC（也支持 PPO）

---

## 目录

1. [任务描述](#1-任务描述)
2. [背景知识](#2-背景知识)
3. [环境详解](#3-环境详解)
4. [奖励函数详解](#4-奖励函数详解)
5. [训练配置](#5-训练配置)
6. [训练步骤](#6-训练步骤)
7. [结果分析](#7-结果分析)
8. [常见问题](#8-常见问题)

---

## 1. 任务描述

### 1.1 目标

训练一个 JetBot 机器人在 8×8 的随机迷宫中找到出口。机器人从迷宫中心出发，需要利用 LiDAR 传感器感知墙壁，规划路径到达迷宫南侧出口。

### 1.2 为什么从迷宫开始？

迷宫逃脱是学习 RL 的理想起点：
- **动作空间小**：只有 2 维（前进/转弯），易于训练
- **奖励信号清晰**：越接近出口奖励越高，到达出口有大额奖励
- **可视化直观**：能直观看到机器人的导航行为
- **训练时间短**：3M 步约 4-8 小时可收敛

### 1.3 迷宫规格

| 参数 | 值 | 说明 |
|------|-----|------|
| 网格大小 | 8×8 = 64 个格子 | 逻辑坐标 |
| 格子尺寸 | 0.45 m | 物理尺寸 |
| 墙壁高度 | 0.3 m | 足以阻挡 LiDAR |
| 迷宫种子 | 42 | 确定性迷宫，可复现 |
| 起始位置 | 中心格 (4,4) = 世界坐标 (0, 0) | 每次可随机化 |
| 出口位置 | 南墙中心 = 世界坐标 (0.0, -3.15) | 固定 |
| 出口判定距离 | 0.5 m | 机器人距出口 < 0.5m 即成功 |
| 迷宫对角线 | ~5.7 m | 用于距离归一化 |

迷宫使用 **DFS（深度优先搜索）** 算法生成，确保：
- 迷宫是 **简单连通** 的（任意两点之间恰好一条路径）
- 没有环路，但有死胡同

---

## 2. 背景知识

### 2.1 差分驱动（Differential Drive）

JetBot 使用差分驱动，即通过左右两个轮子的速度差来控制前进和转弯：

```
        ┌───────────┐
        │   JetBot   │
   v_L  │  ○     ○  │  v_R
   ◄────┤  左    右  ├────►
        │           │
        └───────────┘
         wheel_dist = 0.118m

线速度 v = (v_R + v_L) / 2
角速度 ω = (v_R - v_L) / wheel_dist

反推:
v_L = (v - ω × wheel_dist/2) / wheel_radius
v_R = (v + ω × wheel_dist/2) / wheel_radius

wheel_radius = 0.0325m
```

**直觉理解：**
- 两轮速度相同 → 直走
- 右轮快 → 左转
- 左轮快 → 右转
- 两轮反向 → 原地旋转

### 2.2 LiDAR（激光雷达）

LiDAR 通过发射激光并测量反射时间来获取周围物体的距离：

```
                    0° (前方)
                     │
            315°     │     45°
               ╲     │     ╱
                ╲    │    ╱
         270° ───── JetBot ───── 90°
                ╱    │    ╲
               ╱     │     ╲
            225°     │     135°
                     │
                   180° (后方)
```

- **发射 360 条射线**（每 1° 一条，覆盖 360°）
- 每条射线返回到最近障碍物的 **距离**（最远 3.5m）
- 下采样到 36 条射线（每 10° 取一条），输入到策略网络
- 距离归一化到 [0, 1]：距离 / 3.5m

### 2.3 BFS 距离场

**BFS（广度优先搜索）** 是一种图搜索算法，用于计算迷宫中每个格子到出口的最短距离。

```
┌───┬───┬───┬───┐
│ 6 │ 5 │ 4 │ 3 │
├───┼   ┼───┼   ┤
│ 7 │ 6 │ 5 │ 2 │
├   ┼───┼   ┼───┤
│ 8 │ 7 │ 6 │ 1 │
├───┼   ┼───┼   ┤
│ 9 │ 8 │ 7 │ 0 │  ← 出口（距离=0）
└───┴───┴───┴───┘
```

**为什么用 BFS 而不是直线距离？**
- 直线距离（欧氏距离）无法感知墙壁，可能指向墙后面
- BFS 距离沿着实际路径计算，能正确引导机器人绕过死胡同

**用途：** 作为奖励信号——机器人每走一步，如果 BFS 距离减小，给予正奖励。

### 2.4 Body Frame（机体坐标系）

所有观测都在 **机体坐标系** 下表示，而非世界坐标系：

```
世界坐标系:                  机体坐标系:
    Y                            前方 (X_body)
    ▲                              ▲
    │                              │
    │                              │
    └────► X                左 ◄────┘────► 右
                                         (Y_body)
```

**优势：** 无论机器人朝哪个方向，"前方有墙"总是相同的观测值。这使得策略更容易泛化。

**出口方向转换：**
```python
# 世界坐标下的出口偏移
dx_world = exit_x - robot_x
dy_world = exit_y - robot_y

# 转换到机体坐标
dx_body = dx_world * cos(yaw) + dy_world * sin(yaw)
dy_body = -dx_world * sin(yaw) + dy_world * cos(yaw)

# 用角度表示方向（sin/cos 编码避免 -π/+π 不连续）
angle = atan2(dy_body, dx_body)
exit_dir = [sin(angle), cos(angle)]
```

### 2.5 LiDAR 传感器配置（Isaac Lab）

LiDAR 使用 Isaac Lab 的 `RayCaster` 传感器模拟：

```python
JETBOT_LIDAR_CFG = RayCasterCfg(
    prim_path="/World/envs/env_.*/Robot/chassis",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.05)),
    pattern_cfg=patterns.LidarPatternCfg(
        channels=1,
        vertical_fov_range=(0.0, 0.0),      # 2D 平面扫描
        horizontal_fov_range=(0.0, 360.0),   # 360° 全向
        horizontal_res=1.0,                   # 每度 1 条射线 → 共 360 条
    ),
    max_distance=3.5,                         # 最大检测距离 3.5m
)
```

36 条下采样射线在机器人周围均匀分布：

```
          前方 (idx=0, 0°)
            ↑
    idx=5 / | \ idx=35
   (50°) /  |  \ (350°)
        /   |   \
idx=9 ←  机器人  → idx=27
(90°)    (左)    (右, 270°)
        \   |   /
         \  |  /
    idx=13\ | /idx=23
            ↓
      后方 (idx=18, 180°)
```

### 2.6 里程计（Odometry）

**仿真中**直接从物理引擎获取精确值：

```python
root_pos = robot.data.root_pos_w        # 世界位置 (x, y, z)
root_quat = robot.data.root_quat_w      # 四元数朝向
lin_vel = robot.data.root_com_lin_vel_b  # 机体坐标系线速度
ang_vel = robot.data.root_com_ang_vel_b  # 机体坐标系角速度
```

**真实机器人**通过轮式编码器计算：

```
左轮编码器 → v_left  = 左轮角速度 × 轮径 (0.0325m)
右轮编码器 → v_right = 右轮角速度 × 轮径 (0.0325m)

线速度: v = (v_left + v_right) / 2
角速度: ω = (v_right - v_left) / 轮距 (0.118m)

位置积分:
  yaw += ω × dt
  x   += v × cos(yaw) × dt
  y   += v × sin(yaw) × dt
```

里程计漂移参考（MPU6050 IMU）：

| 运行时间 | 角度漂移 | 位置漂移 | 对迷宫的影响 |
|----------|---------|---------|-------------|
| 10 秒 | ~0.5° | ~1 cm | 可忽略 |
| 60 秒 | ~3° | ~5 cm | 很小 |
| 120 秒 | ~6° | ~10 cm | 可接受（通道宽 ~0.45m） |

---

## 3. 环境详解

### 3.1 观测空间（41 维）

| 组件 | 维度 | 范围 | 来源 | 作用 |
|------|------|------|------|------|
| **velocity** | 2 | 连续 | 机器人 IMU | 当前前进速度 + 角速度 |
| **lidar** | 36 | [0, 1] | RayCaster 下采样 | 360° 障碍物距离感知 |
| **exit_dir** | 2 | [-1, 1] | 计算得出 | 出口方向（机体坐标系下 sin/cos） |
| **exit_dist** | 1 | [0, 1] | 计算得出 | 到出口的归一化距离 |

**组装顺序：** `obs = [velocity(2), lidar(36), exit_dir(2), exit_dist(1)]` → 共 41 维

### 3.2 动作空间（2 维）

| 索引 | 含义 | 网络输出范围 | 物理量 |
|------|------|------------|--------|
| 0 | 线速度 | [-1, 1] | × 0.3 → [-0.3, 0.3] m/s |
| 1 | 角速度 | [-1, 1] | × 2.0 → [-2.0, 2.0] rad/s |

**动作到轮子的转换：**
```
action[0] × 0.3 m/s = 线速度 v
action[1] × 2.0 rad/s = 角速度 ω
→ 通过差分驱动公式转换为左右轮速度
→ 设置轮子关节速度目标
```

### 3.3 Episode 流程

```
1. [重置] 机器人放置在随机的迷宫格子中心，随机朝向
2. [循环] 每步:
   a. 读取观测 (速度 + LiDAR + 出口方向/距离)
   b. 策略网络输出动作 (线速度 + 角速度)
   c. 仿真执行 2 步 (decimation=2)
   d. 计算奖励
   e. 检查终止条件:
      - 距出口 < 0.5m → 成功 (terminated=True)
      - 超时 120 秒 → 失败 (truncated=True)
3. [结束] 记录 episode 数据，重置环境
```

### 3.4 时间参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 仿真时间步 (sim_dt) | 1/120 秒 | 物理引擎频率 |
| 决策间隔 (decimation) | 2 | 每 2 个仿真步执行一次策略 |
| 控制频率 | 60 Hz | = 120/2 |
| Episode 最大时长 | 120 秒 | 成功则提前结束 |
| 最大步数 | ~7200 | 120 × 60 |

---

## 4. 奖励函数详解

### 4.1 奖励组件

| 组件 | 权重 | 类型 | 触发条件 | 设计意图 |
|------|------|------|---------|---------|
| BFS 距离减少 | +5.0 | 密集 | 每步 | 引导机器人沿最短路径前进 |
| 到达出口 | +200.0 | 稀疏 | 距出口 < 0.5m | 最终目标的大额奖励 |
| 碰撞惩罚 | -5.0 | 密集 | LiDAR 最小距离 < 0.20m | 惩罚擦墙行为 |
| 时间惩罚 | -0.05 | 密集 | 每步固定 | 鼓励尽快完成 |
| 前进奖励 | +0.3 | 密集 | 前进速度 > 0 | 防止原地打转 |

### 4.2 计算公式

```python
# 1. BFS 距离奖励（最重要的密集信号）
bfs_reward = 5.0 × (prev_bfs_distance - curr_bfs_distance) / max_bfs_distance
# 机器人每靠近出口一步，获得正奖励；远离则负奖励

# 2. 到达出口奖励（一次性）
exit_reward = 200.0 × (distance_to_exit < 0.5).float()

# 3. 碰撞惩罚
collision = -5.0 × (lidar_min_distance < 0.20).float()

# 4. 时间惩罚
time_penalty = -0.05  # 每步固定

# 5. 前进奖励
forward_bonus = 0.3 × max(forward_velocity, 0)

# 总奖励
reward = bfs_reward + exit_reward + collision + time_penalty + forward_bonus
```

### 4.3 典型奖励值

| Episode 类型 | 典型奖励范围 | 原因 |
|-------------|------------|------|
| 完全随机 | -50 ~ -10 | 大量碰撞和时间惩罚 |
| 学会基本移动 | -10 ~ 50 | 减少碰撞，开始接近出口 |
| 成功逃脱 | 100 ~ 200 | +200 出口奖励 - 路径中的惩罚 |
| 快速逃脱 | 150 ~ 200 | 短路径 = 少时间惩罚 |

---

## 5. 训练配置

### 5.1 SAC 配置（推荐）

文件位置：`train/smartcleaningrobot/smartcleaningrobot/tasks/maze_escape/agents/sac.yaml`

| 参数 | 值 | 含义 |
|------|-----|------|
| **网络结构** | [256, 128, 64] | 3 层 MLP，逐层缩小 |
| **激活函数** | ELU | 比 ReLU 更平滑 |
| **经验池大小** | 500,000 | 存储 50 万条历史经验 |
| **批量大小** | 4,096 | 每次采样 4096 条更新 |
| **Actor 学习率** | 3.0e-4 | 策略网络更新速度 |
| **Critic 学习率** | 3.0e-4 | Q 网络更新速度 |
| **折扣因子 (γ)** | 0.99 | 重视长期奖励 |
| **软更新系数 (τ)** | 0.005 | 目标网络缓慢更新 |
| **自动熵调节** | 是 | 自动平衡探索与利用 |
| **初始熵值** | 1.0 | 初期高探索 |
| **随机步数** | 1,000 | 开始前纯随机探索 |
| **梯度裁剪** | 1.0 | 防止梯度爆炸 |
| **总训练步数** | 3,000,000 | 3M 步 |
| **种子** | 42 | 可复现 |

### 5.2 PPO 配置（备选）

文件位置：`train/smartcleaningrobot/smartcleaningrobot/tasks/maze_escape/agents/ppo.yaml`

| 参数 | 值 | 含义 |
|------|-----|------|
| **网络结构** | [128, 64] | 2 层 MLP，更小更快 |
| **Rollout 长度** | 128 | 每次收集 128 步经验 |
| **学习 epoch** | 8 | 每批数据重复学 8 次 |
| **Mini batches** | 16 | 分 16 份小批量更新 |
| **学习率** | 1.0e-3 | PPO 可以用更大的 LR |
| **折扣因子 (γ)** | 0.995 | 略高于 SAC |
| **GAE λ** | 0.95 | 优势函数平滑 |
| **裁剪比率 (ε)** | 0.2 | PPO 核心参数 |
| **熵损失权重** | 0.02 | 探索力度 |
| **总训练步数** | 2,000,000 | 2M 步 |

### 5.3 SAC vs PPO 对比

| 维度 | SAC | PPO |
|------|-----|-----|
| 样本效率 | 高（经验回放） | 低（on-policy） |
| 稳定性 | 中等 | 高 |
| 调参难度 | 中等 | 简单 |
| 适合稀疏奖励 | 是（重复利用成功经验） | 一般 |
| 推荐用于此任务 | **是** | 备选 |

### 5.4 与传统方法对比

| 算法 | 类型 | 需要地图 | 适用范围 | 原因 |
|------|------|---------|---------|------|
| A* | 图搜索 | 是 | 有完整地图时最优 | 无地图不可用 |
| 左手法则 | 反应式 | 否 | 仅简单连通迷宫 | 有环路则死循环 |
| PPO | On-policy RL | 否 | 通用 | 样本效率低，稀疏奖励学习慢 |
| **SAC** | Off-policy RL | 否 | **通用，最适合本任务** | 内置熵最大化 + 经验回放 |

### 5.5 SAC 网络架构详解

SAC 使用 **5 个独立网络**（对比 PPO 的 2 个）：

```
Policy (Actor):                           1 个
    obs(41) → [256, 128, 64] ELU → actions(2)
    输出: 均值 + 对数标准差 → 高斯分布采样

Critic (Q 网络):                          2 个（双 Q 减少过估计）
    [obs(41), act(2)] → [256, 128, 64] ELU → Q值(1)

Target Critic:                            2 个（软更新 τ=0.005）
    结构同 Critic，参数缓慢跟踪
```

**SAC vs PPO 结构差异：**

| 方面 | PPO | SAC |
|------|-----|-----|
| 网络数量 | 2 (policy + value) | 5 (policy + 2 critics + 2 targets) |
| Critic 输入 | 仅观测 | 观测 + 动作（拼接后 43 维） |
| 数据存储 | Rollout buffer（用完即弃） | Replay buffer（50 万条可重复利用） |
| 策略更新 | 限制 KL 散度 | 最大熵目标 + 自动温度调节 |

---

## 6. 训练步骤

### 6.1 第一步：快速验证

```bash
# 确认环境可用
python train/scripts/common/list_envs.py --keyword MazeEscape

# 零动作测试（机器人站着不动，检查场景是否正确加载）
python train/scripts/common/zero_agent.py \
  --task SmartCleaningRobot-MazeEscape-v0 --num_envs 1

# 随机动作测试（机器人随机乱走，检查物理仿真）
python train/scripts/common/random_agent.py \
  --task SmartCleaningRobot-MazeEscape-v0 --num_envs 1 --headless
```

### 6.2 第二步：正式训练

```bash
# SAC 训练（推荐）
python train/scripts/common/train.py \
  --task SmartCleaningRobot-MazeEscape-v0 \
  --algorithm SAC \
  --num_envs 128 \
  --headless \
  --seed 42

# 或 PPO 训练
python train/scripts/common/train.py \
  --task SmartCleaningRobot-MazeEscape-v0 \
  --algorithm PPO \
  --num_envs 128 \
  --headless \
  --seed 42
```

**训练过程中会看到类似输出：**
```
[INFO] Training SmartCleaningRobot-MazeEscape-v0 with SAC
[INFO] Logging to: train/logs/skrl/maze_escape/2026-03-21_10-00-00_sac_torch
...
[5000 steps] mean_reward: -42.31 | escape_rate: 0.02 | ep_length: 3421
[10000 steps] mean_reward: -28.47 | escape_rate: 0.08 | ep_length: 2890
[20000 steps] mean_reward: -5.12 | escape_rate: 0.22 | ep_length: 1856
...
```

### 6.3 第三步：监控训练（可选，另一个终端）

```bash
# TensorBoard 实时监控
tensorboard --logdir train/logs/skrl/maze_escape
# 打开浏览器访问 http://localhost:6006
```

### 6.4 第四步：绘制训练曲线

训练结束后（或中途也可以）：

```bash
python train/scripts/common/plot_training.py \
  train/logs/skrl/maze_escape/<你的运行目录> \
  --smooth 10
```

### 6.5 第五步：回放最佳策略

```bash
python train/scripts/common/play.py \
  --task SmartCleaningRobot-MazeEscape-v0 \
  --algorithm SAC \
  --num_envs 1 \
  --video \
  --video_length 500
```

---

## 7. 结果分析

### 7.1 预期训练时间线（SAC, 128 envs）

| 阶段 | 步数 | 预期奖励 | 预期逃脱率 | 行为描述 |
|------|------|---------|-----------|---------|
| 随机探索 | 0 - 100K | -100 ~ 0 | 0-5% | 碰墙、原地打转 |
| 初步学习 | 100K - 500K | 0 ~ 50 | 5-30% | 学会前进、减少碰撞 |
| 快速提升 | 500K - 1.5M | 50 ~ 150 | 30-80% | 学会路径规划 |
| 收敛稳定 | 1.5M - 3M | 150+ | 80-95% | 行为稳定优化 |

### 7.2 关键指标判读

| 指标 | 好的信号 | 坏的信号 |
|------|---------|---------|
| mean_reward | 稳步上升至 150+ | 始终 < 0 或剧烈震荡 |
| escape_rate | 上升至 80%+ | 始终 < 10% |
| mean_ep_length | 逐渐下降 | 始终在最大值附近 |
| rolling_mean_reward | 平滑上升 | 上升后回落 |

### 7.3 如何判断训练是否成功？

**成功标准：**
- escape_rate > 80%（至少 80% 的 episode 能找到出口）
- mean_reward > 100（稳定获得正奖励）
- mean_ep_length 明显低于最大值（机器人不是靠超时结束）

**如果没有达标：**
1. 检查训练曲线是否还在上升（可能需要更多步数）
2. 尝试不同的种子（`--seed 43`）
3. 参考 [TRAINING_GUIDE.md](../../TRAINING_GUIDE.md) 第 12 节的调参建议

### 7.4 回放时观察要点

在 `play.py` 回放时，关注：

| 行为 | 好的表现 | 需要改进 |
|------|---------|---------|
| 导航效率 | 直奔出口方向，很少走死胡同 | 在迷宫中打转、走回头路 |
| 避障 | 平滑地绕过墙壁 | 频繁擦墙或卡在角落 |
| 转弯 | 在分叉处果断选择 | 在分叉处犹豫/抖动 |
| 速度 | 保持匀速前进 | 频繁停顿或忽快忽慢 |

---

## 8. 常见问题

### Q1: 训练了很久但逃脱率始终为 0

**可能原因：** 出口阈值或出口位置配置错误。
**排查步骤：**
1. 用 `zero_agent.py` 启动，在 GUI 中检查出口位置
2. 确认 `exit_threshold = 0.5` 在 `env_cfg.py` 中

### Q2: 训练开始时奖励为 -100 左右

**这是正常的。** 随机策略会不断碰撞墙壁（-5.0 × 每次碰撞）+ 时间惩罚（-0.05 × 每步）。随着训练进行，奖励会上升。

### Q3: SAC 比 PPO 慢很多

**正常现象。** SAC 每步需要更新 5 个网络（vs PPO 的 2 个），计算量更大。但 SAC 的样本效率更高，总训练步数可以更少。

### Q4: 训练曲线剧烈震荡

**解决方案：**
- 增大 `plot_training.py --smooth` 值来平滑显示
- 如果实际性能不稳定，减小 `learning_rate`

### Q5: 机器人总是沿一个方向旋转

**原因：** 策略陷入局部最优。
**解决：** 增大探索力度——SAC 增大 `initial_entropy_value`，PPO 增大 `entropy_loss_scale`。

---

## 9. 部署与测试

训练完成后，可以将模型部署到 ROS 2 + Gazebo 仿真，或真实机器人。

### 9.1 导出模型

将 skrl checkpoint 转为 TorchScript 格式：

```bash
python deploy/scripts/export_model.py \
    --task maze \
    --checkpoint train/logs/skrl/maze_escape/<run>/checkpoints/best_agent.pt \
    --output deploy/models/maze_agent_scripted.pt
```

验证输出：输入 (1, 41) → 输出 (1, 2)。

### 9.2 Gazebo 仿真

```bash
# 1. 生成与训练一致的迷宫（同 DFS 算法 + seed=42）
python deploy/scripts/gen_maze_world.py

# 2. 启动 Gazebo（在 WSL2 中）
ros2 launch cleaning_robot_ros gazebo.launch.py world:=maze.world

# 3. 启动 RL 策略节点
ros2 run cleaning_robot_ros policy_node --ros-args \
    -p policy_type:=maze_model \
    -p model_path:=deploy/models/maze_agent_scripted.pt
```

### 9.3 传统方法对比测试

```bash
# 左手法则（仅适用于简单连通迷宫）
ros2 run cleaning_robot_ros policy_node --ros-args -p policy_type:=wall_follower
```

| 指标 | 左手法则 | SAC 训练模型 |
|------|----------|-------------|
| 需要全局地图 | 否 | 否 |
| 需要出口方向 | 否 | 是（里程计估算） |
| 适用迷宫类型 | 仅简单连通 | 任意结构 |
| 动作平滑度 | 低（急转） | 高（连续控制） |
| 对里程计漂移敏感 | 否 | 轻度敏感 |

### 9.4 真实机器人注意事项

- **里程计对齐**：将机器人放在已知位置，确保初始朝向与训练一致
- **LiDAR 对齐**：确保射线数=360，最大距离=3.5m，0号射线指向正前方
- **降速部署**：如果表现不稳，可降低 `max_linear_vel` 到 0.2 m/s
- **漂移补偿**：120 秒内漂移通常 <10cm，对 0.45m 宽通道足够；严重时可用 EKF 融合

---

## 参考文件

| 文件 | 路径 | 内容 |
|------|------|------|
| 环境实现 | `train/.../tasks/maze_escape/env.py` | 观测/奖励/终止逻辑 |
| 环境配置 | `train/.../tasks/maze_escape/env_cfg.py` | 空间维度/物理参数 |
| SAC 超参 | `train/.../tasks/maze_escape/agents/sac.yaml` | SAC 网络/学习参数 |
| PPO 超参 | `train/.../tasks/maze_escape/agents/ppo.yaml` | PPO 网络/学习参数 |
| 机器人资产 | `train/.../robot_assets/jetbot.py` | JetBot 物理参数 |
| 训练脚本 | `train/scripts/common/train.py` | CLI 入口 |
