# V2 训练端说明书（Isaac Lab）

> 对应目录: `train/smartcleaningrobot/`
> 运行环境: Windows 原生, conda 环境 `isaaclab`
> 最后更新: 2026-03-03

---

## 一、项目概述

V2 基于 Isaac Lab 框架，使用强化学习 (PPO) 训练 JetBot 差速驱动机器人。项目包含两个独立任务：

- **机器人**: JetBot（差速驱动，2轮，内置摄像头，外挂 LiDAR）
- **场景**: Simple / Maze / Hospital / Office / Warehouse（5 个场景）
- **RL 框架**: Isaac Lab DirectRLEnv + skrl PPO

### 已注册任务

| 环境 ID | 任务 | 说明 |
|---------|------|------|
| `SmartCleaningRobot-Coverage-v0` | 覆盖清扫 | 机器人最大化地面覆盖率 |
| `SmartCleaningRobot-MazeEscape-v0` | 迷宫逃脱 | 机器人以最快速度找到出口 |

---

## 二、项目结构

```
SmartCleaningRobot/
├── environment.yml                  # conda 环境规格（env name: isaaclab）
├── 自用说明书_train.md              # 本文件
├── 自用说明书_deployment.md         # 部署端说明书
│
├── train/
│   ├── smartcleaningrobot/                         # Isaac Lab 扩展包 (pip install -e train/smartcleaningrobot)
│   │   └── smartcleaningrobot/
│   │       ├── robot_assets/jetbot.py  # JetBot 配置 + LiDAR + Camera + 差速驱动
│   │       ├── scenes/builders/     # 场景构建器（maze, simple, usd_scene）
│   │       ├── tasks/
│   │       │   ├── maze_escape/     # 迷宫逃脱任务（env, env_cfg, agents/sac.yaml）
│   │       │   └── coverage/        # 覆盖清扫任务（env, env_cfg, agents/ppo.yaml）
│   │       ├── utils/               # coverage_tracker, map_viewer, metrics
│   │       └── traditional/         # random_explorer
│   └── scripts/
│       ├── common/                  # 通用脚本（用 --task 指定任务）
│       │   ├── train.py             # RL 训练入口
│       │   ├── play.py              # 训练后策略回放
│       │   ├── list_envs.py         # 列出已注册 Gym 环境
│       │   ├── random_agent.py      # 随机动作测试
│       │   └── zero_agent.py        # 零动作测试
│       ├── MazeDFSv1_Jetbot/
│       │   ├── MazeDFSv1_Jetbot_EscapeMaze.py   # 迷宫逃脱可视化演示
│       │   └── benchmark/maze/      # 对比实验（A*, 左手法则, RL）
│       └── Simple_Jetbot/
│           └── Simple_Jetbot_Coverage.py         # 覆盖清扫可视化演示
│
└── deploy/                          # 部署端（详见 自用说明书_deployment.md）
```

---

## 三、场景一览

| 场景 | 类型 | 来源 | 大小 | 特点 | 状态 |
|------|------|------|------|------|------|
| `simple` | 程序化 | `builders/simple.py` | 8m × 8m | 4 面墙 + 2 个障碍物，快速测试 | 已验证 |
| `maze` | 程序化 | `builders/maze.py` | ~7.7m × 7.7m | 8×8 DFS 迷宫，死胡同，南墙出口 | 已验证 |
| `hospital` | USD | Nucleus 服务器 | 30m × 30m | 走廊 + 病房，复杂室内 | 已验证 |
| `office` | USD | Nucleus 服务器 | 20m × 20m | 办公隔断 + 家具 | 已验证 |
| `warehouse` | USD | Nucleus 服务器 | 24m × 24m | 仓库货架 + 开阔地 | 已验证 |

### 迷宫场景说明

迷宫由 DFS 算法生成（seed=42，可复现），与 Gazebo 部署端完全一致。

当前特性：
- **Simply Connected（简单连通）**: 任意两点间只有唯一路径，左手法则可解
- **死胡同 (dead-ends)**: DFS 自然产生的分支末端，RL agent 需学会回退
- **唯一出口**: 南墙中央，机器人可从任意位置出发
- **无天花板**: 俯视可见机器人
- **出生点随机化**: MazeEscape 任务中，机器人随机出现在 64 个合法通道格子之一

可选扩展（代码中已实现但默认关闭）：
- **空腔 (chambers)**: 2×2 开放房间，会创建环路
- **环路 (loops)**: 随机移除内墙，使左手法则失效
- 启用方法：取消 `builders/maze.py` 中 `_carve_chamber` / `_add_loops` 的注释

### 如何添加新场景

1. 在 `scenes/builders/` 下创建 `my_scene.py`，实现 `def build_my_scene(scene_config) -> None`
2. 在 `scenes/builders/__init__.py` 的 `SCENE_BUILDERS` 字典中注册
3. 在 `scenes/scene_cfg.py` 中添加 `SceneConfig` 条目（含 `scene_bounds`, `env_spacing` 等）
4. 在对应的任务演示脚本（`scripts/Simple_Jetbot/Simple_Jetbot_Coverage.py` 或 `scripts/MazeDFSv1_Jetbot/MazeDFSv1_Jetbot_EscapeMaze.py`）的 `--scene choices` 中添加名称

---

## 四、核心数据流

```
动作空间 (2): [linear_vel, angular_vel]  归一化到 [-1, 1]
    ↓ 差速驱动转换
    ↓ left_wheel_vel, right_wheel_vel
    ↓
观测空间 (1066):
    ├── 位姿 (3): x, y, yaw
    ├── 速度 (2): 前进速度, 角速度
    ├── LiDAR (36): 360射线降采样10倍, 归一化到 [0,1]
    ├── 局部覆盖图 (1024): 32×32 以机器人为中心的覆盖网格
    └── 覆盖率 (1): 全局覆盖比例 [0,1]

奖励:
    + 10.0 × 新覆盖格子数
    + 50.0 × 首次达到里程碑 (50%/75%/90%/95%)
    - 5.0  × 碰撞检测 (LiDAR 最小距离 < 0.15m)
    - 0.01 × 每步时间惩罚

Camera: 每步采集 RGB 但不进入观测（预留给小物件模块，默认关闭）
```

### 4.2 迷宫逃脱（MazeEscape）数据流

此任务与覆盖清扫完全独立，目标是尽快找到出口。

```
动作空间 (2): [linear_vel, angular_vel]  归一化到 [-1, 1]（与覆盖任务一致）

观测空间 (44):
    ├── 位姿 (3): x, y, yaw
    ├── 速度 (2): 前进速度, 角速度
    ├── LiDAR (36): 360射线降采样10倍, 归一化到 [0,1]
    ├── 出口方向 (2): 机器人到出口的单位向量 (dx, dy)
    └── 出口距离 (1): 欧几里得距离 / 迷宫对角线长度, 归一化到 [0,1]

奖励:
    + 5.0  × 距离缩减量 (每步接近出口的程度, 归一化)
    + 200.0 × 到达出口 (一次性大奖励)
    + 0.5  × 探索新格子 (覆盖追踪, 防止原地打转)
    - 1.0  × 碰撞 (LiDAR 最小距离 < 0.15m)
    - 0.05 × 每步时间惩罚 (催促快速逃脱)

结束条件:
    - 成功: 机器人距出口 < 0.3m
    - 超时: 120秒

出生点: 随机选择迷宫中任意合法通道格子 (64 个候选位置)
出口位置: 南墙中央 (0.0, -3.15) — 与 Gazebo 部署端一致
```

---

## 五、环境配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `episode_length_s` | 300s | 每回合最长 5 分钟 |
| `decimation` | 2 | 每 2 物理步执行 1 次策略 |
| `sim.dt` | 1/120 | 物理仿真步长 |
| `num_envs` | 64 | 默认并行环境数 |
| `max_linear_vel` | 0.3 m/s | 最大前进速度 |
| `max_angular_vel` | 2.0 rad/s | 最大角速度 |
| `coverage_grid_resolution` | 0.05m | 覆盖网格分辨率 |
| `coverage_done_threshold` | 0.98 | 覆盖率达到 98% 时回合结束 |
| `enable_camera` | False | Camera 默认关闭，需要时加 `--enable_cameras` |

### 5.2 迷宫逃脱环境配置 (MazeEscapeEnvCfg)

| 参数 | 值 | 说明 |
|------|-----|------|
| `episode_length_s` | 120s | 每回合最长 2 分钟 |
| `observation_space` | 44 | pose(3) + vel(2) + lidar(36) + exit(3) |
| `num_envs` | 128 | 默认并行环境数（迷宫较小，可多开） |
| `exit_pos` | (0.0, -3.15) | 出口世界坐标 |
| `exit_threshold` | 0.3m | 到达出口的判定距离 |
| `randomize_spawn` | True | 每回合随机出生点 |
| `rew_dist_reduction` | 5.0 | 距离缩减奖励权重 |
| `rew_exit_bonus` | 200.0 | 到达出口一次性奖励 |
| `rew_collision_penalty` | -1.0 | 碰撞惩罚（比清扫任务轻） |
| `rew_time_penalty` | -0.05 | 时间惩罚（比清扫任务重） |
| `rew_exploration` | 0.5 | 探索新区域小奖励 |

---

## 六、测试步骤

### 前置条件

- Isaac Sim 已安装（D:\Isaac_Sim）
- Isaac Lab 已安装（D:\Isaac_Sim\IsaacLab）
- conda 环境 `isaaclab` 已配置

> Windows 上不能直接用 `isaaclab` 命令，统一用 `python` 代替 `isaaclab -p`。

### Step 1: 激活环境并安装项目

```bash
conda activate isaaclab
cd D:\Projects\SmartCleaningRobot
python -m pip install -e train/smartcleaningrobot
```

### Step 2: 验证环境注册

```bash
python train/scripts/common/list_envs.py
```

预期输出的表格中应包含：
- `SmartCleaningRobot-Coverage-v0`（覆盖清扫）
- `SmartCleaningRobot-MazeEscape-v0`（迷宫逃脱）

### Step 3: Headless 快速测试（无 GUI）

适合验证场景加载和物理仿真是否正常，不弹窗，几秒完成：

```bash
# 3a: 简单房间
python train/scripts/Simple_Jetbot/Simple_Jetbot_Coverage.py --scene simple --policy random --num_envs 1 --headless --max_steps 200

# 3b: 迷宫
python train/scripts/Simple_Jetbot/Simple_Jetbot_Coverage.py --scene maze --policy random --num_envs 1 --headless --max_steps 200

# 3c: 医院
python train/scripts/Simple_Jetbot/Simple_Jetbot_Coverage.py --scene hospital --policy random --num_envs 1 --headless --max_steps 100
```

每个应在 30 秒内完成，exit code = 0 即为通过。

### Step 4: GUI 可视化演示

弹出 Isaac Sim 窗口观察机器人。首次打开需等待 RTX 着色器编译（可能 1-5 分钟）。

```bash
# 4a: 零动作 — 验证场景加载、机器人出现、不动
python train/scripts/Simple_Jetbot/Simple_Jetbot_Coverage.py --scene simple --policy zero --num_envs 1

# 4b: 随机动作 — 验证机器人能动
python train/scripts/Simple_Jetbot/Simple_Jetbot_Coverage.py --scene hospital --policy random --num_envs 1

# 4c: bump-and-turn — 验证完整管线 (LiDAR避障 + 覆盖追踪)
python train/scripts/Simple_Jetbot/Simple_Jetbot_Coverage.py --scene hospital --policy explorer --num_envs 1

# 4d: 迷宫场景
python train/scripts/Simple_Jetbot/Simple_Jetbot_Coverage.py --scene maze --policy random --num_envs 1 --max_steps 3000
```

> **提示**：Isaac Sim 打开后如果界面卡住，是 RTX 着色器编译，请耐心等待，不是崩溃。
> USD 场景有天花板，选中天花板按 `E` 键可隐藏以便俯视机器人。

### Step 5: 迷宫逃脱任务测试

```bash
# 5a: Headless 快速验证迷宫逃脱环境
python train/scripts/MazeDFSv1_Jetbot/MazeDFSv1_Jetbot_EscapeMaze.py --policy random --num_envs 1 --headless --max_steps 200

# 5b: GUI 可视化 — 随机动作测试
python train/scripts/MazeDFSv1_Jetbot/MazeDFSv1_Jetbot_EscapeMaze.py --policy random --num_envs 1
```

### Step 6: RL 训练

#### 覆盖清扫训练

```bash
# 无头模式 64 环境并行训练
python train/scripts/common/train.py --task SmartCleaningRobot-Coverage-v0 --num_envs 64 --headless

# 带可视化训练（较慢，用于观察）
python train/scripts/common/train.py --task SmartCleaningRobot-Coverage-v0 --num_envs 4
```

训练日志输出在 `train/logs/skrl/cleaning_coverage/`。

#### 迷宫逃脱训练

```bash
# 无头模式 128 环境并行训练
python train/scripts/common/train.py --task SmartCleaningRobot-MazeEscape-v0 --num_envs 128 --headless

# 带可视化训练
python train/scripts/common/train.py --task SmartCleaningRobot-MazeEscape-v0 --num_envs 4
```

训练日志输出在 `train/logs/skrl/maze_escape/`。

### Step 7: 策略回放

```bash
# 覆盖清扫策略
python train/scripts/common/play.py --task SmartCleaningRobot-Coverage-v0 --num_envs 1

# 迷宫逃脱策略
python train/scripts/common/play.py --task SmartCleaningRobot-MazeEscape-v0 --num_envs 1
```

### 切换场景

```bash
# 覆盖清扫（默认 simple 场景）
python train/scripts/Simple_Jetbot/Simple_Jetbot_Coverage.py --scene simple --policy explorer --num_envs 1
python train/scripts/Simple_Jetbot/Simple_Jetbot_Coverage.py --scene maze --policy explorer --num_envs 1
python train/scripts/Simple_Jetbot/Simple_Jetbot_Coverage.py --scene hospital --policy explorer --num_envs 1
python train/scripts/Simple_Jetbot/Simple_Jetbot_Coverage.py --scene warehouse --policy explorer --num_envs 1
python train/scripts/Simple_Jetbot/Simple_Jetbot_Coverage.py --scene office --policy explorer --num_envs 1

# 迷宫逃脱（默认 maze 场景）
python train/scripts/MazeDFSv1_Jetbot/MazeDFSv1_Jetbot_EscapeMaze.py --policy random --num_envs 1
```

---

## 七、场景构建器架构

场景生成逻辑已从 `cleaningrobotv2_env.py` 中完全解耦，env 只需一行调用：

```python
# cleaningrobotv2_env.py → _setup_scene()
build_scene(self.scene_config)
```

各 builder 之间的关系：

```
scenes/builders/__init__.py
    SCENE_BUILDERS = {
        "simple":    build_simple_scene,     ← simple.py
        "maze":      build_maze_scene,       ← maze.py
        "office":    build_usd_scene,        ← usd_scene.py
        "hospital":  build_usd_scene,        ← usd_scene.py
        "warehouse": build_usd_scene,        ← usd_scene.py
    }

    build_scene(scene_config)  → 根据 scene_config.name 分发到对应 builder

_common.py
    spawn_wall_scene(walls, color)      → 逐墙生成 MeshCuboidCfg (物理碰撞)
    _create_combined_raycaster_mesh()   → 合并所有墙为单个 UsdGeom.Mesh (RayCaster)
```

`SceneConfig` 数据类字段说明：

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | str | 场景标识，对应 `--scene` 参数 |
| `usd_path` | str | USD 路径（程序化场景为空字符串） |
| `scene_bounds` | (x_min, y_min, x_max, y_max) | 覆盖追踪边界 |
| `env_spacing` | float | 多环境间距 |
| `fixed_spawn_pos` | (x, y) 或 None | 固定出生点（None 则随机） |
| `description` | str | 场景描述 |

---

## 八、可能遇到的问题

### 1. Isaac Sim 打开后卡住

**原因**: 首次 GUI 模式需要编译 RTX 着色器，可能需要 1-5 分钟。
**处理**: 耐心等待，不要强制关闭。后续启动会快很多。

### 2. 场景 USD 加载失败

**症状**: 报错找不到 USD 文件
**排查**: 在 Isaac Sim 的 Content 浏览器中搜索 "hospital" 或 "warehouse"
**修复**: 修改 `scenes/scene_cfg.py` 中对应场景的 `usd_path`

### 3. RayCaster 报错 "only supports one mesh prim"

**原因**: 程序化场景必须将所有墙合并为单个 mesh
**现状**: `_common.py` 中已处理，不应再出现此问题

### 4. 多环境模式异常

**排查**: 先用 `--num_envs 1` 确认单环境正常
**修复**: 复杂 USD 场景减少 `num_envs`，或用程序化场景（simple/maze）

### 5. `list_envs.py` 输出空表

**原因**: 脚本中的过滤器不匹配
**修复**: 确认 `list_envs.py` 过滤条件为 `"SmartCleaningRobot-"`

### 6. Camera 相关报错

**症状**: `RuntimeError: A camera was spawned without the --enable_cameras flag`
**修复**: `enable_camera` 默认为 False，如需启用需加 `--enable_cameras` 参数

---

## 九、已验证的 Isaac Lab 资源路径

| 资源 | 路径 | 状态 |
|------|------|------|
| JetBot USD | `{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd` | 已验证 |
| Hospital 场景 | `{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd` | 已验证 |
| Warehouse 场景 | `{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd` | 已验证 |
| Office 场景 | `{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd` | 已验证 |
| Simple 场景 | 程序化生成（无 USD） | 已验证 |
| Maze 场景 | 程序化生成（无 USD） | 已验证 |

---

## 十、原理

### 10.1 神经网络架构

本项目使用两个结构相同的 MLP（多层感知机）网络，分别承担不同角色：

#### 策略网络（Policy Network）

负责决策——给定当前观测，输出机器人应该执行的动作。

```
输入层 (1066 维)
  │
  ├── 位姿 (3): x, y, yaw
  ├── 速度 (2): 前进线速度, 旋转角速度
  ├── LiDAR (36): 360° 降采样到 36 条射线, 归一化 [0,1]
  ├── 局部覆盖地图 (1024): 32×32 网格展平, 0=未扫/1=已扫
  └── 全局覆盖率 (1): 整个房间的已覆盖比例 [0,1]
  │
  ▼
Linear(1066, 256) → ELU 激活
  │
  ▼
Linear(256, 128) → ELU 激活
  │
  ▼
Linear(128, 64) → ELU 激活
  │
  ▼
Linear(64, 2) → Tanh 激活
  │
  ▼
输出层 (2 维)
  ├── actions[0]: 归一化线速度 ∈ [-1, 1]  →  × 0.3 = 实际线速度 (m/s)
  └── actions[1]: 归一化角速度 ∈ [-1, 1]  →  × 2.0 = 实际角速度 (rad/s)
```

对应配置（`skrl_ppo_cfg.yaml`）：

```yaml
policy:
  class: GaussianMixin
  network:
    - name: net
      input: OBSERVATIONS
      layers: [256, 128, 64]
      activations: elu
  output: ACTIONS
```

**关于"高斯策略"**: 策略网络的输出不是一个确定的动作值，而是一个**高斯分布的均值**。网络还维护一个可学习的 `log_std` 参数（初始值 0，即标准差为 1），实际动作从 N(μ, σ²) 中采样。这保证了训练早期的探索性——动作带有随机扰动，不会一开始就固定在某个模式上。随着训练进行，σ 逐渐减小，动作趋于确定。

#### 价值网络（Value Network）

负责评估——给定当前观测，估计"从这个状态开始，未来总共还能拿多少奖励"。

```
输入层 (1066 维)  ← 和策略网络输入完全一样
  │
  ▼
Linear(1066, 256) → ELU
  ▼
Linear(256, 128) → ELU
  ▼
Linear(128, 64) → ELU
  ▼
Linear(64, 1)         ← 没有激活函数，因为价值可以是任意实数
  │
  ▼
输出: 1 个标量 V(s)   ← "这个状态值多少分"
```

**价值网络只在训练时使用**，部署时只需要策略网络。它的作用是帮助 PPO 计算"这个动作比平均水平好多少"（优势函数），详见下文。

#### 为什么用 MLP 而不是 CNN/Transformer？

- 输入是**结构化的低维向量**（不是原始图像），MLP 足以建模这种映射
- 32×32 的覆盖地图虽然有空间结构，但它是简单的 0/1 二值网格，展平成 1024 维后 MLP 能学会
- MLP 推理极快，适合部署到边缘设备（Jetson Nano 等）
- 机器人 RL 领域的主流选择（ETH Zurich ANYmal 四足机器人、NVIDIA Isaac Lab 官方示例等均使用 MLP）

#### 可能的改进方向

- **CNN 编码器**：用小型 CNN 处理 32×32 覆盖地图，提取空间特征后再和其他观测拼接进 MLP，可能提升对空间模式的感知
- **LSTM/GRU**：当前观测是部分可观测的（只有局部视野），加入循环层让策略拥有"记忆"，可能减少重复覆盖

---

### 10.2 PPO 算法原理

PPO（Proximal Policy Optimization，近端策略优化）是目前机器人强化学习中最常用的算法。以下从零解释它的完整工作原理。

#### 10.2.1 强化学习的基本框架

强化学习的核心是一个循环：

```
  ┌─────────────────────────────────────────┐
  │                                         │
  ▼                                         │
环境(仿真器)  ──观测 s_t──▶  策略网络 π(a|s)  │
  │                           │              │
  │                        动作 a_t          │
  ◄────────────────────────────┘              │
  │                                         │
  ├──奖励 r_t ──▶  用于更新网络参数           │
  ├──下一个观测 s_{t+1} ────────────────────-─┘
  └──是否结束 done
```

每一步 t：
1. 环境给出当前观测 s_t（1066 维向量）
2. 策略网络根据 s_t 输出动作 a_t（2 维：线速度、角速度）
3. 环境执行 a_t，返回奖励 r_t 和下一个观测 s_{t+1}
4. 重复，直到回合结束（覆盖率 ≥ 98% 或超时 300s）

**目标：找到一组网络参数 θ，使得累积奖励的期望最大化。**

#### 10.2.2 三个核心概念

**（1）回报 G_t（Return）—— 未来奖励的折扣总和**

机器人不只关心当前这一步的奖励，还关心未来。但越远的未来越不确定，所以要打折扣：

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·r_{t+3} + ...
```

其中 γ = 0.99（折扣因子）。含义：下一步的奖励打 99 折，再下一步打 98 折……这让策略更重视近期回报。

**（2）价值函数 V(s) —— "这个状态值多少分"**

价值网络学习的就是 V(s)，即"从状态 s 开始，按照当前策略走下去，预期的回报 G 是多少"。

例如：
- 机器人在空旷区域中心，周围大片未覆盖 → V(s) 高（未来能拿很多覆盖奖励）
- 机器人被困在角落，周围都已扫过 → V(s) 低（未来奖励有限）

**（3）优势函数 A(s, a) —— "这个动作比平均水平好多少"**

```
A(s_t, a_t) = "实际拿到的回报" - "价值网络预测的平均回报"
            = (r_t + γ·V(s_{t+1})) - V(s_t)
```

- A > 0：这个动作比预期好 → 应该增大这个动作的概率
- A < 0：这个动作比预期差 → 应该减小这个动作的概率
- A ≈ 0：这个动作和预期差不多 → 概率基本不变

实际使用中，优势函数通过 **GAE（Generalized Advantage Estimation）** 计算，它是多步优势的加权平均，兼顾偏差和方差：

```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)          ← 单步 TD 误差
A_t^GAE = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
```

其中 λ = 0.95（GAE 参数）。λ 越大，用越多未来信息（方差大但偏差小）；λ 越小，越依赖当前价值估计（方差小但偏差大）。0.95 是常用值。

#### 10.2.3 PPO 的核心公式

普通的策略梯度算法会直接最大化：

```
L = E[ log π_θ(a|s) · A(s,a) ]
```

含义：如果动作 a 的优势 A > 0，就增大 log π（即增大该动作的概率）；反之减小。

但这有个问题：**一次更新太大，策略可能突变，训练崩溃**。PPO 的核心创新就是限制每次更新的幅度。

定义**概率比**：

```
r(θ) = π_θ(a|s) / π_θ_old(a|s)
```

即"新策略选择这个动作的概率" / "旧策略选择这个动作的概率"。如果新旧策略一样，r = 1。

PPO 的裁剪目标函数（Clipped Surrogate Objective）：

```
L_CLIP = E[ min( r(θ)·A,  clip(r(θ), 1-ε, 1+ε)·A ) ]
```

其中 ε = 0.2（裁剪范围，对应配置中的 `ratio_clip: 0.2`）。

**直觉理解**：
- 当 A > 0（好动作）：r(θ) 想变大（增大概率），但被裁剪到最多 1.2，不能一次加太多
- 当 A < 0（坏动作）：r(θ) 想变小（减小概率），但被裁剪到最少 0.8，不能一次减太多
- 这保证了策略更新的"步子不会迈太大"，训练更稳定

#### 10.2.4 完整的损失函数

PPO 的总损失由三部分组成：

```
L_total = L_CLIP - c₁·L_VF + c₂·L_entropy
```

| 组成部分 | 公式 | 作用 | 本项目配置 |
|---------|------|------|-----------|
| L_CLIP | 见上文 | 策略优化（核心） | ratio_clip = 0.2 |
| L_VF | MSE(V(s), G_t) | 价值网络拟合实际回报 | value_loss_scale = 2.0 |
| L_entropy | -Σ π·log(π) | 鼓励探索，防止策略过早收敛 | entropy_loss_scale = 0.01 |

- **L_VF**：价值网络的预测 V(s) 应该尽量接近真实回报 G_t，用均方误差训练
- **L_entropy**：策略的熵越大，动作分布越均匀（越随机）。加一个小权重的熵奖励，防止策略太早"确信"某个动作

#### 10.2.5 训练迭代过程

每一轮训练（对应 `rollouts: 64` 步）的完整流程：

```
┌─────────────────────────────────────────────────────────────────────┐
│ 第一阶段：数据采集 (Rollout)                                        │
│                                                                     │
│   64 个并行环境同时运行，每个环境跑 64 步                             │
│   每步记录: (s_t, a_t, r_t, s_{t+1}, log π_old(a_t|s_t), V(s_t))  │
│   共收集 64 × 64 = 4096 条 transition                              │
│                                                                     │
│   注意：这里用的是"旧"策略 π_old 采集数据                            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 第二阶段：计算优势                                                   │
│                                                                     │
│   用 GAE 公式计算每条 transition 的优势 A_t                          │
│   计算回报 G_t = A_t + V(s_t)                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 第三阶段：网络更新（重复 8 个 epoch）                                 │
│                                                                     │
│   for epoch in range(8):            ← learning_epochs = 8          │
│       将 4096 条数据随机分成 16 个 mini-batch (每个 256 条)          │
│       for each mini-batch:          ← mini_batches = 16            │
│           1. 用当前 π_θ 计算 log π_θ(a|s)                          │
│           2. 算概率比 r(θ) = exp(log π_θ - log π_old)              │
│           3. 算裁剪损失 L_CLIP                                      │
│           4. 算价值损失 L_VF                                        │
│           5. 算熵奖励 L_entropy                                     │
│           6. 反向传播，用 Adam 更新参数（lr = 3e-4）                 │
│           7. 梯度裁剪: grad_norm_clip = 1.0                        │
│                                                                     │
│   8 个 epoch × 16 个 mini-batch = 128 次参数更新/轮                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 第四阶段：学习率自适应调整                                            │
│                                                                     │
│   计算新旧策略的 KL 散度                                             │
│   if KL > kl_threshold (0.008):                                     │
│       降低学习率（策略变化太大，需要更保守）                            │
│   if KL < kl_threshold / 1.5:                                       │
│       提高学习率（策略变化太小，可以更激进）                            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    回到第一阶段，用更新后的策略
                    继续采集数据...

              总共重复直到达到 1,000,000 步 (timesteps)
```

#### 10.2.6 输入预处理

训练时，观测和价值都经过 **RunningStandardScaler**（滑动标准化）预处理：

```
对应配置:
  state_preprocessor: RunningStandardScaler
  value_preprocessor: RunningStandardScaler
```

它维护一个滑动的均值和方差，将输入标准化为均值 0、方差 1 的分布。这对训练稳定性很重要，因为观测的 1066 维中各分量的量级差异很大（位姿是米级，覆盖率是 0~1，LiDAR 也是 0~1）。

#### 10.2.7 超参数总表

| 超参数 | 值 | 含义 |
|--------|-----|------|
| γ (discount_factor) | 0.99 | 未来奖励折扣因子，越大越重视长期回报 |
| λ (GAE lambda) | 0.95 | GAE 平滑参数，越大越依赖多步回报 |
| ε (ratio_clip) | 0.2 | 概率比裁剪范围 [0.8, 1.2]，限制策略更新幅度 |
| learning_rate | 3e-4 | Adam 优化器初始学习率 |
| learning_epochs | 8 | 每轮 rollout 数据重复训练 8 遍 |
| mini_batches | 16 | 每个 epoch 分 16 个 mini-batch |
| rollouts | 64 | 每次采集 64 步数据 |
| entropy_loss_scale | 0.01 | 熵奖励权重，鼓励探索 |
| value_loss_scale | 2.0 | 价值损失权重 |
| grad_norm_clip | 1.0 | 梯度范数裁剪，防止梯度爆炸 |
| value_clip | 0.2 | 价值函数也做裁剪，稳定训练 |
| kl_threshold | 0.008 | KL 散度阈值，用于自适应调整学习率 |
| timesteps | 1,000,000 | 总训练步数 |
| seed | 42 | 随机种子，保证可复现 |

#### 10.2.8 训练过程中发生了什么（直觉版）

把整个训练过程想象成教一只小狗扫地：

1. **早期（0 ~ 10 万步）**：小狗乱跑（σ 大，动作随机），偶尔碰到新区域拿到覆盖奖励，经常撞墙被惩罚。策略网络开始学到"别撞墙"和"往没扫过的地方走"
2. **中期（10 ~ 50 万步）**：小狗学会了基本的避障和直线清扫，覆盖率能到 50%~70%，但经常重复走已经扫过的区域。里程碑奖励引导它继续提升
3. **后期（50 ~ 100 万步）**：小狗的策略趋于稳定（σ 小，动作确定），学会了高效的覆盖模式，能稳定达到 90%+ 的覆盖率。时间惩罚推动它加速完成

训练曲线的关键指标：
- **episode reward**（回合总奖励）：应该持续上升
- **coverage ratio**（覆盖率）：应该从 ~10% 逐渐提升到 90%+
- **policy entropy**（策略熵）：应该从高（随机探索）逐渐降低（确定策略）
- **value loss**（价值损失）：应该先升后降（先学粗略估计，再精细化）
