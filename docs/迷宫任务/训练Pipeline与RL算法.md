# 迷宫逃脱任务 — 训练 Pipeline 与 RL 算法

> 最后更新：2026-03-16

---

## 1. 任务概述

JetBot 在 8×8 DFS 迷宫中从随机位置出发，学习自主导航到出口。

| 项目 | 值 |
|------|----|
| Gym ID | `SmartCleaningRobot-MazeEscape-v0` |
| 环境类 | `MazeEscapeEnv` (`tasks/maze_escape/env.py`) |
| 配置类 | `MazeEscapeEnvCfg` (`tasks/maze_escape/env_cfg.py`) |
| 迷宫生成 | DFS，seed=42，与部署端迷宫完全一致 |
| 出口位置 | (0.0, -3.15)，位于迷宫南侧中央 |
| 判定逃出 | 距出口 < 0.3m |
| 最大时长 | 120 秒 |

---

## 2. 训练 Pipeline 全流程

### 2.1 一条命令启动训练

```bash
# 在项目根目录执行（需要先 conda activate isaaclab）
python train/scripts/common/train.py \
    --task SmartCleaningRobot-MazeEscape-v0 \
    --algorithm SAC \
    --num_envs 256 \
    --headless
```

### 2.2 Pipeline 各阶段详解

```
命令行参数解析
    │
    ▼
AppLauncher 启动 Isaac Sim
    │
    ▼
Hydra 加载任务配置
  ├── env_cfg  ← tasks/maze_escape/env_cfg.py (MazeEscapeEnvCfg)
  └── agent_cfg ← tasks/maze_escape/agents/sac.yaml
    │
    ▼
gym.make("SmartCleaningRobot-MazeEscape-v0")
  │  创建 MazeEscapeEnv 实例
  │  ├── 生成 DFS 迷宫几何体 (spawn_wall_scene)
  │  ├── 加载 JetBot USD 模型
  │  ├── 初始化 LiDAR RayCaster
  │  ├── 预计算 BFS 距离场
  │  └── 准备有效出生点列表
    │
    ▼
TrainingMetricLogger 包装
  │  拦截 step() 调用，记录每个 episode 的：
  │  reward, length, escaped → 写入 CSV
    │
    ▼
SkrlVecEnvWrapper 包装
  │  转换为 skrl 兼容的向量化环境接口
    │
    ▼
Runner(env, agent_cfg) 实例化
  │  ├── 根据 agent_cfg 创建 SAC Agent
  │  │   ├── Policy 网络 (GaussianMixin)
  │  │   ├── 2 个 Critic 网络
  │  │   └── 2 个 Target Critic 网络
  │  ├── 创建 Replay Buffer (50,000)
  │  └── 创建 SequentialTrainer
    │
    ▼
runner.run()  ← 训练主循环 (3,000,000 步)
  │  每一步：
  │  1. Agent 根据 policy 选动作
  │  2. 环境执行动作，返回 obs/reward/done
  │  3. 存入 replay buffer
  │  4. 从 buffer 采样 batch 更新网络
  │  5. 软更新 target critics
  │  6. (MetricLogger) 检测 episode 结束 → 记录指标
    │
    ▼
训练结束
  ├── 保存最终 checkpoint
  ├── MetricLogger 输出汇总统计
  └── 打印总训练时间
```

### 2.3 训练输出目录结构

```
train/logs/skrl/maze_escape/<时间戳>_sac_torch/
├── params/
│   ├── env.yaml              # 环境配置快照
│   └── agent.yaml            # SAC 超参快照
├── checkpoints/
│   ├── agent_300000.pt       # 每 N 步自动保存
│   ├── agent_600000.pt
│   ├── ...
│   └── best_agent.pt         # 最佳模型
├── training_metrics.csv      # 聚合指标（用于绘图）
├── training_episodes.csv     # 每个 episode 的详细记录
├── training_curves.png       # 由 plot_training.py 生成
└── events.out.tfevents.*     # TensorBoard 日志（skrl 自动写入）
```

### 2.4 查看训练曲线

训练产生的 CSV 日志可以直接用脚本绘图，无需 TensorBoard：

```bash
python train/scripts/common/plot_training.py \
    train/logs/skrl/maze_escape/<时间戳>_sac_torch/
```

生成的 `training_curves.png` 包含 4 个子图：

| 子图 | 含义 | 学习成功的表现 |
|------|------|----------------|
| Mean Episode Reward | 每个窗口内 episode 平均奖励 | 持续上升 |
| Mean Episode Length | 平均 episode 步数 | 持续下降（逃出更快） |
| Escape Success Rate | 窗口内逃出成功率 | 从 0% 上升 |
| Rolling Averages | 最近 200 个 episode 的滑动平均 | 奖励上升 + 逃出率上升 |

也可以通过 TensorBoard 查看（skrl 同时写入 tfevents）：

```bash
tensorboard --logdir train/logs/skrl/maze_escape
```

### 2.5 常用训练命令

```bash
# --- SAC (推荐，默认配置) ---
python train/scripts/common/train.py \
    --task SmartCleaningRobot-MazeEscape-v0 \
    --algorithm SAC \
    --num_envs 256 \
    --headless

# --- PPO (备选) ---
python train/scripts/common/train.py \
    --task SmartCleaningRobot-MazeEscape-v0 \
    --algorithm PPO \
    --num_envs 256 \
    --headless

# --- 短时测试（只跑 10 万步） ---
python train/scripts/common/train.py \
    --task SmartCleaningRobot-MazeEscape-v0 \
    --algorithm SAC \
    --num_envs 64 \
    --headless \
    --max_iterations 100000

# --- 从 checkpoint 恢复训练 ---
python train/scripts/common/train.py \
    --task SmartCleaningRobot-MazeEscape-v0 \
    --algorithm SAC \
    --num_envs 256 \
    --headless \
    --checkpoint train/logs/skrl/maze_escape/<run>/checkpoints/agent_1500000.pt

# --- 调整日志频率 ---
python train/scripts/common/train.py \
    --task SmartCleaningRobot-MazeEscape-v0 \
    --algorithm SAC \
    --headless \
    --log_interval 2000   # 每 2000 个 trainer step 记录一次
```

---

## 3. 观测空间（41 维，体坐标系）

所有观测都相对于机器人自身坐标系，不依赖全局位置。

```
索引      含义                    维度    范围              来源
────────────────────────────────────────────────────────────────
[0]       前进线速度              1       [-1, 1]           root_com_lin_vel_b[:, 0]
[1]       角速度                  1       [-1, 1]           root_com_ang_vel_b[:, 2]
[2:38]    LiDAR (360→36 rays)    36      [0, 1]            360 rays 每隔 10 取一条
                                                            归一化: dist / max_distance
[38]      出口方向 sin(θ)         1       [-1, 1]           体坐标系下出口角度
[39]      出口方向 cos(θ)         1       [-1, 1]           体坐标系下出口角度
[40]      出口距离（归一化）       1       [0, ~1]           欧氏距离 / maze_diagonal(5.7m)
```

**设计要点**：
- LiDAR 从 360 条射线降采样到 36 条，节省计算量同时保留空间感知
- 出口方向用 (sin θ, cos θ) 表示而非角度，避免 ±π 处的不连续
- 所有值都归一化到 [-1, 1] 或 [0, 1]，利于神经网络学习

---

## 4. 动作空间（2 维）

```
索引    含义           范围          物理量
─────────────────────────────────────────────
[0]     线速度指令     [-1, 1]      × max_linear_vel(0.3)  → [-0.3, 0.3] m/s
[1]     角速度指令     [-1, 1]      × max_angular_vel(2.0) → [-2.0, 2.0] rad/s
```

动作通过差速驱动模型转换为左右轮角速度：

```
left_wheel  = (v - ω × wheel_sep / 2) / wheel_radius
right_wheel = (v + ω × wheel_sep / 2) / wheel_radius

wheel_radius    = 0.0325 m
wheel_separation = 0.118 m
```

---

## 5. 奖励函数设计

```python
total_reward = (
    dist_reward          # 核心引导：BFS 距离缩减
    + exit_bonus         # 稀疏：逃出奖励
    + collision_penalty  # 惩罚碰撞
    + time_penalty       # 惩罚时间消耗
    + explore_bonus      # 鼓励探索新区域
    + forward_bonus      # 鼓励前进，抑制原地旋转
)
```

| 分量 | 权重 | 类型 | 计算方式 |
|------|------|------|----------|
| **BFS 距离缩减** | +5.0 | 密集 | `(prev_bfs - curr_bfs) / max_bfs` |
| **逃出奖励** | +200.0 | 稀疏 | 到达出口时触发 |
| **碰撞惩罚** | -5.0 | 密集 | LiDAR 最小距离 < 0.15m 时触发 |
| **时间惩罚** | -0.05 | 密集 | 每步固定扣分 |
| **探索奖励** | +0.5 | 密集 | 访问新的覆盖网格单元时触发 |
| **前进奖励** | +0.1 | 密集 | `clamp(forward_vel, min=0) × 0.1` |

### BFS 距离场（核心设计）

不使用欧氏距离，而是基于迷宫拓扑的 BFS 最短路径距离来引导 agent。这保证了：
- 沿正确路径移动总是获得正奖励，即使暂时远离出口
- 走错方向（如走入死胡同）会获得负奖励
- 不会被墙壁另一侧的"虚假近距离"误导

```
BFS 距离场示意（8×8 迷宫，数字 = 到出口的步数）：

  12  11  10   9   8   7   6   5
  13  12  11  10   9   8   7   6
  10   9   8   7   8   9  10   7
  11  10   7   6   7  10  11   8
   8   9   6   5   6   7   8   9
   7   8   5   4   5   6   7  10
   6   5   4   3   2   3   4   5
   5   4   3   2   1   2   3   4
                   ↑
                 出口 (距离=0)
```

---

## 6. SAC 算法配置详解

### 6.1 为什么选择 SAC

| 特性 | SAC | PPO |
|------|-----|-----|
| 数据效率 | 高（replay buffer 复用数据） | 低（on-policy，用完即弃） |
| 探索能力 | 强（最大熵原理，自动调节） | 中等（依赖 entropy bonus） |
| 连续动作 | 原生支持 | 需要高斯采样 |
| 适合场景 | 复杂导航、稀疏奖励 | 简单任务、离散动作 |
| 训练稳定性 | 较好（双 critic 减少过估计） | 依赖超参调节 |

迷宫任务的关键难点是**稀疏奖励**（只有到出口才有大奖励）和**长序列决策**（需要连续做出正确转弯），SAC 的最大熵探索和高数据效率更适合。

### 6.2 网络架构

```
Policy (Actor):
    输入: observations (41 维)
    ├── Linear(41, 256) + ELU
    ├── Linear(256, 128) + ELU
    ├── Linear(128, 64) + ELU
    └── 输出: mean + log_std → actions (2 维)
    类型: GaussianMixin (采样高斯分布)

Critic × 2 (双 Q 网络):
    输入: concat(observations, actions) = 43 维
    ├── Linear(43, 256) + ELU
    ├── Linear(256, 128) + ELU
    ├── Linear(128, 64) + ELU
    └── 输出: Q-value (1 维)
    类型: DeterministicMixin

Target Critic × 2:
    与 Critic 结构相同，通过 Polyak 软更新
```

### 6.3 关键超参数

```yaml
# --- 训练规模 ---
timesteps: 3,000,000       # 总 trainer step 数
                            # 实际 env 交互 = timesteps × num_envs
num_envs: 256 (默认 128)   # 并行环境数（CLI 可覆盖）

# --- SAC 核心参数 ---
discount_factor: 0.99       # γ，折扣因子
polyak: 0.005               # τ，target 网络软更新系数
                            # target_param = τ × param + (1-τ) × target_param
batch_size: 4096            # 每次从 replay buffer 采样的 batch 大小
memory_size: 50000          # Replay buffer 容量

# --- 学习率 ---
actor_learning_rate: 3e-4   # Policy 网络学习率
critic_learning_rate: 3e-4  # Critic 网络学习率
entropy_learning_rate: 3e-4 # 熵系数学习率

# --- 探索 ---
random_timesteps: 1000      # 前 1000 步纯随机动作（填充 buffer）
learning_starts: 1000       # 第 1000 步后开始学习
learn_entropy: True         # 自动调节熵系数（关键！）
initial_entropy_value: 1.0  # 初始熵权重

# --- 稳定性 ---
grad_norm_clip: 1.0         # 梯度裁剪
state_preprocessor: RunningStandardScaler  # 观测标准化
seed: 42                    # 随机种子
```

### 6.4 SAC 训练循环伪代码

```
初始化 policy π, critic Q1 Q2, target Q1' Q2', replay buffer D
初始化熵系数 α

for step = 1 to 3,000,000:
    if step < 1000:
        action = random()                    # 纯随机探索
    else:
        action = π.sample(observation)       # 从 policy 采样

    next_obs, reward, done = env.step(action)  # 256 个环境同步执行
    D.store(obs, action, reward, next_obs, done)

    if step >= 1000:
        batch = D.sample(4096)               # 从 buffer 采样

        # 更新 Critics
        with no_grad:
            next_action = π.sample(next_obs)
            target_Q = min(Q1'(next_obs, next_action),
                          Q2'(next_obs, next_action))
            target_Q -= α × log_prob(next_action)   # 熵正则化
            y = reward + γ × (1 - done) × target_Q

        loss_Q1 = MSE(Q1(obs, action), y)
        loss_Q2 = MSE(Q2(obs, action), y)
        优化 Q1, Q2

        # 更新 Policy
        new_action = π.sample(obs)
        loss_π = mean(α × log_prob(new_action) - min(Q1, Q2)(obs, new_action))
        优化 π

        # 更新熵系数
        loss_α = -α × mean(log_prob(new_action) + target_entropy)
        优化 α

        # 软更新 Target Critics
        Q1' = 0.005 × Q1 + 0.995 × Q1'
        Q2' = 0.005 × Q2 + 0.995 × Q2'
```

---

## 7. 物理仿真参数

| 参数 | 值 | 说明 |
|------|----|------|
| 仿真频率 | 120 Hz | `sim.dt = 1/120` |
| 控制频率 | 60 Hz | `decimation = 2`（每 2 个物理步执行一次控制） |
| 每 episode 最大步数 | 7200 | `120s × 60Hz` |
| GPU 物理引擎 | PhysX 5 | Isaac Sim 内置 |
| 环境间距 | 15.0 m | 防止相邻环境干扰 |

---

## 8. 迷宫几何

```
参数              值
──────────────────────
格局              8×8 逻辑单元（17×17 网格）
单元尺寸          0.45 m
墙高              0.3 m
生成算法          DFS（栈式深度优先搜索）
种子              42
起点              中心单元 (4, 4)
出口              南侧中央 (7, 4) → 世界坐标 (0.0, -3.15)
迷宫对角线        5.7 m
有效出生点        64 个可通行单元
```

---

## 9. 关键文件索引

```
train/
├── scripts/common/
│   ├── train.py              # 训练入口脚本
│   ├── play.py               # 推理/回放脚本
│   ├── metric_logger.py      # CSV 指标日志包装器
│   └── plot_training.py      # 训练曲线绘图工具
│
├── smartcleaningrobot/smartcleaningrobot/
│   ├── tasks/maze_escape/
│   │   ├── __init__.py       # gym.register() 注册
│   │   ├── env.py            # MazeEscapeEnv 环境实现
│   │   ├── env_cfg.py        # MazeEscapeEnvCfg 配置
│   │   └── agents/
│   │       ├── sac.yaml      # SAC 超参配置（推荐）
│   │       └── ppo.yaml      # PPO 超参配置（备选）
│   │
│   ├── scenes/builders/
│   │   ├── maze.py           # DFS 迷宫生成 + BFS 距离场 + 坐标转换
│   │   └── _common.py        # 墙体生成工具
│   │
│   └── robot_assets/
│       └── jetbot.py         # JetBot 配置 + 差速驱动模型
│
└── logs/skrl/maze_escape/    # 训练日志输出目录
```

---

## 10. 训练后续操作

### 导出模型（用于部署）

```bash
python deploy/scripts/export_model.py \
    --task maze \
    --checkpoint train/logs/skrl/maze_escape/<run>/checkpoints/agent_3000000.pt \
    --output deploy/models/maze_agent_scripted.pt
```

### Benchmark 对比测试

```bash
cd train/scripts/MazeDFSv1_Jetbot/benchmark/maze/

# 使用启发式 RL 策略（无需训练好的模型）
python run_benchmark.py --num_trials 10 --save_plots results/

# 使用训练好的 RL 模型
python run_benchmark.py \
    --rl_model deploy/models/maze_agent_scripted.pt \
    --num_trials 10 \
    --save_plots results/
```

### 回放训练好的策略

```bash
python train/scripts/common/play.py \
    --task SmartCleaningRobot-MazeEscape-v0 \
    --algorithm SAC \
    --num_envs 4 \
    --checkpoint train/logs/skrl/maze_escape/<run>/checkpoints/best_agent.pt
```
