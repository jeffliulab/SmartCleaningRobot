# SmartCleaningRobot — 通用训练、评估与结果分析指南

> 本文档适用于所有 7 个任务（T1-T7），介绍从训练启动到结果分析的完整流程。
> 阅读本文前，请确保已完成 [GETTING_STARTED.md](GETTING_STARTED.md) 中的环境搭建。

---

## 目录

1. [训练流水线总览](#1-训练流水线总览)
2. [RL 基础知识（5分钟速通）](#2-rl-基础知识5分钟速通)
3. [PPO 算法详解](#3-ppo-算法详解)
4. [SAC 算法详解](#4-sac-算法详解)
5. [训练命令与参数详解](#5-训练命令与参数详解)
6. [训练日志结构](#6-训练日志结构)
7. [CSV 指标详解](#7-csv-指标详解)
8. [训练曲线解读](#8-训练曲线解读)
9. [评估与回放](#9-评估与回放)
10. [可视化工具](#10-可视化工具)
11. [TensorBoard 使用](#11-tensorboard-使用)
12. [训练调参指南](#12-训练调参指南)
13. [常见问题与排障](#13-常见问题与排障)

---

## 1. 训练流水线总览

```
                ┌─────────────────────────────────────────────┐
                │           完整训练流水线                      │
                └─────────────────────────────────────────────┘

 [1. 验证环境]     [2. 冒烟测试]     [3. 正式训练]     [4. 评估]     [5. 分析]
 list_envs.py  →  zero/random_agent →  train.py     →  play.py  →  plot_training.py
                                          │                            │
                                          ▼                            ▼
                                    logs/skrl/task/               training_curves.png
                                    ├── checkpoints/              training_metrics.csv
                                    ├── training_metrics.csv
                                    ├── training_episodes.csv
                                    └── params/
```

**建议的工作顺序：**

```bash
# 1. 验证所有环境已注册
python train/scripts/common/list_envs.py

# 2. 零动作测试（确认场景加载正常）
python train/scripts/common/zero_agent.py --task <TASK_ID> --num_envs 1

# 3. 随机动作测试（确认动力学正常）
python train/scripts/common/random_agent.py --task <TASK_ID> --num_envs 1 --headless

# 4. 正式训练
python train/scripts/common/train.py --task <TASK_ID> --algorithm PPO --num_envs 64 --headless

# 5. 绘制训练曲线
python train/scripts/common/plot_training.py train/logs/skrl/<task>/<run_dir>

# 6. 回放最佳策略
python train/scripts/common/play.py --task <TASK_ID> --num_envs 1 --video
```

---

## 2. RL 基础知识（5分钟速通）

### 2.1 什么是强化学习？

强化学习 (Reinforcement Learning, RL) 是让 **智能体 (Agent)** 通过与 **环境 (Environment)** 交互来学习最优行为的方法。

```
    ┌──────────┐    观测 (observation)     ┌──────────────┐
    │          │ ◄──────────────────────── │              │
    │  智能体   │                           │    环境       │
    │ (Policy) │ ────────────────────────► │ (Simulator)  │
    │          │    动作 (action)           │              │
    └──────────┘                           └──────────────┘
         ▲                                        │
         │              奖励 (reward)              │
         └────────────────────────────────────────┘
```

- **观测 (Observation):** 智能体能"看到"的信息（如 LiDAR 距离、速度、目标方向）
- **动作 (Action):** 智能体的输出（如轮子速度、机械臂关节角度）
- **奖励 (Reward):** 告诉智能体做得好不好的数字信号
- **策略 (Policy):** 从观测到动作的映射函数，由神经网络实现

### 2.2 关键概念

| 概念 | 含义 | 类比 |
|------|------|------|
| **Episode（回合）** | 一次完整的任务尝试，从开始到成功/超时 | 游戏的一局 |
| **Step（步）** | 一次 观测→动作→奖励 的循环 | 游戏的一帧 |
| **Reward（奖励）** | 每一步获得的反馈信号 | 分数 |
| **Return（回报）** | 一个 Episode 中所有奖励的（折扣）总和 | 总分 |
| **Discount Factor (γ)** | 未来奖励的衰减系数，通常 0.99 | "看得多远" |
| **Value Function V(s)** | 从状态 s 开始，预期能获得的总回报 | "这个位置值多少分" |
| **Advantage A(s,a)** | 采取动作 a 比平均好多少：A = Q(s,a) - V(s) | "这个动作比平均好多少" |

### 2.3 本项目的神经网络

所有任务使用 **MLP（多层感知机）** 作为策略网络：

```
输入层              隐藏层                          输出层
(观测向量)    (全连接 + ELU 激活)               (动作均值)

 obs_dim  →  [hidden_1] → [hidden_2] → [hidden_3] →  act_dim
              + ELU         + ELU         + ELU

例如 T1 MazeEscape (SAC):
 41-dim   →  [256] → [128] → [64] → 2-dim (线速度, 角速度)
```

**为什么不用 CNN/Transformer？**
- 输入是一维向量（不是图像），MLP 已经足够
- MLP 推理速度快，适合 GPU 并行仿真中的实时控制

---

## 3. PPO 算法详解

**PPO (Proximal Policy Optimization)** 是目前最流行的 RL 算法之一，大多数任务的默认选择。

### 3.1 核心思想

PPO 每次更新都保证策略不会变化太大（"近端"约束），避免训练不稳定。

### 3.2 训练循环

```
每次迭代:
  1. [Rollout] 用当前策略收集 N 步经验 (观测, 动作, 奖励)
  2. [Advantage] 用 GAE 计算每步的优势函数 A(s,a)
  3. [Update] 用收集的数据更新策略网络（重复 K 个 epoch）
  4. [Adapt] 根据 KL 散度自适应调整学习率
```

### 3.3 关键超参数

| 参数 | 典型值 | 含义 | 调参建议 |
|------|--------|------|----------|
| `rollouts` | 64-128 | 每次更新前收集多少步 | 越大越稳定，但更慢 |
| `learning_epochs` | 8 | 每批数据重复学习几次 | 过高会过拟合 |
| `mini_batches` | 8-16 | 将数据分成几份 | batch_size = rollouts×num_envs / mini_batches |
| `discount_factor (γ)` | 0.99 | 未来奖励衰减 | 长期任务用高值 |
| `lambda (λ)` | 0.95 | GAE 平滑系数 | 高值=高方差低偏差 |
| `learning_rate` | 1e-4 ~ 3e-4 | 步长大小 | 过大震荡，过小太慢 |
| `ratio_clip (ε)` | 0.2 | PPO 裁剪范围 | 几乎不需要调 |
| `entropy_loss_scale` | 0.01-0.02 | 探索力度 | 高=更多探索 |
| `grad_norm_clip` | 1.0 | 梯度裁剪 | 防止梯度爆炸 |

### 3.4 损失函数

```
总损失 = L_clip (策略损失) - c_vf × L_value (值函数损失) + c_ent × L_entropy (熵奖励)
```

- **L_clip:** 鼓励策略向优势方向更新，但裁剪更新幅度
- **L_value:** 让值函数准确预测回报
- **L_entropy:** 鼓励探索（防止策略过早收敛）

### 3.5 适用场景

| 场景 | 推荐 |
|------|------|
| 高维连续动作空间 | PPO |
| 大规模并行环境 | PPO（on-policy，天然并行） |
| 初学者 / 调参少 | PPO（对超参不敏感） |
| 需要高样本效率 | 用 SAC 代替 |

---

## 4. SAC 算法详解

**SAC (Soft Actor-Critic)** 是 off-policy 算法，样本效率更高，推荐用于 T1 MazeEscape。

### 4.1 核心思想

SAC 最大化 **回报 + 熵**，在利用已有知识的同时保持探索。

### 4.2 网络结构（5个网络）

```
┌─────────────────────────────────┐
│  Policy (Actor)                 │  输出: 动作的高斯分布 (μ, σ)
├─────────────────────────────────┤
│  Critic 1 (Q-function)          │  输入: [观测, 动作] → Q值
├─────────────────────────────────┤
│  Critic 2 (Q-function)          │  输入: [观测, 动作] → Q值（双 Q 防过估计）
├─────────────────────────────────┤
│  Target Critic 1                │  Critic 1 的滞后拷贝（软更新 τ=0.005）
├─────────────────────────────────┤
│  Target Critic 2                │  Critic 2 的滞后拷贝
└─────────────────────────────────┘
```

### 4.3 关键超参数

| 参数 | 典型值 | 含义 |
|------|--------|------|
| `memory_size` | 500,000 | 经验回放缓冲区大小 |
| `batch_size` | 4096 | 每次采样多少条经验 |
| `polyak (τ)` | 0.005 | 目标网络软更新速率 |
| `actor_learning_rate` | 3e-4 | Actor 学习率 |
| `critic_learning_rate` | 3e-4 | Critic 学习率 |
| `learn_entropy` | True | 自动调整温度参数 |
| `random_timesteps` | 1000 | 开始学习前纯随机探索步数 |
| `learning_starts` | 1000 | 缓冲区最少多少条才开始学习 |

### 4.4 适用场景

| 场景 | 推荐 |
|------|------|
| 稀疏奖励（如迷宫出口） | SAC（经验回放能复用稀疏成功） |
| 需要高样本效率 | SAC（off-policy 重复利用数据） |
| 小批量并行环境 | SAC |
| 超大规模并行（>128 envs） | PPO 更合适 |

---

## 5. 训练命令与参数详解

### 5.1 基本命令

```bash
python train/scripts/common/train.py --task <TASK_ID> [选项]
```

### 5.2 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--task` | str | **必填** | 任务 ID，如 `SmartCleaningRobot-MazeEscape-v0` |
| `--algorithm` | str | PPO | 算法：`PPO` / `SAC` / `TD3` |
| `--num_envs` | int | 任务默认 | 并行环境数量 |
| `--headless` | flag | False | 无 GUI 模式（训练时推荐开启） |
| `--seed` | int | 42 | 随机种子（影响可复现性） |
| `--checkpoint` | str | None | 从检查点恢复训练 |
| `--max_iterations` | int | YAML定义 | 最大迭代次数 |
| `--device` | str | cuda:0 | 计算设备 |
| `--video` | flag | False | 是否录制训练视频 |
| `--video_interval` | int | 2000 | 每隔多少步录一次 |
| `--video_length` | int | 200 | 每段视频帧数 |
| `--log_interval` | int | 5000 | 指标日志记录间隔（trainer steps） |
| `--distributed` | flag | False | 多 GPU 训练 |
| `--ml_framework` | str | torch | ML 框架 |

### 5.3 各任务推荐命令

```bash
# T1 MazeEscape — 推荐 SAC
python train/scripts/common/train.py \
  --task SmartCleaningRobot-MazeEscape-v0 \
  --algorithm SAC --num_envs 128 --headless

# T2 Coverage — PPO
python train/scripts/common/train.py \
  --task SmartCleaningRobot-Coverage-v0 \
  --algorithm PPO --num_envs 64 --headless

# T4 ArmGrasp — PPO
python train/scripts/common/train.py \
  --task SmartCleaningRobot-ArmGrasp-v0 \
  --algorithm PPO --num_envs 128 --headless

# T5 ObjectPickup — PPO
python train/scripts/common/train.py \
  --task SmartCleaningRobot-ObjectPickup-v0 \
  --algorithm PPO --num_envs 64 --headless

# T6 CoverageAvoid — PPO
python train/scripts/common/train.py \
  --task SmartCleaningRobot-CoverageAvoid-v0 \
  --algorithm PPO --num_envs 128 --headless

# T7 CoveragePickup — PPO (低学习率微调)
python train/scripts/common/train.py \
  --task SmartCleaningRobot-CoveragePickup-v0 \
  --algorithm PPO --num_envs 32 --headless
```

---

## 6. 训练日志结构

每次训练会在 `train/logs/skrl/` 下创建一个目录：

```
train/logs/skrl/
└── <task_name>/
    └── <timestamp>_<algorithm>_<ml_framework>/
        ├── params/
        │   ├── env.yaml            ← 环境配置快照
        │   └── agent.yaml          ← 算法超参数快照
        ├── checkpoints/
        │   ├── agent_1000.pt       ← 定期保存的模型
        │   ├── agent_2000.pt
        │   ├── ...
        │   └── best_agent.pt       ← 最佳模型
        ├── training_metrics.csv    ← 聚合统计（每 log_interval 步）
        ├── training_episodes.csv   ← 每个 episode 的详细数据
        ├── events.out.tfevents.*   ← TensorBoard 数据
        └── videos/                 ← 录制的视频（如果开启）
            └── train/
```

---

## 7. CSV 指标详解

### 7.1 training_metrics.csv（聚合指标）

每 `log_interval` 步（默认 5000 步）写入一行：

| 列名 | 含义 | 怎么看 |
|------|------|--------|
| `timestep` | 当前总步数（trainer_steps × num_envs） | X 轴 |
| `wall_time_s` | 训练已耗时（秒） | 估算剩余时间 |
| `total_episodes` | 累计完成的 episode 总数 | 训练进度 |
| `window_episodes` | 本窗口内完成的 episode 数 | 采样量 |
| `mean_reward` | 本窗口 episode 平均奖励 | **核心指标**：应该上升 |
| `std_reward` | 奖励标准差 | 越小越稳定 |
| `min_reward` / `max_reward` | 奖励范围 | 观察极端值 |
| `mean_ep_length` | 平均 episode 长度（步） | 应该下降（更快完成） |
| `std_ep_length` | 长度标准差 | 越小越一致 |
| `escape_rate` | 成功率（仅限有终止条件的任务） | **核心指标**：应趋近 1.0 |
| `escape_count` | 本窗口成功次数 | 绝对值 |
| `rolling_mean_reward` | 最近 200 个 episode 的滚动平均奖励 | 平滑趋势 |
| `rolling_escape_rate` | 最近 200 个 episode 的滚动成功率 | 平滑趋势 |

### 7.2 training_episodes.csv（逐 Episode 数据）

每完成一个 episode 写入一行：

| 列名 | 含义 |
|------|------|
| `timestep` | episode 结束时的全局步数 |
| `wall_time_s` | 结束时刻 |
| `episode_id` | 全局 episode 编号（从 0 开始） |
| `reward` | 该 episode 的累计奖励 |
| `length` | 该 episode 的步数 |
| `escaped` | 1 = 成功（terminated），0 = 超时（truncated） |

---

## 8. 训练曲线解读

### 8.1 什么是"好的"训练曲线？

**正常训练的四个阶段：**

```
奖励
 ▲
 │                         ┌──────── 4. 收敛
 │                    ┌────┘
 │               ┌────┘          3. 快速上升
 │          ┌────┘
 │     ┌────┘                 2. 缓慢探索
 │─────┘
 │ 1. 随机探索
 └──────────────────────────────────► 步数
```

1. **随机探索期**（0-10%）：奖励很低且随机波动，这是正常的
2. **缓慢探索期**（10-30%）：奖励缓慢上升，智能体开始学到基本行为
3. **快速上升期**（30-70%）：奖励快速上升，这是最关键的阶段
4. **收敛期**（70-100%）：奖励趋于稳定，接近最优

### 8.2 各任务的预期指标

| 任务 | 训练步数 | 收敛奖励 | 成功率目标 | 预计时间 (RTX 4070) |
|------|---------|---------|-----------|-------------------|
| T1 MazeEscape (SAC) | 3M | ~150+ | >80% escape | 4-8 小时 |
| T1 MazeEscape (PPO) | 2M | ~100+ | >60% escape | 3-6 小时 |
| T2 Coverage | 5M | ~200+ | >60% coverage | 6-12 小时 |
| T4 ArmGrasp | 3M | ~50+ | >70% grasp | 4-8 小时 |
| T5 ObjectPickup | 5M | ~100+ | >60% single pickup | 8-16 小时 |
| T6 CoverageAvoid | 3M | ~150+ | >80% coverage | 4-8 小时 |
| T7 CoveragePickup | 8M | ~200+ | >70% cov 或 3+ objects | 12-24 小时 |

> **注意：** 以上时间为估算值，实际取决于 GPU 型号、并行环境数、散热等因素。

### 8.3 常见异常曲线及诊断

| 曲线形状 | 可能原因 | 解决方案 |
|----------|---------|----------|
| 始终平坦（不上升） | 奖励设计有问题 / 学习率太小 | 检查奖励函数、增大 LR |
| 剧烈震荡 | 学习率过大 / batch 太小 | 减小 LR、增大 mini_batches |
| 上升后突然崩溃 | PPO 更新步长过大 | 减小 ratio_clip 或 LR |
| 缓慢上升但不收敛 | 探索不足 | 增大 entropy_loss_scale |
| 快速收敛到次优值 | 过早停止探索 | 增大 entropy_loss_scale、增加训练步数 |

---

## 9. 评估与回放

### 9.1 回放训练好的模型

```bash
python train/scripts/common/play.py \
  --task <TASK_ID> \
  --algorithm <PPO/SAC> \
  --checkpoint train/logs/skrl/<task>/<run>/checkpoints/best_agent.pt \
  --num_envs 1 \
  --video \
  --video_length 500
```

### 9.2 参数说明

| 参数 | 说明 |
|------|------|
| `--checkpoint` | 如不指定，自动寻找最新的检查点 |
| `--num_envs` | 回放时建议用 1-4 个环境，方便观察 |
| `--video` | 录制视频到 `videos/play/` 目录 |
| `--real-time` | 以实时速度运行（不加则全速） |

### 9.3 评估指标

回放时观察：
- 机器人是否能完成任务（到达出口 / 覆盖地面 / 抓取物体）
- 行为是否自然流畅（不抖动、不原地打转）
- 是否有碰撞墙壁或物体
- 完成速度是否合理

---

## 10. 可视化工具

### 10.1 plot_training.py

```bash
python train/scripts/common/plot_training.py <log_dir> [--smooth N] [--save_only]
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `log_dir` | **必填** | 训练日志目录路径 |
| `--smooth` | 5 | 移动平均窗口大小（越大越平滑） |
| `--save_only` | False | 只保存 PNG 不显示窗口 |

**输出：** `training_curves.png`（2×2 子图）

```
┌─────────────────────────────┬─────────────────────────────┐
│  Mean Episode Reward         │  Mean Episode Length         │
│  (平均奖励，应上升)           │  (平均步数，应下降)           │
│  蓝线 = 原始数据              │                              │
│  橙线 = 平滑后                │                              │
│  阴影 = 标准差范围            │                              │
├─────────────────────────────┼─────────────────────────────┤
│  Escape Success Rate (%)     │  Rolling Averages            │
│  (成功率，应趋近 100%)        │  (200-episode 滚动平均)       │
│                              │  左轴: 奖励, 右轴: 成功率     │
└─────────────────────────────┴─────────────────────────────┘
```

### 10.2 使用示例

```bash
# 默认平滑
python train/scripts/common/plot_training.py \
  train/logs/skrl/maze_escape/2026-03-16_10-00-00_sac_torch

# 大窗口平滑（适合长训练）
python train/scripts/common/plot_training.py \
  train/logs/skrl/maze_escape/2026-03-16_10-00-00_sac_torch \
  --smooth 20

# 无 GUI 环境下保存
python train/scripts/common/plot_training.py <log_dir> --save_only
```

---

## 11. TensorBoard 使用

```bash
# 启动 TensorBoard
tensorboard --logdir train/logs/skrl/<task_name>

# 在浏览器打开 http://localhost:6006
```

**关键面板：**
- **Scalars:** 奖励、损失、学习率等曲线
- **Custom Scalars:** SKRL 框架记录的额外指标

**多次训练对比：**
```bash
# 对比不同算法
tensorboard --logdir_spec \
  SAC:train/logs/skrl/maze_escape/run1_sac_torch,\
  PPO:train/logs/skrl/maze_escape/run2_ppo_torch
```

---

## 12. 训练调参指南

### 12.1 从哪里开始？

**建议顺序（从影响最大到最小）：**

1. **num_envs** — 并行环境数量（影响训练速度和稳定性）
2. **learning_rate** — 学习率（影响收敛速度和稳定性）
3. **network size** — 网络大小（影响表达能力）
4. **rollouts / batch_size** — 采样量（影响更新质量）
5. **entropy_loss_scale** — 探索系数（影响探索-利用平衡）

### 12.2 如何修改超参数？

超参数定义在各任务的 YAML 文件中：

```
train/smartcleaningrobot/smartcleaningrobot/tasks/<task_name>/agents/
├── ppo.yaml    ← PPO 配置
└── sac.yaml    ← SAC 配置（部分任务）
```

修改 YAML 文件后重新训练即生效。

### 12.3 实用调参技巧

| 情况 | 调整方案 |
|------|---------|
| 训练太慢 | 增加 `num_envs`，开启 `--headless` |
| GPU 显存不足 | 减少 `num_envs`，减小网络层数 |
| 奖励不上升 | 检查奖励函数，增大 `learning_rate` |
| 训练不稳定 | 减小 `learning_rate`，增大 `mini_batches` |
| 收敛到次优 | 增大 `entropy_loss_scale`，增加训练步数 |
| SAC 学不动 | 增大 `memory_size` 和 `random_timesteps` |

---

## 13. 常见问题与排障

### Q1: 训练中途报错 `NaN detected in rewards`

**原因：** 物理仿真中机器人穿墙或关节超限。
**解决：**
- 降低 `max_linear_vel` 或 `max_angular_vel`
- 增大 `decimation`（降低控制频率）

### Q2: 奖励始终为负且不变

**原因：** 策略停在原地不动（局部最优）。
**解决：**
- 增大 `entropy_loss_scale`（0.01 → 0.05）
- 添加前进奖励（forward_bonus）

### Q3: 训练一段时间后奖励突然下降

**原因：** 学习率过大导致策略崩溃。
**解决：**
- 使用 `KLAdaptiveLR` 调度器（默认已启用）
- 减小初始学习率

### Q4: GPU 利用率很低（<30%）

**原因：** 并行环境数量不足或 CPU 瓶颈。
**解决：**
- 增加 `--num_envs`（128 或 256）
- 确认使用了 `--headless`

### Q5: 如何恢复中断的训练？

```bash
python train/scripts/common/train.py \
  --task <TASK_ID> \
  --algorithm <ALG> \
  --checkpoint train/logs/skrl/<task>/<run>/checkpoints/agent_<step>.pt \
  --num_envs 64 --headless
```

### Q6: 两次训练结果差异很大

**原因：** RL 训练天然具有高方差。
**解决：**
- 固定 `--seed`（默认 42）保证可复现
- 多跑几次取平均，报告均值和标准差

---

## 下一步

- 阅读 [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md) 了解如何整理训练结果为报告
- 阅读具体任务的 FULL_GUIDE.md 获取任务专属知识和参数
