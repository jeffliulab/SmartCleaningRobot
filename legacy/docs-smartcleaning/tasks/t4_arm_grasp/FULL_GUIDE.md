# T4: ArmGrasp — 完整指南

> 本文档是 T4 ArmGrasp（固定基座机械臂抓取）任务的自包含指南。
> 前置要求：[GETTING_STARTED.md](../../GETTING_STARTED.md)

**Gym ID:** `SmartCleaningRobot-ArmGrasp-v0`
**机器人:** TurtleBot4 + WidowX 250 (5-DOF 机械臂，底盘锁定)
**场景:** 简单平面（无家具/墙壁）
**推荐算法:** PPO

---

## 目录

1. [任务描述](#1-任务描述)
2. [背景知识](#2-背景知识)
3. [环境详解](#3-环境详解)
4. [奖励函数详解](#4-奖励函数详解)
5. [课程学习（Curriculum）](#5-课程学习curriculum)
6. [训练配置](#6-训练配置)
7. [训练步骤](#7-训练步骤)
8. [结果分析](#8-结果分析)
9. [常见问题](#9-常见问题)

---

## 1. 任务描述

### 1.1 目标

训练机械臂伸出、抓取一个目标物体（硬币），并将其抬起。底盘完全锁定不动——只训练手臂控制能力。

### 1.2 为什么先训练纯手臂？

这是 **课程学习 (Curriculum Learning)** 的思想：
- T4 先学会"伸手抓东西"（简单：6 维动作）
- T5 再加上"边走边抓"（复杂：8 维动作）

如果直接从 T5 开始，机器人需要同时学走路和抓取，难度太大。

### 1.3 核心挑战

| 挑战 | 说明 |
|------|------|
| 精确位置控制 | 5 个关节需要协调到达 4cm 精度内的目标 |
| 夹爪时机 | 必须在正确位置闭合夹爪 |
| 关节限制 | 每个关节有角度限制，不能无限旋转 |

---

## 2. 背景知识

### 2.1 机械臂运动学基础

WidowX 250 有 5 个旋转关节（自由度 = 5-DOF）：

```
            J1 (腰 Waist)
             │ ← 水平旋转 ±180°
             ▼
        J2 (肩 Shoulder)
             │ ← 前后摆动
             ▼
        J3 (肘 Elbow)
             │ ← 弯曲
             ▼
        J4 (前臂 Forearm Roll)
             │ ← 绕轴旋转
             ▼
        J5 (腕 Wrist Angle)
             │ ← 上下倾斜
             ▼
         ┌──────┐
         │ 夹爪  │ ← 开/合
         └──────┘
         End-Effector (EE)
```

### 2.2 关节限制

| 关节 | 名称 | 最小值 (rad) | 最大值 (rad) | 范围 (°) |
|------|------|-------------|-------------|---------|
| J1 | Waist | -3.14 | 3.14 | 360° |
| J2 | Shoulder | -1.88 | 1.99 | 222° |
| J3 | Elbow | -2.15 | 1.57 | 213° |
| J4 | Forearm Roll | -3.14 | 3.14 | 360° |
| J5 | Wrist Angle | -1.75 | 2.15 | 223° |

### 2.3 位置增量控制

本任务使用 **位置增量控制** 而非直接位置控制：

```
每一步:
  new_joint_pos = current_joint_pos + action × max_joint_delta
  new_joint_pos = clamp(new_joint_pos, joint_min, joint_max)

max_joint_delta = 0.05 rad ≈ 2.9°/步
```

**优势：** 动作空间 [-1, 1] 表示"往哪个方向转多少"，比直接指定角度更容易学习。

### 2.4 末端执行器 (End-Effector, EE)

EE 是机械臂末端的工作点（夹爪中心），它的 3D 位置由 5 个关节角度共同决定（正向运动学）。策略不直接控制 EE 位置，而是通过关节角度间接控制。

---

## 3. 环境详解

### 3.1 观测空间（21 维）

| 组件 | 维度 | 范围 | 说明 |
|------|------|------|------|
| ee_pos_rel | 3 | 连续 | EE 在机器人基座坐标系中的 XYZ 位置 |
| ee_quat | 4 | [-1,1] | EE 的四元数姿态 (w,x,y,z) |
| joint_pos_norm | 5 | [-1,1] | 5 个关节角度归一化到 [-1,1] |
| gripper_state | 1 | [0,1] | 夹爪开合程度 (0=关, 1=全开) |
| target_pos_rel | 3 | 连续 | 目标物体在基座坐标系中的 XYZ |
| target_dist | 1 | [0,1] | EE 到目标的欧氏距离 (截断到 1m) |
| contact | 1 | {0,1} | EE 是否在目标 4cm 范围内 |
| holding | 1 | {0,1} | 是否正在抓住物体 |
| arm_reach_norm | 1 | [0,1] | 臂伸展程度 (EE距肩 / MAX_REACH) |

**关节角度归一化公式：**
```
joint_pos_norm = 2 × (joint_pos - joint_min) / (joint_max - joint_min) - 1
```

### 3.2 动作空间（6 维）

| 索引 | 含义 | 范围 | 缩放 |
|------|------|------|------|
| 0-4 | 关节角度增量 | [-1,1] | ×0.05 rad/步 |
| 5 | 夹爪 | [-1,1] | >0.5 关闭, ≤0.5 打开 |

### 3.3 目标物体

- **类型：** 硬币 (coin)，最容易抓取的形状
- **出生高度：** 0.01m（地面上方 1cm）
- **位置：** 取决于课程阶段（见第 5 节）

### 3.4 Episode 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Episode 时长 | **15 秒** | 远短于其他任务（纯手臂，动作快） |
| 控制频率 | 60 Hz | |
| 最大步数 | ~900 | 15 × 60 |
| 成功终止 | 物体被抬起 5cm | |

---

## 4. 奖励函数详解

| 组件 | 权重 | 触发条件 | 设计意图 |
|------|------|---------|---------|
| reach | +2.0 | (prev_dist - curr_dist).clamp(min=0) | 引导 EE 接近目标 |
| contact | +1.0 | EE 在目标 4cm 内 | 奖励接触 |
| grasp_success | +20.0 | 首次成功抓取（一次性） | 抓取是核心目标 |
| lift_success | +10.0 | 物体抬起 5cm 以上（一次性） | 奖励完整的抬起动作 |
| time_penalty | -0.01 | 每步 | 鼓励快速完成 |
| drop_penalty | -5.0 | 曾持有物体但掉落 | 惩罚意外松手 |

**奖励梯度：** reach(密集) → contact(密集) → grasp(稀疏) → lift(稀疏)

这种设计形成了清晰的学习路径：先学会靠近，再学会接触，再学会抓取，最后学会抬起。

### 抓取判定条件

- **contact:** EE 距目标 < 0.04m (4cm)
- **holding:** contact 为真 **且** 夹爪关节位置 < 0.005 rad（几乎完全闭合）
- **lift:** holding 为真 **且** 目标 Z 坐标 > spawn_height + 0.05m

---

## 5. 课程学习（Curriculum）

### 5.1 三个阶段

| 阶段 | 目标位置 | 手臂初始姿态 | 目标 | 训练步数 |
|------|---------|-------------|------|---------|
| **Stage 1** | 固定：正前方 0.15m | 默认 | >90% 到达目标 | 2M |
| **Stage 2** | 随机：0.05-0.30m 前方 ± 0.10m 侧向 | 默认 | >70% 抓取 | +2M |
| **Stage 3** | 随机（同 Stage 2） | 随机 ±20% | 全面鲁棒性 | +2M |

### 5.2 如何切换阶段

修改 `env_cfg.py` 中的 `curriculum_stage` 字段：

```python
curriculum_stage: int = 1  # 改为 2 或 3
```

### 5.3 阶段之间的迁移

```bash
# Stage 1: 从零开始
python train/scripts/common/train.py \
  --task SmartCleaningRobot-ArmGrasp-v0 --algorithm PPO --headless

# Stage 2: 从 Stage 1 检查点继续
# 先修改 env_cfg.py: curriculum_stage = 2
python train/scripts/common/train.py \
  --task SmartCleaningRobot-ArmGrasp-v0 --algorithm PPO --headless \
  --checkpoint train/logs/skrl/arm_grasp/<stage1_run>/checkpoints/best_agent.pt

# Stage 3: 从 Stage 2 检查点继续
# 先修改 env_cfg.py: curriculum_stage = 3
python train/scripts/common/train.py \
  --task SmartCleaningRobot-ArmGrasp-v0 --algorithm PPO --headless \
  --checkpoint train/logs/skrl/arm_grasp/<stage2_run>/checkpoints/best_agent.pt
```

---

## 6. 训练配置

### 6.1 PPO 超参数

文件：`train/.../tasks/arm_grasp/agents/ppo.yaml`

| 参数 | 值 | 说明 |
|------|-----|------|
| 网络结构 | [256, 128, 64] | |
| Rollout 长度 | 128 | 短 episode → 长 rollout |
| 学习 epoch | 8 | |
| Mini batches | 16 | |
| 学习率 | 3.0e-4 | |
| 折扣因子 (γ) | **0.995** | 比其他任务略高（延迟 lift 奖励） |
| GAE λ | 0.95 | |
| 裁剪比率 | 0.2 | |
| 熵损失权重 | 0.01 | |
| Time limit bootstrap | False | |
| 总训练步数 | **2,000,000** per stage | |
| 并行环境数 | **128** | 小场景可以多并行 |

### 6.2 环境参数

| 参数 | 值 |
|------|-----|
| Episode 时长 | 15 秒 |
| 环境间距 | 3.0m（无场景，只有机器人和物体） |
| 抓取阈值 | 0.04m |
| 抬起高度 | 0.05m |
| 关节增量上限 | 0.05 rad/步 |

---

## 7. 训练步骤

### 7.1 验证

```bash
python train/scripts/common/list_envs.py --keyword ArmGrasp

python train/scripts/common/zero_agent.py \
  --task SmartCleaningRobot-ArmGrasp-v0 --num_envs 1
```

### 7.2 Stage 1 训练

```bash
# 确认 env_cfg.py 中 curriculum_stage = 1
python train/scripts/common/train.py \
  --task SmartCleaningRobot-ArmGrasp-v0 \
  --algorithm PPO \
  --num_envs 128 \
  --headless \
  --seed 42
```

### 7.3 分析与回放

```bash
python train/scripts/common/plot_training.py \
  train/logs/skrl/arm_grasp/<run_dir> --smooth 10

python train/scripts/common/play.py \
  --task SmartCleaningRobot-ArmGrasp-v0 \
  --num_envs 1 --video
```

---

## 8. 结果分析

### 8.1 预期训练时间线（Stage 1）

| 阶段 | 步数 | 行为 |
|------|------|------|
| 0 - 200K | 手臂随机摇摆 |
| 200K - 800K | 学会向前伸展 |
| 800K - 1.5M | 学会接触目标并闭合夹爪 |
| 1.5M - 2M | 抓取+抬起成功率 >80% |

### 8.2 成功标准

| 阶段 | 指标 | 目标 |
|------|------|------|
| Stage 1 | 到达目标 (contact) | > 90% |
| Stage 2 | 抓取成功 | > 70% |
| Stage 3 | 随机初始姿态下抓取 | > 60% |

### 8.3 回放观察要点

| 行为 | 好的表现 | 需改进 |
|------|---------|--------|
| 到达 | 手臂平滑伸出到目标上方 | 抖动或绕远路 |
| 降低 | EE 精准下降到物体位置 | 偏移或碰到地面 |
| 夹取 | 夹爪在正确时机闭合 | 太早/太晚闭合 |
| 抬起 | 稳定抬起不掉落 | 抬起后物体滑落 |

---

## 9. 常见问题

### Q1: 手臂完全不动

**原因：** 关节增量太小或动作输出被截断。
**排查：** 检查 `max_joint_delta`（默认 0.05 rad）是否合理。

### Q2: 总是从侧面接近而不是从上方

**正常现象：** 策略可能找到了非直觉但有效的抓取角度。只要抓取成功率高就不必担心。

### Q3: Stage 2 从 Stage 1 检查点加载后性能下降

**正常现象：** 随机化目标位置后需要重新适应，通常 200-500K 步后会恢复并超过 Stage 1。

### Q4: gripper 始终不闭合

**排查：**
- 检查 `gripper_threshold`（默认 0.5）
- 打印 action[5] 的值，确认策略有输出 > 0.5 的值
- 增大 `rew_grasp_success` 给予更强激励

### Q5: 如何从 T4 过渡到 T5？

T4 训练好的手臂策略权重可以作为 T5 的初始化：
1. 训练 T4 Stage 2 到收敛
2. 将 best_agent.pt 作为 T5 的 `--checkpoint` 参数
3. T5 会在此基础上加入底盘控制学习

---

## 参考文件

| 文件 | 路径 |
|------|------|
| 环境实现 | `train/.../tasks/arm_grasp/env.py` |
| 环境配置 | `train/.../tasks/arm_grasp/env_cfg.py` |
| PPO 超参 | `train/.../tasks/arm_grasp/agents/ppo.yaml` |
| 机器人资产 | `train/.../robot_assets/turtlebot4_with_arm.py` |
| 物体资产 | `train/.../small_objects/object_assets.py` |
