# T7: CoveragePickup — 完整指南（端到端任务）

> 本文档是 T7 CoveragePickup 的自包含指南。这是本项目的**最终任务**，整合了所有前置能力。
> 前置要求：[GETTING_STARTED.md](../../GETTING_STARTED.md)，强烈建议先完成 T5 和 T6。

**Gym ID:** `SmartCleaningRobot-CoveragePickup-v0`
**机器人:** TurtleBot4 + WidowX 250 (5-DOF 机械臂)
**场景:** 房间 + 10 个物体（6 个可拾取 + 4 个需避开）
**推荐算法:** PPO（从 T5+T6 检查点微调）

---

## 目录

1. [任务描述](#1-任务描述)
2. [背景知识：有限状态机 (FSM)](#2-背景知识有限状态机-fsm)
3. [环境详解](#3-环境详解)
4. [奖励函数详解](#4-奖励函数详解)
5. [训练配置](#5-训练配置)
6. [训练步骤](#6-训练步骤)
7. [结果分析](#7-结果分析)
8. [常见问题](#8-常见问题)

---

## 1. 任务描述

### 1.1 目标

模拟完整的智能清扫机器人：在房间中覆盖清扫地面，遇到小物体（硬币、发卡等）自动拾取放入篮子，遇到大物体（粗线缆、大橡皮）自动绕开。

### 1.2 为什么是"端到端"？

这个任务整合了之前所有技能：

```
T6 CoverageAvoid ──┐
  (底盘导航+避障)    ├──► T7 CoveragePickup（全部整合）
T5 ObjectPickup ───┘
  (移动操控+抓取)
```

### 1.3 物体分类

| 类别 | 物体 | 数量 | 行为 |
|------|------|------|------|
| **可拾取** | 硬币×3, 发卡×1, 橡皮×1, 硬币×1 | 6 | 抓取→放入篮子 |
| **需避开** | 橡皮×2, 数据线×2 | 4 | 绕行避开 |

---

## 2. 背景知识：有限状态机 (FSM)

### 2.1 什么是 FSM？

FSM（Finite State Machine）是一种简单的行为控制模式——机器人在任意时刻只处于一个状态，根据条件自动切换。

### 2.2 本任务的三个状态

```
                   发现可拾取物体 (距离 < 1.5m)
        ┌─────────────────────────────────────────┐
        │                                         ▼
   ┌─────────┐                              ┌──────────┐
   │  CLEAN   │                              │ APPROACH  │
   │ 覆盖清扫  │                              │ 接近物体   │
   │ 手臂收起  │                              │ 手臂准备   │
   └─────────┘                              └──────────┘
        ▲                                         │
        │         放置完成 / 所有物体已拾取          │  EE 进入抓取范围 (< 0.04m)
        │                                         ▼
        │                                    ┌─────────┐
        └────────────────────────────────────│ PICKUP   │
                                             │ 抓取放置  │
                                             │ 手臂活跃  │
                                             └─────────┘
```

### 2.3 状态转换规则

| 转换 | 条件 |
|------|------|
| CLEAN → APPROACH | 任何可拾取 FREE 物体距机器人 < 1.5m |
| APPROACH → PICKUP | EE 距任何可拾取 FREE 物体 < 0.04m |
| PICKUP → CLEAN | 物体放入篮子且 EE 不在抓取范围，或所有物体已拾取 |

### 2.4 动作门控

FSM 决定策略输出的哪些部分生效：

| 状态 | 轮子 | 手臂 | 夹爪 |
|------|------|------|------|
| CLEAN | 活跃 | 收起（不受策略控制） | 关闭 |
| APPROACH | 活跃 | 活跃（准备姿态） | 关闭 |
| PICKUP | 活跃 | 活跃 | 活跃 |

**实现方式：**
```python
# 手臂增量：CLEAN 状态下被屏蔽
arm_delta = arm_delta * max_joint_delta * (fsm_state != CLEAN)

# 夹爪：只在 PICKUP 状态下才能关闭
gripper_close = (action[7] > 0.5) AND (fsm_state == PICKUP)
```

---

## 3. 环境详解

### 3.1 观测空间（112 维，从 97 维填充）

| 组件 | 维度 | 范围 | 说明 |
|------|------|------|------|
| velocity | 2 | 连续 | 前进速度 + 角速度 |
| lidar | 72 | [0,1] | 72 射线归一化距离 |
| detector | 6 | [-1,1] | 3 最近**可拾取**物体检测 |
| coverage | 1 | [0,1] | 当前覆盖率 |
| ee_state | 5 | [0,1] | EE 位置(3) + 臂伸展(1) + 夹爪(1) |
| target_rel_pos | 3 | 连续 | 最近可拾取 FREE 物体的相对位置 |
| target_dist_norm | 1 | [0,1] | EE 到目标距离 |
| in_grasp_range | 1 | {0,1} | 是否在抓取范围 |
| basket_norm | 1 | [0,1] | 已放置数 / 6（可拾取总数） |
| holding | 1 | {0,1} | 是否持有物体 |
| wrist_height | 1 | [0,1] | 手腕高度归一化 |
| fsm_onehot | 3 | {0,1} | FSM 状态独热编码 |
| **padding** | **15** | 0 | 填充到 112（预留给 YOLOv8 特征） |

**关键细节：** detector 只检测**可拾取**物体（索引 0-5），不检测需避开的物体（索引 6-9）。避障完全依赖 LiDAR。

### 3.2 动作空间（8 维）

与 T5 相同：

| 索引 | 含义 | FSM 门控 |
|------|------|---------|
| 0 | 线速度 | 始终活跃 |
| 1 | 角速度 | 始终活跃 |
| 2-6 | 手臂关节增量 | APPROACH + PICKUP 活跃 |
| 7 | 夹爪 | 仅 PICKUP 活跃 |

---

## 4. 奖励函数详解

| 组件 | 权重 | 触发条件 | 设计意图 |
|------|------|---------|---------|
| coverage | +2.0 | 新覆盖格子数 | 清扫是基本任务 |
| grasp_success | +10.0 | 可拾取物体被抓取 | 奖励成功抓取 |
| place_in_basket | +50.0 | 物体放入篮子 | 最高奖励 |
| wall_collision | -3.0 | LiDAR 最小距离 < 0.20m | 惩罚撞墙 |
| obj_collision_avoid | -2.0 | 碰到需避开物体（< 0.12m） | 惩罚碰到大物体 |
| obj_collision_pick | -0.5 | 碰到可拾取物体（< 0.10m，非抓取时） | 轻微惩罚（不是目的但可接受） |
| time_penalty | -0.02 | 每步 | 鼓励效率 |

**设计思路：** 碰到需避开物体的惩罚（-2.0）远大于碰到可拾取物体（-0.5），因为后者本来就是要去抓的。

---

## 5. 训练配置

### 5.1 PPO 超参数

文件：`train/.../tasks/coverage_pickup/agents/ppo.yaml`

| 参数 | 值 | 与 T5/T6 对比 |
|------|-----|-----------|
| 网络结构 | [512, 256, 128] | 同 T5 |
| Rollout 长度 | 32 | **更短**（环境更少） |
| Mini batches | 8 | **更少** |
| **学习率** | **1.0e-4** | **更低**（微调用） |
| **熵损失权重** | **0.005** | **更低**（精细利用） |
| Time limit bootstrap | True | 长时间任务 |
| **总训练步数** | **8,000,000** | **最长** |
| **并行环境数** | **32** | **最少**（最复杂） |

### 5.2 为什么参数与 T5/T6 不同？

| 差异 | 原因 |
|------|------|
| LR 1.0e-4（vs 3.0e-4） | 从 T5+T6 检查点微调，不希望覆盖已学会的技能 |
| 熵 0.005（vs 0.01） | 更倾向利用已有知识，减少随机探索 |
| 32 envs（vs 64/128） | 每个环境有 10 个物体 + 手臂，GPU 内存更紧张 |
| 8M steps（vs 3-5M） | 端到端任务最复杂，需要更长训练 |

### 5.3 环境参数

| 参数 | 值 |
|------|-----|
| Episode 时长 | **300 秒**（最长） |
| 物体数 | 10（6 可拾取 + 4 避开） |
| 接近触发距离 | 1.5m |
| 抓取距离 | 0.04m |

---

## 6. 训练步骤

### 6.1 验证

```bash
python train/scripts/common/list_envs.py --keyword CoveragePickup

python train/scripts/common/zero_agent.py \
  --task SmartCleaningRobot-CoveragePickup-v0 --num_envs 1
```

### 6.2 训练（推荐从 T5+T6 检查点开始）

```bash
# 如果有 T5/T6 检查点，可以用 --checkpoint 加载（手动合并权重）
# 否则从零开始：
python train/scripts/common/train.py \
  --task SmartCleaningRobot-CoveragePickup-v0 \
  --algorithm PPO \
  --num_envs 32 \
  --headless \
  --seed 42
```

### 6.3 分析与回放

```bash
python train/scripts/common/plot_training.py \
  train/logs/skrl/coverage_pickup/<run_dir> --smooth 10

python train/scripts/common/play.py \
  --task SmartCleaningRobot-CoveragePickup-v0 \
  --num_envs 1 --video
```

---

## 7. 结果分析

### 7.1 预期训练时间线

| 阶段 | 步数 | 行为 |
|------|------|------|
| 0 - 1M | 学习基本移动和覆盖 |
| 1M - 3M | 学会避开大物体，覆盖率提升 |
| 3M - 5M | 开始尝试抓取可拾取物体 |
| 5M - 8M | 完整 CLEAN→APPROACH→PICKUP 循环 |

### 7.2 成功标准

| 指标 | 目标 |
|------|------|
| 覆盖率 | > 70% |
| 物体放置 | ≥ 3 / 6 个可拾取物体 |
| 大物体碰撞 | 极少 |

**成功条件（触发 Episode 结束）：** 覆盖率 > 70% **或** 6 个可拾取物体全部放入篮子。

### 7.3 回放观察要点

| 行为 | 好的表现 | 需改进 |
|------|---------|--------|
| FSM 切换 | 自然过渡 CLEAN↔APPROACH↔PICKUP | 频繁无意义切换 |
| 覆盖效率 | 系统性路线 | 重复覆盖同一区域 |
| 物体分类 | 绕开大物体，接近小物体 | 对所有物体相同行为 |
| 抓取流程 | 接近→抓取→放置流畅 | 卡在某一步 |
| 整体节奏 | 清扫中自然穿插拾取 | 只做清扫或只做拾取 |

---

## 8. 常见问题

### Q1: FSM 一直停在 CLEAN 状态

**原因：** approach_trigger_dist（1.5m）太小或物体离得太远。
**排查：** 检查物体生成位置是否在合理范围内。

### Q2: 训练 8M 步但覆盖率不到 50%

**原因：** 机器人花太多时间尝试抓取。
**解决：** 增大 coverage 奖励权重（2.0→3.0）。

### Q3: 抓取成功率很低

**原因：** 从零训练的手臂控制不够精确。
**解决：** 先训练 T5 到收敛，用其检查点初始化 T7。

### Q4: 32 个环境 GPU 显存不足

**解决：** 降到 16 个环境：
```bash
python train/scripts/common/train.py ... --num_envs 16
```

### Q5: 如何知道 FSM 在哪个状态？

检查观测中的 `fsm_onehot` 组件（最后 3+15 维之前）：
- `[1, 0, 0]` = CLEAN
- `[0, 1, 0]` = APPROACH
- `[0, 0, 1]` = PICKUP

---

## 参考文件

| 文件 | 路径 |
|------|------|
| 环境实现 | `train/.../tasks/coverage_pickup/env.py` |
| 环境配置 | `train/.../tasks/coverage_pickup/env_cfg.py` |
| PPO 超参 | `train/.../tasks/coverage_pickup/agents/ppo.yaml` |
| 物体跟踪器 | `train/.../utils/object_tracker.py` |
| 覆盖追踪器 | `train/.../utils/coverage_tracker.py` |
| 检测器 | `train/.../small_objects/detector.py` |
| 共享工具 | `train/.../tasks/_shared.py` |
