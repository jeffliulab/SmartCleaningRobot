# SmartCleaningRobot 文档索引

> 本项目包含 7 个任务（T1-T7），从简单到复杂逐步构建智能清扫机器人。
> 所有文档设计为完全离线可用——一个零基础的新手跟着文档即可完成学习、训练、分析、报告。

---

## 快速导航

### 全局指南

| 文档 | 内容 | 建议阅读顺序 |
|------|------|-------------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | 从零搭建环境（驱动 -> Isaac Sim -> Conda -> 验证） | 1. 最先阅读 |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | 训练/评估/分析通用流程 + RL 基础 + PPO/SAC 详解 | 2. 环境搭好后 |
| [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md) | 通用任务报告模板 | 5. 训练完成后 |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | ROS 2 + Gazebo 部署指南（可选） | 6. 需要部署时 |

### 任务完整指南（每个任务一份自包含文档）

| 任务 | 文档 | 机器人 | 难度 | 前置依赖 |
|------|------|--------|------|---------|
| T1 MazeEscape | [FULL_GUIDE.md](tasks/t1_maze_escape/FULL_GUIDE.md) | JetBot | 入门 | 无 |
| T2 Coverage | [FULL_GUIDE.md](tasks/t2_coverage/FULL_GUIDE.md) | JetBot | 基础 | T1 (推荐) |
| T3 ObjectDetection | [FULL_GUIDE.md](tasks/t3_object_detection/FULL_GUIDE.md) | — (CV) | 知识 | 无 |
| T4 ArmGrasp | [FULL_GUIDE.md](tasks/t4_arm_grasp/FULL_GUIDE.md) | TurtleBot4+Arm | 中等 | 无 |
| T5 ObjectPickup | [FULL_GUIDE.md](tasks/t5_object_pickup/FULL_GUIDE.md) | TurtleBot4+Arm | 进阶 | T4 |
| T6 CoverageAvoid | [FULL_GUIDE.md](tasks/t6_coverage_avoid/FULL_GUIDE.md) | JetBot | 中等 | T2 (推荐) |
| T7 CoveragePickup | [FULL_GUIDE.md](tasks/t7_coverage_pickup/FULL_GUIDE.md) | TurtleBot4+Arm | 高级 | T5 + T6 |

### 建议学习路径

```
入门路线（底盘导航）：   T1 -> T2 -> T6
手臂路线（机械臂操控）： T4 -> T5
端到端（最终整合）：     T5 + T6 -> T7
感知路线（物体检测）：   T3（知识，贯穿 T4-T7）
```

---

## 文档结构

```
docs/
├── README.md               ← 本文件（文档索引）
├── GETTING_STARTED.md      ← 从零搭建环境
├── TRAINING_GUIDE.md       ← RL 基础 + 训练/评估通用流程
├── REPORT_TEMPLATE.md      ← 任务报告模板
├── DEPLOYMENT_GUIDE.md     ← ROS 2 + Gazebo 部署指南
└── tasks/
    ├── t1_maze_escape/FULL_GUIDE.md
    ├── t2_coverage/FULL_GUIDE.md
    ├── t3_object_detection/FULL_GUIDE.md
    ├── t4_arm_grasp/FULL_GUIDE.md
    ├── t5_object_pickup/FULL_GUIDE.md
    ├── t6_coverage_avoid/FULL_GUIDE.md
    └── t7_coverage_pickup/FULL_GUIDE.md
```

每个任务的 FULL_GUIDE.md 是**自包含**的完整指南，包含：
- 任务描述与目标
- 背景知识与算法原理
- 环境详解（观测/动作空间）
- 奖励函数设计
- 训练配置与步骤
- 结果分析与预期指标
- 常见问题
