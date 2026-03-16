# 迷宫逃脱任务 (Maze Escape Task)

本目录包含迷宫逃脱任务的完整教程文档。迷宫任务是一个独立于扫地机器人清洁任务的 RL 练手项目，目标是训练 JetBot 机器人在 DFS 生成的迷宫中找到并到达出口。

## 文档索引

| 文档 | 内容 |
|------|------|
| [01_task_overview.md](01_task_overview.md) | 任务概述：迷宫生成、环境参数、与清洁任务的区别 |
| [02_sensors.md](02_sensors.md) | 传感器使用：LiDAR、里程计、机体坐标系计算 |
| [03_algorithm.md](03_algorithm.md) | 算法详解：SAC 策略、观测空间、奖励设计、网络结构 |
| [04_training.md](04_training.md) | 训练指南：环境搭建、训练命令、TensorBoard 监控 |
| [05_deployment.md](05_deployment.md) | 部署测试：模型导出、Gazebo 仿真、ROS 2 部署 |

## 项目结构（迷宫相关）

```
SmartCleaningRobot/
├── train/smartcleaningrobot/smartcleaningrobot/
│   ├── scenes/builders/
│   │   ├── maze.py              # DFS 迷宫生成 + BFS 距离场 (Isaac Lab)
│   │   └── simple.py            # 简单房间场景 (Isaac Lab)
│   └── tasks/maze_escape/
│       ├── env.py               # 迷宫逃脱 RL 环境
│       ├── env_cfg.py           # 环境配置
│       └── agents/
│           ├── sac.yaml         # SAC 训练配置
│           └── ppo.yaml         # PPO 训练配置（旧）
├── train/scripts/
│   ├── common/
│   │   ├── train.py             # 通用训练脚本（用 --task 指定任务）
│   │   └── play.py              # 通用推理回放
│   └── MazeDFSv1_Jetbot/
│       ├── MazeDFSv1_Jetbot_EscapeMaze.py  # 迷宫逃脱可视化演示
│       └── benchmark/maze/      # 对比实验（A*, 左手法则, RL）
├── deploy/
│   ├── scripts/
│   │   ├── gen_maze_world.py    # Gazebo 迷宫 SDF 生成
│   │   └── export_model.py      # 模型导出（支持 --task maze）
│   └── cleaning_robot_ros/cleaning_robot_ros/
│       ├── obs_builders/maze.py # 迷宫 41 维观测构建
│       ├── policy_node.py       # ROS 2 策略节点（支持 maze_model）
│       └── baselines/wall_follower.py  # 左手法则（传统方法对比）
└── docs/tasks/maze_escape/      # ← 你在这里
```

## 快速开始

```bash
# 训练（SAC 算法）
python train/scripts/common/train.py --task SmartCleaningRobot-MazeEscape-v0 --algorithm SAC --num_envs 128

# 查看训练曲线
tensorboard --logdir train/logs/skrl/maze_escape

# 导出模型
python deploy/scripts/export_model.py --task maze \
    --checkpoint train/logs/skrl/maze_escape/.../checkpoints/best_agent.pt \
    --output deploy/models/maze_agent_scripted.pt

# 在 Gazebo 中测试
ros2 run cleaning_robot_ros policy_node --ros-args -p policy_type:=maze_model \
    -p model_path:=models/maze_agent_scripted.pt
```
