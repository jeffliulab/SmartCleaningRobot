# 04 — 训练指南

## 前置要求

- NVIDIA GPU (建议 RTX 3070 及以上)
- NVIDIA Isaac Sim 4.x (含 Isaac Lab)
- Python 3.10+
- skrl >= 1.4.3
- CUDA 12.x

### 环境安装

```bash
# 1. 安装 Isaac Lab (按官方文档)
# 2. 安装项目依赖
conda env create -f environment.yml
conda activate cleanrobot

# 3. 安装本项目
pip install -e train/smartcleaningrobot
```

## 训练命令

### 基本训练（SAC，推荐）

```bash
python train/scripts/common/train.py \
    --task SmartCleaningRobot-MazeEscape-v0 \
    --algorithm SAC \
    --num_envs 128 \
    --headless
```

### 带视频录制的训练

```bash
python train/scripts/common/train.py \
    --task SmartCleaningRobot-MazeEscape-v0 \
    --algorithm SAC \
    --num_envs 64 \
    --video \
    --video_interval 5000
```

### 从 checkpoint 恢复训练

```bash
python train/scripts/common/train.py \
    --task SmartCleaningRobot-MazeEscape-v0 \
    --algorithm SAC \
    --checkpoint train/logs/skrl/maze_escape/.../checkpoints/best_agent.pt
```

### 使用 PPO 训练（旧版算法）

```bash
python train/scripts/common/train.py \
    --task SmartCleaningRobot-MazeEscape-v0 \
    --algorithm PPO \
    --num_envs 128 \
    --headless
```

> 注意：使用 `--algorithm PPO` 时，系统会自动加载 `skrl_maze_ppo_cfg.yaml`。

## 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务 ID | 必须指定 |
| `--algorithm` | RL 算法 (SAC/PPO/TD3) | PPO |
| `--num_envs` | 并行环境数量 | 128 (配置文件) |
| `--headless` | 无 GUI 模式（训练更快） | False |
| `--seed` | 随机种子 | 42 (配置文件) |
| `--max_iterations` | 最大迭代次数 | 配置文件中的 timesteps |
| `--device` | 计算设备 | cuda:0 |

## 监控训练

### TensorBoard

```bash
tensorboard --logdir train/logs/skrl/maze_escape
```

在浏览器打开 `http://localhost:6006`，可以看到：

- **Reward/Total**: 每个 episode 的总奖励（应该逐渐增大）
- **Episode Length**: 到达出口所需步数（应该逐渐减小）
- **Learning Rate**: 学习率变化

### 关键指标含义

| 指标 | 理想趋势 | 说明 |
|------|----------|------|
| Total reward | 持续上升 | 说明策略在改进 |
| Episode length | 持续下降 | 说明机器人更快找到出口 |
| Entropy | 先高后降 | SAC 自动调节，前期探索后期收敛 |

### 预期训练曲线

SAC 的典型收敛过程：

```
Reward
  ↑
  │            ┌─────────── 收敛（~2M 步）
  │           /
  │          /
  │    ╱────╱  快速学习期（~500K-1.5M 步）
  │   ╱
  │──╱  随机探索期（~0-500K 步）
  │
  └────────────────────────→ 训练步数
  0    500K    1M    1.5M   2M    2.5M   3M
```

- **0 - 1K 步**: 纯随机动作（`random_timesteps=1000`）
- **1K - 500K 步**: Replay buffer 填充，策略开始学习基本避障
- **500K - 1.5M 步**: 快速提升期，学会沿迷宫路径移动
- **1.5M - 3M 步**: 精细化，学会高效路径选择

## 常见问题

### Q: 训练奖励一直不增长

**可能原因**:
1. `builders/maze.py` 缺失导致环境构建失败 → 确认 `builders/` 模块存在
2. `num_envs` 太少 → 增加到 128 或 256
3. `memory_size` 太小 → 确认 replay buffer 足够大 (50000)

### Q: 机器人一直在原地转圈

**可能原因**:
1. 前向运动奖励 (`rew_forward_bonus`) 太小 → 可尝试增大到 0.2
2. 熵值过高 → 检查 `initial_entropy_value`，可降低到 0.5

### Q: 机器人频繁撞墙

**可能原因**:
1. 碰撞惩罚 (`rew_collision_penalty`) 不够 → 当前为 -5.0，可增大
2. LiDAR 检测阈值 (`collision_threshold`) 需要调整 → 当前 0.15m

### Q: 如何选择 num_envs

- GPU 显存 8GB: 建议 64-128 envs
- GPU 显存 12GB: 建议 128-256 envs
- GPU 显存 24GB+: 可以尝试 256-512 envs

更多并行环境 = 更多样本 = 更快收敛，但会占用更多显存。

## 推理 / 回放

训练完成后，可以回放最佳 checkpoint：

```bash
python train/scripts/common/play.py \
    --task SmartCleaningRobot-MazeEscape-v0 \
    --algorithm SAC \
    --num_envs 4 \
    --real-time
```

加 `--video` 可录制视频：

```bash
python train/scripts/common/play.py \
    --task SmartCleaningRobot-MazeEscape-v0 \
    --algorithm SAC \
    --num_envs 1 \
    --video \
    --video_length 500
```
