# SmartCleaningRobot — 从零开始环境搭建指南

> 本文档面向完全零基础的新手，假设你在断网环境下（已提前下载好安装包），按步骤即可完成从安装到第一次成功训练。

---

## 目录

1. [硬件要求](#1-硬件要求)
2. [软件依赖总览](#2-软件依赖总览)
3. [第一步：安装 NVIDIA 驱动和 CUDA](#3-第一步安装-nvidia-驱动和-cuda)
4. [第二步：安装 Isaac Sim](#4-第二步安装-isaac-sim)
5. [第三步：安装 Conda 环境](#5-第三步安装-conda-环境)
6. [第四步：安装 Isaac Lab](#6-第四步安装-isaac-lab)
7. [第五步：安装本项目](#7-第五步安装本项目)
8. [第六步：验证安装](#8-第六步验证安装)
9. [常见安装问题](#9-常见安装问题)
10. [（可选）部署环境搭建](#10-可选部署环境搭建)

---

## 1. 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA RTX 3060 (12GB VRAM) | RTX 4070 Ti 或更高 |
| CPU | Intel i7-10700 / AMD Ryzen 7 | 8核16线程以上 |
| 内存 | 16 GB | 32 GB |
| 硬盘 | 50 GB 可用空间（SSD） | 100 GB SSD |
| 操作系统 | Windows 10/11 64-bit | Windows 11 |

> **为什么需要 NVIDIA GPU？** Isaac Sim 是 NVIDIA 开发的物理仿真器，必须使用 NVIDIA GPU 进行渲染和物理计算。AMD 显卡不支持。

---

## 2. 软件依赖总览

本项目的软件栈从底到顶：

```
┌─────────────────────────────────────┐
│  本项目 (SmartCleaningRobot)         │  ← pip install -e
├─────────────────────────────────────┤
│  SKRL 1.4.3 (RL 训练框架)           │  ← conda 环境自带
├─────────────────────────────────────┤
│  Isaac Lab 0.54.2 (机器人 RL 框架)   │  ← 需要单独安装
├─────────────────────────────────────┤
│  Isaac Sim 5.1.0 (物理仿真器)        │  ← Omniverse Launcher
├─────────────────────────────────────┤
│  PyTorch 2.7.0 + CUDA 12.8          │  ← conda 环境自带
├─────────────────────────────────────┤
│  Conda (Miniforge / Miniconda)       │  ← 包管理器
├─────────────────────────────────────┤
│  NVIDIA Driver 560+ / CUDA 12.x     │  ← 系统级别
└─────────────────────────────────────┘
```

---

## 3. 第一步：安装 NVIDIA 驱动和 CUDA

### 3.1 检查当前驱动版本

打开命令行（CMD 或 PowerShell），运行：

```bash
nvidia-smi
```

输出示例：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 560.94   Driver Version: 560.94   CUDA Version: 12.6            |
|-------------------------------+----------------------+----------------------+
| GPU Name          | ...      | Memory-Usage         | GPU-Util             |
| NVIDIA RTX 4070   | ...      | 1024MiB / 12288MiB  | 0%                   |
+-------------------------------+----------------------+----------------------+
```

- **驱动版本 ≥ 560** → 无需更新，跳到第 4 步
- **驱动版本 < 560 或命令报错** → 需要安装/更新驱动

### 3.2 安装驱动

1. 前往 [NVIDIA 驱动下载页面](https://www.nvidia.com/drivers)（断网时提前下载）
2. 选择你的 GPU 型号，下载 Game Ready 或 Studio 驱动
3. 运行安装程序，选择"快速安装"
4. 重启电脑
5. 再次运行 `nvidia-smi` 确认版本 ≥ 560

> **注意：** 不需要单独安装 CUDA Toolkit。Isaac Sim 自带 CUDA 运行时。

---

## 4. 第二步：安装 Isaac Sim

### 4.1 安装 Omniverse Launcher

1. 前往 [NVIDIA Omniverse](https://www.nvidia.com/omniverse) 下载 Omniverse Launcher
2. 安装并登录 NVIDIA 账号（首次需要联网注册，之后可断网使用）
3. 在 Launcher 的 **Exchange** 标签页搜索 "Isaac Sim"
4. 安装 **Isaac Sim 2024.2.0**（对应 Isaac Sim 5.1.0）

### 4.2 验证 Isaac Sim

1. 在 Omniverse Launcher 中点击 **Launch** 启动 Isaac Sim
2. 等待加载完成（首次启动需要 3-5 分钟编译着色器）
3. 看到 Isaac Sim 编辑器界面即安装成功
4. 关闭 Isaac Sim

### 4.3 记录 Isaac Sim 路径

默认安装路径为：
```
C:\Users\<用户名>\AppData\Local\ov\pkg\isaac-sim-5.1.0
```

后续步骤需要用到这个路径。可以设置环境变量方便使用：

```bash
# 在 PowerShell 中（根据你的实际路径修改）
[System.Environment]::SetEnvironmentVariable("ISAACSIM_PATH", "C:\Users\<用户名>\AppData\Local\ov\pkg\isaac-sim-5.1.0", "User")
```

---

## 5. 第三步：安装 Conda 环境

### 5.1 安装 Miniforge（推荐）

如果你还没有 Conda，推荐安装 Miniforge：

1. 下载 Miniforge：https://github.com/conda-forge/miniforge/releases
2. 运行安装程序，勾选 "Add to PATH"
3. 重启终端

验证：
```bash
conda --version
# 应输出类似: conda 24.x.x
```

### 5.2 从 environment.yml 创建环境

本项目提供了完整的 conda 环境文件，包含所有依赖（约 250 个包）：

```bash
cd d:\Projects\SmartCleaningRobot
conda env create -f environment.yml
```

这会创建名为 `isaaclab` 的 conda 环境，包含：

| 核心包 | 版本 | 用途 |
|--------|------|------|
| Python | 3.11.14 | 运行时 |
| PyTorch | 2.7.0+cu128 | 深度学习框架 |
| Isaac Lab | 0.54.2 | 机器人 RL 框架 |
| SKRL | 1.4.3 | RL 算法库（PPO, SAC 等） |
| Gymnasium | 1.2.1 | 标准 RL 接口 |
| NumPy | 1.26.0 | 数值计算 |
| Matplotlib | 3.10.3 | 绘图 |
| TensorBoard | 2.20.0 | 训练监控 |

### 5.3 激活环境

```bash
conda activate isaaclab
```

> **提示：** 每次打开新终端都需要运行 `conda activate isaaclab`。

---

## 6. 第四步：安装 Isaac Lab

Isaac Lab 需要在 conda 环境内安装，并链接到 Isaac Sim：

```bash
conda activate isaaclab

# 如果 Isaac Lab 尚未在 conda 环境中安装，参考官方文档：
# https://isaac-sim.github.io/IsaacLab/main/source/setup/installation.html

# 验证 Isaac Lab 安装
python -c "import isaaclab; print(isaaclab.__version__)"
# 应输出: 0.54.2 或类似版本
```

> **注意：** 如果 `environment.yml` 中已经包含了 Isaac Lab，则这步可以跳过。运行上面的验证命令确认即可。

---

## 7. 第五步：安装本项目

```bash
conda activate isaaclab
cd d:\Projects\SmartCleaningRobot

# 以开发模式安装（-e 表示 editable，修改代码后无需重新安装）
pip install -e train/smartcleaningrobot
```

安装完成后，项目的 7 个 Gym 环境会自动注册到 Isaac Lab。

---

## 8. 第六步：验证安装

### 8.1 验证环境注册

```bash
python train/scripts/common/list_envs.py
```

预期输出（6 个 RL 环境）：

```
+-------+-------------------------------------------+
| S.No. | Task Name                                 |
+-------+-------------------------------------------+
| 1     | SmartCleaningRobot-Coverage-v0            |
| 2     | SmartCleaningRobot-MazeEscape-v0          |
| 3     | SmartCleaningRobot-ArmGrasp-v0            |
| 4     | SmartCleaningRobot-ObjectPickup-v0        |
| 5     | SmartCleaningRobot-CoverageAvoid-v0       |
| 6     | SmartCleaningRobot-CoveragePickup-v0      |
+-------+-------------------------------------------+
```

### 8.2 验证仿真（零动作测试）

```bash
# 机器人站在原地不动，验证场景加载和物理仿真正常
python train/scripts/common/zero_agent.py --task SmartCleaningRobot-MazeEscape-v0 --num_envs 1
```

如果看到 Isaac Sim 窗口打开、迷宫场景加载、机器人出现在迷宫中，则安装成功。

### 8.3 验证训练（快速冒烟测试）

```bash
# 只跑 100 步，确认训练流程没有报错
python train/scripts/common/train.py \
  --task SmartCleaningRobot-MazeEscape-v0 \
  --algorithm SAC \
  --num_envs 2 \
  --headless \
  --max_iterations 100
```

无报错 + 看到日志输出即验证成功。

---

## 9. 常见安装问题

### Q1: `ModuleNotFoundError: No module named 'isaacsim'`

**原因：** Isaac Sim 路径未正确配置。

**解决：**
```bash
# 确认 ISAACSIM_PATH 环境变量已设置
echo $ISAACSIM_PATH
# 如果为空，手动设置（见第 4.3 节）
```

### Q2: `CUDA out of memory`

**原因：** GPU 显存不足。

**解决：** 减少并行环境数量：
```bash
python train/scripts/common/train.py --task ... --num_envs 4  # 从默认的 64/128 降低
```

### Q3: `RuntimeError: CUDA error: device-side assert triggered`

**原因：** 通常是观测值中出现 NaN 或 Inf。

**解决：**
1. 确认安装版本匹配（PyTorch 2.7.0 + CUDA 12.8）
2. 尝试重启并用少量环境测试

### Q4: conda 创建环境极慢或失败

**原因：** conda 依赖求解器性能问题。

**解决：**
```bash
# 使用 mamba 替代 conda（更快的求解器）
conda install -n base mamba
mamba env create -f environment.yml
```

### Q5: Isaac Sim 启动后黑屏

**原因：** GPU 驱动版本过旧或显卡不支持 RTX。

**解决：**
1. 更新到最新 NVIDIA 驱动
2. 检查是否为 RTX 系列显卡（GTX 系列不完全支持）

### Q6: `pip install -e` 报错 `setup.py not found`

**原因：** 路径不正确。

**解决：**
```bash
# 确认是在正确目录下
ls train/smartcleaningrobot/setup.py  # 应该存在
pip install -e train/smartcleaningrobot
```

---

## 10. （可选）部署环境搭建

部署环境用于将训练好的模型部署到 ROS 2 仿真或实际机器人上。如果你只需要训练和分析，可以跳过这部分。

### 10.1 系统要求

- **操作系统：** Ubuntu 22.04（推荐使用 WSL2）
- **ROS 2 版本：** Humble Hawksbill
- **仿真器：** Gazebo 11

### 10.2 安装步骤

详见 [deploy/README.md](../deploy/README.md) 和 [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)。

### 10.3 快速验证

```bash
# 使用简单策略（不需要训练模型）
ros2 launch cleaning_robot_ros gazebo_sim.launch.py scene:=hospital policy_type:=simple
```

---

## 下一步

环境搭建完成后，请阅读：

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) — 通用训练、评估、结果分析指南
- [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md) — 任务报告模板
- 各任务的完整指南：
  - [T1 MazeEscape](tasks/t1_maze_escape/FULL_GUIDE.md)
  - [T2 Coverage](tasks/t2_coverage/FULL_GUIDE.md)
  - [T3 ObjectDetection](tasks/t3_object_detection/FULL_GUIDE.md)
  - [T4 ArmGrasp](tasks/t4_arm_grasp/FULL_GUIDE.md)
  - [T5 ObjectPickup](tasks/t5_object_pickup/FULL_GUIDE.md)
  - [T6 CoverageAvoid](tasks/t6_coverage_avoid/FULL_GUIDE.md)
  - [T7 CoveragePickup](tasks/t7_coverage_pickup/FULL_GUIDE.md)
