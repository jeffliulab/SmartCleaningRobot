# T3: ObjectDetection — 完整指南

> 本文档是 T3 ObjectDetection（物体检测）的自包含指南。
> T3 不是 RL 任务，而是为 T4-T7 提供感知能力的 CV 流水线。

**类型:** 计算机视觉流水线（非 RL 训练）
**用途:** 为 T4-T7 任务提供物体检测能力
**三阶段策略:** OracleDetector → YOLOv8-nano → SmolVLA-450M

---

## 目录

1. [任务描述](#1-任务描述)
2. [背景知识](#2-背景知识)
3. [Phase A: OracleDetector（仿真训练用）](#3-phase-a-oracledetector仿真训练用)
4. [Phase B: YOLOv8-nano（部署用）](#4-phase-b-yolov8-nano部署用)
5. [Phase C: SmolVLA-450M（高级部署）](#5-phase-c-smolvla-450m高级部署)
6. [物体类型定义](#6-物体类型定义)
7. [如何测试检测器](#7-如何测试检测器)
8. [常见问题](#8-常见问题)

---

## 1. 任务描述

### 1.1 为什么需要物体检测？

在 T5-T7 任务中，机器人需要知道地面上有哪些物体、在哪里、是什么类型（可拾取 vs 需避开）。物体检测模块提供这个感知能力。

### 1.2 三阶段策略

```
阶段 A (训练)           阶段 B (部署)           阶段 C (高级)
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ OracleDetector│      │ YOLOv8-nano  │      │ SmolVLA-450M │
│ 直接读仿真坐标 │ →    │ 相机图像识别   │ →    │ 视觉语言动作  │
│ 零计算开销    │      │ 15ms/帧      │      │ 端到端推理    │
│ 完美准确     │      │ 90%+ mAP    │      │ sim2real <5%  │
└──────────────┘      └──────────────┘      └──────────────┘
```

**为什么分阶段？**
- **训练时**用 OracleDetector：将感知与策略学习解耦，策略不需要学习"看懂图像"
- **部署时**用 YOLOv8：在真实机器人上没有仿真器，必须用相机
- **最终**用 SmolVLA：端到端更优雅，sim-to-real gap 更小

---

## 2. 背景知识

### 2.1 物体检测基础

物体检测的任务是：给定一张图像，输出图像中所有目标物体的**位置**和**类别**。

```
输入: 相机图像 (640×480 RGB)
输出: [
  {类别: "coin", 位置: (x1, y1, x2, y2), 置信度: 0.95},
  {类别: "hairpin", 位置: (x1, y1, x2, y2), 置信度: 0.87},
  ...
]
```

### 2.2 YOLOv8 简介

YOLO (You Only Look Once) 是最流行的实时物体检测算法：
- **单阶段检测器**：一次前向传播即可输出所有检测结果
- **v8-nano**：最轻量版本，适合边缘设备
- **推理速度**：~15ms/帧 on Orin NX 16GB

### 2.3 OracleDetector 输出格式

OracleDetector 不输出图像上的边界框，而是输出**极坐标格式**（更适合 RL 策略）：

```
每个检测到的物体输出 2 个值:
  r_norm = 物体距离 / FOV 范围，归一化到 [0, 1]
  sin_theta = sin(物体在机体坐标系中的方向角)

最多检测 K=3 个最近物体 → 输出维度 = 6
```

---

## 3. Phase A: OracleDetector（仿真训练用）

### 3.1 工作原理

OracleDetector 直接从 Isaac Sim 的 USD 场景中读取物体的世界坐标，无需相机渲染：

```python
def detect_batch(robot_xy, object_positions, robot_yaw):
    """
    输入:
        robot_xy: (N, 2) 机器人 XY 位置
        object_positions: (N, M, 3) M 个物体的 XYZ 位置
        robot_yaw: (N,) 机器人偏航角

    处理:
        1. 计算每个物体相对于机器人的距离和角度
        2. 转换到机体坐标系
        3. 选择最近的 K 个物体
        4. 归一化为 [r_norm, sin_theta] 格式

    输出: (N, K*2) 检测结果张量
    """
```

### 3.2 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| max_detections (K) | 3 | 最多返回 3 个最近物体 |
| fov_range | 2.0m | 超过 2m 的物体不检测 |
| 填充值 | (1.0, 0.0) | 不足 K 个时用"远处正前方"填充 |

### 3.3 文件位置

```
train/smartcleaningrobot/smartcleaningrobot/small_objects/detector.py
```

### 3.4 在各任务中的使用

| 任务 | 检测目标 | 用途 |
|------|---------|------|
| T5 ObjectPickup | 5 个可抓取物体 | 定位最近的目标物体 |
| T6 CoverageAvoid | 10 个地面物体 | 避开所有物体 |
| T7 CoveragePickup | 6 个可拾取物体 | 只检测可拾取类型 |

---

## 4. Phase B: YOLOv8-nano（部署用）

### 4.1 训练数据

使用 Isaac Sim 生成合成训练数据：

| 参数 | 值 |
|------|-----|
| 图像数量 | 16,000 张 |
| 分辨率 | 640×480 |
| 物体类别 | 4 类（coin, hairpin, eraser, cable） |
| 场景变化 | 随机光照、随机位置、随机背景 |

### 4.1b ColorBlobDetector（轻量替代方案）

在 YOLOv8 尚未训练好时，可以使用 HSV 颜色分割作为过渡：

```python
# 工作流程: RGB 图像 → HSV 转换 → 颜色掩膜 → 连通域分析 → bbox → 深度估计
```

颜色映射表（物体颜色需与仿真一致）：

| 物体类型 | HSV 范围 (H, S) | 说明 |
|---------|----------------|------|
| 发卡 (hairpin) | H:200-220, S:50-255 | 蓝色金属 |
| 数据线 (cable) | H:0-10, S:100-255 | 红色 |
| 硬币 (coin) | H:25-35, S:50-200 | 金黄色 |
| 橡皮 (eraser) | H:140-160, S:80-255 | 紫色 |

**局限性**：颜色受光照影响大，仅适合受控环境。

### 4.2 模型导出

训练好的 YOLOv8 模型导出为 ONNX + TensorRT 格式，在 Orin NX 上运行：

```bash
# 训练（待实现）
python train/scripts/detector/train_yolov8.py

# 导出
yolo export model=best.pt format=onnx

# 在 Orin NX 上转换为 TensorRT
trtexec --onnx=best.onnx --saveEngine=best.engine
```

### 4.2b 训练数据生成（Isaac Sim 域随机化）

使用 Isaac Sim 自动渲染合成训练数据：

- **随机光照**：方向、强度、颜色
- **随机摄像头**：俯角 30°~70°，距物体 0.1~0.5m
- **随机背景**：地板材质、颜色
- **随机物体姿态**：旋转、倾斜
- **数据量目标**：每类 2000 张 × 4 类 + 混合场景 4000 张 = 12,000 张

类别定义 (`data.yaml`)：

```yaml
nc: 6
names:
  0: hairpin        # 发卡
  1: cable          # 数据线
  2: coin           # 硬币
  3: eraser         # 橡皮
  4: obstacle_large # 不可拾取大障碍
  5: unknown        # 其他小物体
```

YOLOv8 训练命令：

```bash
pip install ultralytics
yolo detect train \
  data=train/datasets/small_objects/data.yaml \
  model=yolov8n.pt epochs=100 imgsz=320 batch=32 device=0

# 导出 ONNX（边缘部署）
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

### 4.3 当前状态

> Phase B 的训练脚本和数据生成脚本尚未实现（标记为 TODO）。
> 当前所有任务训练均使用 Phase A 的 OracleDetector。

---

## 5. Phase C: SmolVLA-450M（高级部署）

### 5.1 概念

SmolVLA（Small Vision-Language-Action）是一个端到端模型，直接从相机图像输出机器人动作：

```
相机图像 → SmolVLA → 机器人动作
```

优势：
- 无需单独的检测+策略两阶段
- Sim-to-real gap < 5%
- 可通过自然语言指令调整行为

### 5.2 性能

| 指标 | 值 |
|------|-----|
| 模型大小 | 450M 参数 |
| 推理速度 | ~15-20 FPS on Orin NX 16GB |
| YOLOv8 作为后备 | 当 SmolVLA 置信度 < 0.5 时回退 |

### 5.3 当前状态

> Phase C 仍处于规划阶段，尚未实现。

---

## 6. 物体类型定义

文件：`train/.../small_objects/object_assets.py`

| 类型 | 英文 | 典型尺寸 | 是否可拾取 | 说明 |
|------|------|---------|-----------|------|
| 硬币 | coin | 24mm 直径 | 是 | 圆盘形，最容易抓取 |
| 发卡 | hairpin | 50mm 长 | 是 | U形金属丝 |
| 橡皮 | eraser | 30×15mm | 是/否 | T7中分大小 |
| 数据线 | cable | 80mm 卷 | 是/否 | T7中作为大物体需避开 |

---

## 7. 如何测试检测器

### 7.1 在 RL 环境中间接测试

检测器是 RL 环境的一部分，随任何使用它的任务自动运行：

```bash
# T6 环境包含 OracleDetector
python train/scripts/common/random_agent.py \
  --task SmartCleaningRobot-CoverageAvoid-v0 --num_envs 1
```

### 7.2 查看检测输出

在 `env.py` 的 `_get_observations()` 中添加打印语句：

```python
detect_obs, _ = detect_objects(self.detector, ...)
print(f"Detection output: {detect_obs[0]}")
# 输出类似: tensor([0.35, 0.71, 0.62, -0.23, 1.00, 0.00])
# 含义: 物体1距0.35(×2m=0.7m), 方向sin=0.71
#       物体2距0.62(×2m=1.24m), 方向sin=-0.23
#       物体3不存在，用(1.0, 0.0)填充
```

### 7.3 深度估计（单目 2D→3D）

YOLOv8 输出 2D bbox，需要估计物体在地面的 3D 位置：

```python
def bbox_to_world(
    bbox: np.ndarray,           # [x1, y1, x2, y2] 像素坐标
    camera_matrix: np.ndarray,  # 3×3 内参矩阵
    camera_height: float = 0.1, # 摄像头距地面高度 (m)
) -> np.ndarray:                # [x_world, y_world] 地面坐标
    """利用地面平面约束将 2D bbox 中心点投影到地面"""
```

精度预期：
- 距离 0.5m：误差 < 3cm（足够触发接近逻辑）
- 距离 1.0m：误差 < 8cm（足够导航定向）

### 7.4 统一检测接口

所有检测器（Oracle / ColorBlob / YOLOv8）实现相同接口，切换时只需改配置：

```python
@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    rel_x: float     # 相对机器人 body frame，前向为正
    rel_y: float     # 相对机器人 body frame，左向为正
    distance: float  # 欧氏距离 (m)
    pickable: bool   # 是否可拾取

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, ...) -> list[Detection]: ...
```

Phase A→C 替换只需改配置中的 `detector_class`，无需修改策略代码。

---

## 8. 常见问题

### Q1: 为什么不一开始就用 YOLOv8 训练？

**原因：** 将感知与决策分离是 RL 训练的最佳实践。如果策略同时要学习"看懂图像"和"做出决策"，训练会非常困难且不稳定。OracleDetector 提供完美的感知，让策略专注于学习决策。

### Q2: OracleDetector 在真实机器人上能用吗？

**不能。** OracleDetector 依赖仿真器的内部数据，在真实世界中不存在。部署时必须用 YOLOv8 或 SmolVLA。

### Q3: 检测器输出的 sin_theta 是什么意思？

```
sin_theta > 0: 物体在机器人左前方
sin_theta = 0: 物体在正前方或正后方
sin_theta < 0: 物体在机器人右前方
```

### Q4: 为什么不用 cos_theta？

只用 sin_theta（而非完整的 sin+cos）是一种设计简化。对于近距离物体，sin_theta 已经提供了足够的方向信息。

---

## 参考文件

| 文件 | 路径 |
|------|------|
| OracleDetector | `train/.../small_objects/detector.py` |
| 物体资产定义 | `train/.../small_objects/object_assets.py` |
| 共享检测函数 | `train/.../tasks/_shared.py` → `detect_objects()` |
