# 02 — 传感器使用

## 传感器概览

迷宫逃脱任务只使用两种传感器（不需要摄像头）：

| 传感器 | 用途 | 仿真实现 | 真实硬件 |
|--------|------|----------|----------|
| **LiDAR** | 障碍物检测 | RayCaster (360 射线) | 2D 激光雷达 |
| **里程计 (Odometry)** | 位置和速度估计 | 仿真器直接提供 | 轮式编码器 + IMU |

## LiDAR

### 配置

```python
JETBOT_LIDAR_CFG = RayCasterCfg(
    prim_path="/World/envs/env_.*/Robot/chassis",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.05)),
    pattern_cfg=patterns.LidarPatternCfg(
        channels=1,
        vertical_fov_range=(0.0, 0.0),      # 2D 平面扫描
        horizontal_fov_range=(0.0, 360.0),   # 360° 全向
        horizontal_res=1.0,                   # 每度 1 条射线
    ),
    max_distance=3.5,                         # 最大检测距离 3.5m
)
```

### 数据处理流程

```
原始 LiDAR (360 条射线)
    ↓ 每 10 条取 1 条 (::10)
下采样 (36 条射线)
    ↓ clamp(0, 3.5) / 3.5
归一化到 [0, 1] (36 维)
    ↓
输入策略网络
```

### 射线布局

36 条射线均匀分布在 360°，每条间隔 10°：

```
          前方 (idx=0, 0°)
            ↑
    idx=5 / | \ idx=35
   (50°) /  |  \ (350°)
        /   |   \
idx=9 ←  机器人  → idx=27
(90°)    (左)    (右, 270°)
        \   |   /
         \  |  /
    idx=13\ | /idx=23
            ↓
      后方 (idx=18, 180°)
```

## 里程计 (Odometry)

### 获取方式

**仿真中**：直接从仿真器获取精确值

```python
root_pos = robot.data.root_pos_w        # 世界位置 (x, y, z)
root_quat = robot.data.root_quat_w      # 四元数朝向
lin_vel = robot.data.root_com_lin_vel_b  # 机体坐标系线速度
ang_vel = robot.data.root_com_ang_vel_b  # 机体坐标系角速度
```

**真实机器人**：从轮式编码器计算

```
左轮编码器 → v_left  = 左轮角速度 × 轮径 (0.0325m)
右轮编码器 → v_right = 右轮角速度 × 轮径 (0.0325m)

线速度: v = (v_left + v_right) / 2
角速度: ω = (v_right - v_left) / 轮距 (0.118m)

位置积分:
  yaw += ω × dt
  x   += v × cos(yaw) × dt
  y   += v × sin(yaw) × dt
```

### 里程计漂移

真实里程计会随时间累积误差（漂移）：

| 运行时间 | 典型角度漂移 (MPU6050) | 位置漂移 | 对迷宫的影响 |
|----------|----------------------|----------|-------------|
| 10 秒 | ~0.5° | ~1 cm | 可忽略 |
| 60 秒 | ~3° | ~5 cm | 很小 |
| 120 秒 | ~6° | ~10 cm | 可接受（通道宽 ~0.45m） |

## 机体坐标系 (Body Frame)

### 为什么使用机体坐标系

机体坐标系是以机器人自身为中心的坐标系，跟着机器人一起移动和旋转。

**世界坐标系的问题**：同一个出口位置，机器人朝向不同时世界坐标方向相同，但需要的转向动作完全不同。策略网络必须自己学会 yaw + exit_dir → 转向这个隐式三角函数关系。

**机体坐标系的优势**：出口信息直接表示为"在我前方偏左 30°"这种相对信息，策略可以直接做出反应。

### 坐标转换

将出口的世界坐标方向转换为机体坐标系方向：

```python
# 出口相对于机器人的世界坐标偏移
dx_world = exit_x - robot_x
dy_world = exit_y - robot_y

# 旋转到机体坐标系（2D 旋转矩阵）
dx_body =  dx_world * cos(yaw) + dy_world * sin(yaw)   # 前方分量
dy_body = -dx_world * sin(yaw) + dy_world * cos(yaw)   # 左方分量

# 极坐标表示
angle = atan2(dy_body, dx_body)
exit_dir = (sin(angle), cos(angle))   # 用 sin/cos 而非原始角度，避免 ±π 不连续
distance = sqrt(dx_world² + dy_world²)
```

### 不需要额外传感器

机体坐标系转换只是一步数学运算，使用已有的里程计数据（位置 + 朝向）即可完成，不需要任何新传感器：

- **IMU 陀螺仪** → 提供 yaw 角（相对于开机时刻）
- **轮式编码器** → 提供 (x, y) 位置积分
- **已知出口位置** → 固定值 (0, -3.15)

> 注意：JetBot 的 IMU (MPU6050) 没有磁力计，不能感应绝对南北方向。但迷宫任务只需要相对角度变化，从开机时刻积分即可。
