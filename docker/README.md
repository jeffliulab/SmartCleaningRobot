# docker/ — 教学环境

一条命令拿到全系列统一的开发环境：**ROS 2 Jazzy + Gazebo Harmonic + Nav2 + slam_toolbox**。

```bash
cd docker
docker compose build          # 首次约 4–5 GB 下载
docker compose run --rm sim   # 进容器（仓库已挂载到 /ws/src/open-cleaning-robot）
# 容器内：
colcon build && source install/setup.bash
```

## 设计口径（为什么这么选）

- **自建仓内 Dockerfile，不用第三方现成镜像**：学员环境必须可复现，依赖全部钉在这里。
- **Jazzy + Harmonic**：当前 LTS 组合（教学寿命到 2029）；Humble 2027-05 EOL、Gazebo Classic 已 EOL。
- **ROS apt 源可换**：Dockerfile 的 `ROS_APT_REPO` 构建参数默认官方源；国内网络在
  compose.yaml 里已换成清华 TUNA 镜像（packages.ros.org 国内不可达，2026-07-31 实测），
  海外构建把它改回官方即可。
- **explore_lite 不在镜像里**：S6 才用到，且没有 Jazzy 二进制（届时从源码构建加入本 Dockerfile）。
- GUI：compose 已挂 X11；无显示环境一律 headless（`gz sim -s -r`）。
