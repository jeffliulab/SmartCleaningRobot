# open-cleaning-robot 教学环境镜像
# 栈：ROS 2 Jazzy + Gazebo Harmonic + Nav2 + slam_toolbox（选型依据见 docs/ 与站内 E0 路线卡）
# 说明：explore_lite（S6 前沿探索用）无 Jazzy 二进制，届时从源码构建加入，见本目录 README。
# 包名注记：Gazebo 集成用 ros-jazzy-ros-gz（packages.ros.org 官方仓库的元包，拉
# ros_gz 桥接 + gz-harmonic vendor 全家桶）；ros-jazzy-ros-gzharmonic 只在
# OSRF 自家源里有，ROS 官方源不存在（2026-07-31 在 TUNA 索引实证）。
FROM osrf/ros:jazzy-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

# ROS apt 源：默认官方；国内网络构建经 compose 传入清华 TUNA 镜像
# （packages.ros.org 在国内不可达，2026-07-31 实测）。Gazebo Harmonic 的 deb
# 同样在 ROS 源里（gz-harmonic-* vendor 包），不需要额外的 osrf 源。
ARG ROS_APT_REPO=http://packages.ros.org/ros2/ubuntu
# Ubuntu 主源同理：容器不走宿主机 VPN，ISP DNS 解析不动 archive.ubuntu.com，
# 国内构建一并换 TUNA（ARG 默认官方，海外构建不受影响）。
ARG UBUNTU_APT_REPO=http://archive.ubuntu.com/ubuntu
RUN set -e; \
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do \
      [ -f "$f" ] || continue; \
      sed -i "s|http://packages.ros.org/ros2/ubuntu|${ROS_APT_REPO}|g; \
              s|https://packages.ros.org/ros2/ubuntu|${ROS_APT_REPO}|g; \
              s|http://archive.ubuntu.com/ubuntu|${UBUNTU_APT_REPO}|g; \
              s|https://archive.ubuntu.com/ubuntu|${UBUNTU_APT_REPO}|g; \
              s|http://security.ubuntu.com/ubuntu|${UBUNTU_APT_REPO}|g; \
              s|https://security.ubuntu.com/ubuntu|${UBUNTU_APT_REPO}|g; \
              s|^Types: deb deb-src|Types: deb|" "$f"; \
    done; \
    apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-ros-gz \
      ros-jazzy-navigation2 \
      ros-jazzy-nav2-bringup \
      ros-jazzy-slam-toolbox \
      ros-jazzy-teleop-twist-keyboard \
      python3-colcon-common-extensions \
      git \
    && rm -rf /var/lib/apt/lists/*

# 教学仓经 compose 挂载到 /ws/src/open-cleaning-robot，构建产物在 /ws/install
WORKDIR /ws
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc && \
    echo '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash || true' >> /root/.bashrc

CMD ["bash"]
