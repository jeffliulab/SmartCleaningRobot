# Open Cleaning Robot

An open-source robot vacuum cleaner project for education — simulation-first, ROS 2 classic stack.

This repo is the code companion of the **Cleaning Robot Series** on
[yanshirobotics.com](https://yanshirobotics.com) (机器人算法 → 扫地机器人):
S1 ROS & Gazebo basics → S2 maze escape (reactive) → S3 SLAM & localization →
S4 navigation & go-home → S5 coverage cleaning → S6 explore-and-clean → SS1 sim2real.

**Stack:** ROS 2 Jazzy · Gazebo Harmonic · Nav2 · slam_toolbox · Docker.
**Robot model:** vendored from [oomwoo-one](https://github.com/makerspet/oomwoo-one) (Apache-2.0),
a ~349 mm disc-shaped differential-drive vacuum robot with 2D LiDAR.

## Quick start

```bash
cd docker
docker compose build          # first build downloads ~4–5 GB
docker compose run --rm sim   # repo is mounted at /ws/src/open-cleaning-robot
# inside the container:
colcon build && source install/setup.bash
```

## Layout

- `docker/` — the one-command teaching environment (see `docker/README.md`)
- `src/cr_description/` — robot model (vendored oomwoo-one URDF)
- `src/cr_gazebo/` — worlds + cleaning score tooling
- `src/cr_bringup/` — one-launch entry points per chapter
- `src/cr_teleop|cr_maze|cr_slam|cr_navigation|cr_coverage|cr_explore/` — per-chapter packages (S1–S6)
- `docs/` — per-chapter guides
- `legacy/` — archived docs from the previous RL-based iteration (code lives on the `rl-legacy` tag and the `noetic` branch)

## Status

2026-07-31: repository rebuilt as a teaching framework (the earlier Isaac Lab / RL iteration
was retired — see tag `rl-legacy`; the ROS Noetic version remains on branch `noetic`).
Chapter packages are skeletons; content lands as the course chapters roll out.

---

> 🤖 AI agents: read `AGENTS.md` first.
