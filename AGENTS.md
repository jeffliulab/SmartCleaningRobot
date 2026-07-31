# AGENTS.md — Open Cleaning Robot

开源扫地机器人**教学框架仓**：学员 clone 下来跟着偃师造物（yanshirobotics.com）的
「机器人算法 → 扫地机器人 / Cleaning Robot Series」教程逐章动手。
**不是** RL 研究仓、**不是**商业产品——代码即教材，可读性优先于技巧。

## 任务 → 去哪查

| 想干什么 | 去哪 |
|---|---|
| 跑环境 / 起仿真 | `docker/README.md`（Docker 一键：Jazzy + Gazebo Harmonic） |
| 机器人模型（URDF、尺寸、传感器挂载） | `src/cr_description/`（上游出处见其 `UPSTREAM.md`） |
| 某章做什么、学员作业是什么 | `src/cr_<章节包>/README.md` + 站内对应 S 的教程 |
| 课程整体框架（S1–S6/SS1 与 E 计划） | 偃师造物轻项目站各项目的 E0 路线卡 |
| 上一版（Isaac Lab RL）代码 | git tag `rl-legacy`；ROS Noetic 旧版在 `noetic` 分支 |
| 实验数字的出处 | 各章 `docs/` 指南 + 落盘记录（正文数字必须可追） |

## 目录地图

- `docker/` — 教学环境镜像（Dockerfile + compose），全系列统一
- `src/cr_description/` — 机器人模型（vendor 自 oomwoo-one，Apache-2.0）
- `src/cr_gazebo/` — Gazebo Harmonic worlds + 污渍栅格计分节点
- `src/cr_bringup/` — 一键 launch 组合（sim / slam / nav / coverage）
- `src/cr_teleop` / `cr_maze` / `cr_slam` / `cr_navigation` / `cr_coverage` / `cr_explore` — S1–S6 各章包
- `docs/` — 每章指南
- `legacy/` — 上一版文档存档

## 怎么跑起来

```bash
cd docker && docker compose build && docker compose run --rm sim
colcon build && source install/setup.bash   # 容器内
```

章节级入口随各章展开在对应包的 README 里给出（目标：每章一条 launch 命令）。

## 红线

- **禁硬编码**：路径、话题名、阈值、坐标一律进配置/launch 参数/具名常量；
  写死结果、伪造数据让 demo「看起来通了」是绝对禁止项。
- **不模拟吸尘物理**：清扫用污渍栅格按面积计分（扫过即清除），不许假装做了吸力仿真。
- **学员包与答案隔离**：main 分支只放 starter（带 TODO）；参考答案只在 `solutions` 分支。
- **模型 vendor 不依赖**：oomwoo-one 的 URDF 拷在仓内、钉住上游 commit；
  不许让学员环境依赖第三方 Docker 镜像或在线资产。
- **第三方资产标出处**：上游 URL、commit、许可证写进对应目录的 `UPSTREAM.md`。
- **真机命令一律由人亲手执行**，agent 只做软件侧准备（SS1 的红线）。

## 提交约定

- 不写 `Co-Authored-By`；禁 `git add -A`（逐文件 add）；push 由作者（Jeff）发话。

---

> 若本地同目录存在 `CLAUDE.md`，请一并阅读（内部开发笔记，未入库）。
