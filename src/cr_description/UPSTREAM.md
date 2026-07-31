# UPSTREAM — cr_description 的上游出处

本包的机器人模型 **vendor 自 [makerspet/oomwoo-one](https://github.com/makerspet/oomwoo-one)**：

- **上游 commit**：`f52d138f88f6034c85c4a42fdcf548372abe39da`（2026-07-21，"docs: reference renamed oomwoo_gazebo / oomwoo_bringup packages"）
- **许可证**：Apache-2.0（原件存于本目录 `LICENSE.upstream`），作者 Ilia O.（makerspet.com / kaia.ai）
- **Vendor 日期**：2026-07-31

## 取了什么

| 内容 | 位置 | 说明 |
|---|---|---|
| `urdf/`（robot.urdf.xacro、params、plugins、inertial、materials） | `urdf/` | 整机模型 + gz-sim 插件（diff-drive / odometry / joint-state / bumper contact / gpu_lidar），未改动 |
| `config/gz_bridge.yaml` | `config/` | ros_gz 桥接配置；⚠️ bumper 桥接 topic 绑定 world 名 `default` 与模型名 `oomwoo_one`（spawn 名必须一致，见文件内注释） |
| `docs/sim-bumpers.md` | `docs/` | 上游的 bumper 排坑文档（fixed-joint lump 改名等三个雷） |
| `test/test_bumper_wiring.py` | `test/` | 上游的接线回归测试 |

## 没取什么（以及为什么）

- `launch/bringup.launch.py`、`config/kaiaai.yaml`、`telem.yaml`、`vacuum_bridge.yaml`、`self_drive_gazebo.yaml` —— 依赖 kaiaai 生态包与真机 bridge，教学仓不用；launch 由 `cr_bringup` 自写
- `config/navigation.yaml`、`cartographer_lds_2d.lua`、`explore_lite.yaml` —— 我们用 slam_toolbox + 自调 Nav2 参数（`cr_slam` / `cr_navigation`），不用 Cartographer
- `rviz/` —— 各章需要时再挑

## 升级策略

上游尺寸仍是近似值（等真机落地精修）。需要跟进上游时：对照上游 commit diff，
重点看 `urdf/` 与 `config/gz_bridge.yaml`，同步后更新本文件的 commit 号并跑
`test/test_bumper_wiring.py`。
