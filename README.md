# Smart Cleaning Robot

A growing robot vacuum cleaner project, the goal is to build an open source robot vacuum cleaner project for educational learning.

## Brief Introduction and Versions

I am currently rebuilding the project to explore deeper in AI applications on cleaning robot, and adding features such as reinforcement learning; the original ROS Noetic version has been fully migrated to the [noetic branch](https://github.com/jeffliulab/SmartCleaningRobot/tree/noetic). See details in following introductions.

### Main Branch (ROS Humble & Isaac Lab)

Train intelligent cleaning robots using **reinforcement learning** in NVIDIA Isaac Lab simulation, then deploy to ROS 2 + Gazebo / real hardware. The project defines 7 progressive tasks (T1-T7) covering maze navigation, floor coverage, object detection, arm grasping, and end-to-end mobile manipulation.

**Stack:** Isaac Sim 5.1.0 / Isaac Lab 0.54.2 / SKRL 1.4.3 / PyTorch 2.7.0

| Task               | Description                                       | Robot                   | Status                       |
| ------------------ | ------------------------------------------------- | ----------------------- | ---------------------------- |
| T1 MazeEscape      | Navigate DFS maze to exit (SAC)                   | JetBot                  | Code ready, pending training |
| T2 Coverage        | Maximise floor coverage (PPO)                     | JetBot                  | Code ready, pending training |
| T3 ObjectDetection | CV pipeline: Oracle → YOLOv8 → SmolVLA          | —                      | Phase A done, B/C TODO       |
| T4 ArmGrasp        | Fixed-base arm reach & grasp (3-stage curriculum) | TurtleBot4 + WidowX 250 | Code ready, pending training |
| T5 ObjectPickup    | Mobile manipulation: navigate + grasp + place     | TurtleBot4 + WidowX 250 | Code ready, pending training |
| T6 CoverageAvoid   | Coverage while avoiding floor objects             | JetBot                  | Code ready, pending training |
| T7 CoveragePickup  | End-to-end: clean + classify + pick up (FSM)      | TurtleBot4 + WidowX 250 | Code ready, pending training |

**Next:** Train T1/T2 → T4 → T5/T6 → T7, then write detailed experimental reports and deploy via ROS 2 + Gazebo.

**Docs:** Complete offline-ready guides in [`docs/`](docs/README.md).

### ROS Noetic Branch (Origin System)

Noetic branch is based on ROS Noetic, Gazebo, RViz, and traditional control algorithms. You can find details in [noetic branch](https://github.com/jeffliulab/SmartCleaningRobot/tree/noetic).

Noetic Version Demos: (including exploration, mapping, cleaning, ..)

- Real Demo Link: https://youtu.be/zmo8CKolIh4?si=pPlF57XZn4DD5oo9
- Sim Demo Link: https://youtu.be/rqXiXsVubhQ?si=3jIF6Q37Kqr45SRk

<img src="./docs/readme/noetic/modules.png" width="600">

Mapping:

<img src="./docs/readme/noetic/slam_3_explore2.png" width="600">

Control Panel:

<img src="./docs/readme/noetic/control_panel_gui.png" width="400">

Planning:

<img src="./docs/readme/noetic/route_analysis_2.png" width="400">

<img src="./docs/readme/noetic/route_analysis_4.png" width="400">

Hardware Platform:

<img src="./docs/readme/noetic/turtlebot3_built.png" width="400">

### Smart Glove Branch

This is a branch ([repo: SmartGlove](https://github.com/jeffliulab/SmartGlove?tab=readme-ov-file)) focus on smart glove, which can control robot arm and experimentally connect into a self-designed IoT platform.

<img src="./docs/readme/robot_arm/1751419760912.png" width="400">

Hardware implementations:

<img src="./docs/readme/robot_arm/framework.png" width="400">

ROS topics:

<img src="./docs/readme/robot_arm/ros.png" width="400">

Robot arm:

<img src="./docs/readme/robot_arm/1751419900912.png" width="400">

## Furthermore

### Future Plans

* Deploy trained policies to real robot (Jetson Orin NX + TurtleBot4 + WidowX 250), validate sim-to-real transfer
* End-to-end vision-language-action model (SmolVLA-450M) replacing the Oracle+YOLOv8 pipeline
* Multi-room navigation with SLAM-based global planning
* Multi-robot coordination for large-scale cleaning scenarios
* Mobile app control panel with real-time coverage visualization
* Integration with IoT ecosystem (smart home sensors, scheduling, notifications)

### Relevant Literature and References

This project originated in the Brandeis Robotics Lab and was conducted under the guidance of Professor Pito Salas.

Related research, tutorial and reference for algorithms:

- Frontier Based Exploration: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1806.03581
- Dilation Algorithm: https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm
- Greedy Algorithm: https://en.wikipedia.org/wiki/Greedy_algorithm
- Bresenham's Line Algorithm: https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
- Mapping, localization and planning: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://gaoyichao.com/Xiaotu/resource/refs/PR.MIT.en.pdf

Technical reference for integration and structures:

- ROS Frontier Exploration: https://wiki.ros.org/frontier_exploration
- ROS Explore Lite: https://wiki.ros.org/explore_lite
- VOSK: https://alphacephei.com/vosk/
- ROS Full Coverage Path Planning: https://wiki.ros.org/full_coverage_path_planner
- Turtlebot3: https://emanual.robotis.com/
- OpenCV: https://opencv.org/
