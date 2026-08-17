# dual_xArm_Simulation_Experiment

This project is extended from the open‑source xArm simulated manipulator, implementing a dual‑arm cooperative visual positioning and peg‑in‑hole insertion experiment.

基于 ROS 2 Humble、Ignition Gazebo Fortress 和 MoveIt 2 的双 xArm RGB-D 视觉引导仿真工程。

当前流程包含：R_ 夹取空心圆柱、举升翻转、RGB-D 入口和孔轴估计、L_ 长杆 70mm 低速 Cartesian 插入、轴向退出以及 READY/SAFE 回位。

## 当前状态

- R_ 夹取、attachment、举升和翻转已通过 Gazebo 验收。
- RGB-D 圆柱估计已通过验证，默认 RGB/depth 同步阈值为 `0.10s`。
- L_ 分阶段规划、低速 Cartesian 插入、轴向退出和 READY/SAFE 回位已通过 Gazebo 仿真。
- 当前插入验收终点为 `70mm`，不再要求 140mm 穿出。
- 待完成：在程序报告中输出正式的 `SUCCESS`/`FAILURE` 自动判定。

## Source Layout

- `src/dual_xarm_task/`：任务包、Gazebo 世界、URDF、MoveIt 配置、相机桥接和视觉节点。
- `src/dual_xarm_planner/L_xarm_into.py`：L_ 分阶段插入、Cartesian 退出和回位脚本。
- `src/dual_xarm_planner/R_xarm_pickup.py`：R_ 夹取、举升和翻转脚本。
- `src/test_file/`：固定相机调试器和已验收相机预设。
- `src/xarm_ros2/`：xArm 控制器、描述、Gazebo、MoveIt 和 SDK 依赖。
- `src/realsense-ros/`：RealSense 描述和相关 ROS 依赖。
- `src/aruco_ros/`：ArUco 消息和检测依赖。
- `src/easy_handeye2/`：手眼标定工具和消息依赖。

`dual_xarm_planner` 不是独立 ROS 包，两个规划脚本通过 `python3` 直接运行。

## Requirements

- Ubuntu 22.04
- ROS 2 Humble
- Ignition Gazebo Fortress
- MoveIt 2
- Python 3、NumPy、OpenCV、`cv_bridge`

## Build

```bash
cd /home/ayuan/dual_xarm_ros2_ws
source /opt/ros/humble/setup.bash

colcon build --symlink-install \
  --packages-select dual_xarm_task xarm_description xarm_controller uf_ros_lib xarm_moveit_config

source install/setup.bash
```

## Run

先启动主仿真，等待 Gazebo、MoveIt 和所有控制器进入 `active` 状态：

```bash
source /opt/ros/humble/setup.bash
source /home/ayuan/dual_xarm_ros2_ws/install/setup.bash

ros2 launch dual_xarm_task dual_xarm_table_gazebo.launch.py \
  gz_type:=ignition \
  show_rviz:=false
```

检查控制器：

```bash
ros2 control list_controllers
ros2 topic echo /joint_states --once
```

按以下顺序执行 R_、视觉和 L_。不要在 R_ 运动期间启动视觉估计器：

```bash
cd /home/ayuan/dual_xarm_ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

python3 src/dual_xarm_planner/R_xarm_pickup.py \
  --execute \
  --confirm-execute \
  --pause-after 1.0
```

R_ 完成并显示 `Workpiece attachment confirmed` 后，启动锁定位姿视觉：

```bash
cd /home/ayuan/dual_xarm_ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch dual_xarm_task cylinder_pose_estimator.launch.py \
  duration_sec:=0.0 \
  lock_pose:=true \
  lock_after_estimates:=10 \
  max_sync_delta_sec:=0.10 \
  report_path:=/tmp/cylinder_pose_report.json
```

视觉日志出现 `圆柱位姿已锁定` 后，执行完整 L_ 流程：

```bash
cd /home/ayuan/dual_xarm_ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

python3 src/dual_xarm_planner/L_xarm_into.py \
  --phase through \
  --execute \
  --confirm-execute \
  --report-path /tmp/l_xarm_70mm_withdraw_return.json
```

`--phase through` 的当前含义是 70mm 插入、分段 Cartesian 退出、READY 回位和 SAFE 回位。`--phase insert70` 只执行到 70mm，不执行退出和回位。

## Safety Gates

- 插入前必须通过 `3mm/2deg` 严格位姿门控。
- Cartesian 路径必须满足 `fraction >= 0.98`。
- 相邻轨迹点最大关节步长为 `0.10rad`。
- 执行前检查 `/joint_states` 新鲜度和轨迹起点。
- 执行前通过 FK 检查 Cartesian 终点。
- 插入完成后必须先沿孔轴退出到入口外安全距离，再进行关节空间回位。
- 任何门控失败都停止，不盲目执行下一段。

## Generated Files

以下内容不应提交到 GitHub：

```text
build/
install/
log/
workspace_cache_before_rename_*/
**/__pycache__/
*.pyc
.ros/
```
