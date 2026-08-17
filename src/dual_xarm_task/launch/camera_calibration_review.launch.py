#!/usr/bin/env python3
"""启动带 ChArUco 棋盘的固定相机标定复核场景。"""

from typing import List

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    task_share = FindPackageShare("dual_xarm_task")
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                task_share,
                "launch",
                "dual_xarm_table_gazebo.launch.py",
            ])
        ),
        launch_arguments={
            "show_rviz": "false",
            "spawn_charuco_board": "true",
            "charuco_board_xyz": "0.0 0.30 1.42",
            "charuco_board_rpy": "0.0 0.0 0.0",
            "bridge_camera_pointcloud": "true",
            "use_camera_preset": "true",
            "gz_type": "ignition",
        }.items(),
    )

    checker = Node(
        package="dual_xarm_task",
        executable="camera_calibration_check.py",
        name="camera_calibration_check",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "detect_charuco": True,
                "duration_sec": ParameterValue(
                    LaunchConfiguration("duration_sec"), value_type=float
                ),
                "report_path": LaunchConfiguration("report_path"),
                "charuco_report_path": LaunchConfiguration("charuco_report_path"),
                "charuco_min_corners": ParameterValue(
                    LaunchConfiguration("charuco_min_corners"), value_type=int
                ),
                "marker_world_xyz": ParameterValue(
                    LaunchConfiguration("marker_world_xyz"), value_type=List[float]
                ),
                "marker_world_rpy": ParameterValue(
                    LaunchConfiguration("marker_world_rpy"), value_type=List[float]
                ),
                "depth_info_topic": LaunchConfiguration("depth_info_topic"),
            }
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("duration_sec", default_value="30.0"),
        DeclareLaunchArgument(
            "report_path",
            default_value="/tmp/dual_xarm_camera_calibration_review.json",
        ),
        DeclareLaunchArgument(
            "charuco_report_path", default_value="/tmp/charuco_calibration_review.png"
        ),
        DeclareLaunchArgument("charuco_min_corners", default_value="6"),
        DeclareLaunchArgument(
            "marker_world_xyz", default_value="[0.0, 0.2935, 1.42]"
        ),
        DeclareLaunchArgument(
            "marker_world_rpy", default_value="[1.5707963, 0.0, 0.0]"
        ),
        DeclareLaunchArgument(
            "depth_info_topic", default_value="/camera/depth/camera_info"
        ),
        simulation_launch,
        TimerAction(period=12.0, actions=[checker]),
    ])
