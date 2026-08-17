#!/usr/bin/env python3
"""Start the perception-only RGB-D cylinder pose estimator."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("duration_sec", default_value="0.0"),
        DeclareLaunchArgument(
            "report_path", default_value="/tmp/cylinder_pose_report.json"
        ),
        DeclareLaunchArgument("output_frame", default_value="world"),
        DeclareLaunchArgument("publish_debug_image", default_value="true"),
        DeclareLaunchArgument("max_sync_delta_sec", default_value="0.10"),
        DeclareLaunchArgument("lock_pose", default_value="false"),
        DeclareLaunchArgument("lock_after_estimates", default_value="10"),
        Node(
            package="dual_xarm_task",
            executable="cylinder_pose_estimator.py",
            name="cylinder_pose_estimator",
            output="screen",
            parameters=[
                {
                    "use_sim_time": True,
                    "duration_sec": ParameterValue(
                        LaunchConfiguration("duration_sec"), value_type=float
                    ),
                    "report_path": LaunchConfiguration("report_path"),
                    "output_frame": LaunchConfiguration("output_frame"),
                    "publish_debug_image": ParameterValue(
                        LaunchConfiguration("publish_debug_image"), value_type=bool
                    ),
                    "max_sync_delta_sec": ParameterValue(
                        LaunchConfiguration("max_sync_delta_sec"), value_type=float
                    ),
                    "lock_pose": ParameterValue(
                        LaunchConfiguration("lock_pose"), value_type=bool
                    ),
                    "lock_after_estimates": ParameterValue(
                        LaunchConfiguration("lock_after_estimates"), value_type=int
                    ),
                }
            ],
        ),
    ])
