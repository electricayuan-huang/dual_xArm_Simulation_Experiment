from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("duration_sec", default_value="10.0"),
        DeclareLaunchArgument(
            "report_path",
            default_value="/tmp/dual_xarm_camera_calibration_report.json",
        ),
        DeclareLaunchArgument("detect_aruco", default_value="false"),
        DeclareLaunchArgument("marker_id", default_value="0"),
        DeclareLaunchArgument("marker_size_m", default_value="0.0"),
        DeclareLaunchArgument("detect_charuco", default_value="false"),
        DeclareLaunchArgument("charuco_min_corners", default_value="6"),
        DeclareLaunchArgument(
            "depth_info_topic", default_value="/camera/depth/camera_info"
        ),
        DeclareLaunchArgument(
            "marker_world_xyz", default_value="[0.0, 0.2935, 1.42]"
        ),
        DeclareLaunchArgument(
            "marker_world_rpy", default_value="[1.5707963, 0.0, 0.0]"
        ),
        DeclareLaunchArgument(
            "charuco_report_path", default_value="/tmp/charuco_detection.png"
        ),
        Node(
            package="dual_xarm_task",
            executable="camera_calibration_check.py",
            name="camera_calibration_check",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "duration_sec": ParameterValue(
                    LaunchConfiguration("duration_sec"), value_type=float
                ),
                "report_path": LaunchConfiguration("report_path"),
                "detect_aruco": ParameterValue(
                    LaunchConfiguration("detect_aruco"), value_type=bool
                ),
                "marker_id": ParameterValue(
                    LaunchConfiguration("marker_id"), value_type=int
                ),
                "marker_size_m": ParameterValue(
                    LaunchConfiguration("marker_size_m"), value_type=float
                ),
                "detect_charuco": ParameterValue(
                    LaunchConfiguration("detect_charuco"), value_type=bool
                ),
                "charuco_min_corners": ParameterValue(
                    LaunchConfiguration("charuco_min_corners"), value_type=int
                ),
                "depth_info_topic": LaunchConfiguration("depth_info_topic"),
                "marker_world_xyz": ParameterValue(
                    LaunchConfiguration("marker_world_xyz"), value_type=List[float]
                ),
                "marker_world_rpy": ParameterValue(
                    LaunchConfiguration("marker_world_rpy"), value_type=List[float]
                ),
                "charuco_report_path": LaunchConfiguration("charuco_report_path"),
            }],
        ),
    ])
