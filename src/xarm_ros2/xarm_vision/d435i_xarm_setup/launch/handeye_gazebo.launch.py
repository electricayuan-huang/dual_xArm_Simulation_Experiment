#!/usr/bin/env python3

"""Launch the image-based hand-eye calibration chain in Gazebo."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    dof = LaunchConfiguration('dof')
    robot_type = LaunchConfiguration('robot_type')
    gz_type = LaunchConfiguration('gz_type')
    marker_size = LaunchConfiguration('marker_size')
    marker_id = LaunchConfiguration('marker_id')
    marker_x = LaunchConfiguration('marker_x')
    marker_y = LaunchConfiguration('marker_y')
    marker_z = LaunchConfiguration('marker_z')
    marker_roll = LaunchConfiguration('marker_roll')
    marker_pitch = LaunchConfiguration('marker_pitch')
    marker_yaw = LaunchConfiguration('marker_yaw')
    spawn_marker = LaunchConfiguration('spawn_marker')
    robot_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('xarm_moveit_config'),
            'launch',
            '_robot_moveit_gazebo.launch.py',
        ])),
        launch_arguments={
            'dof': dof,
            'robot_type': robot_type,
            'gz_type': gz_type,
            'add_gripper': 'true',
            'add_realsense_d435i': 'true',
            'add_d435i_links': 'true',
            'no_gui_ctrl': 'false',
        }.items(),
    )

    aruco_detector = Node(
        package='aruco_ros',
        executable='single',
        name='aruco_single',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'image_is_rectified': True,
            'marker_size': marker_size,
            'marker_id': marker_id,
            'reference_frame': 'camera_color_optical_frame',
            'camera_frame': 'camera_color_optical_frame',
            'marker_frame': 'camera_marker',
        }],
        remappings=[
            ('/camera_info', '/camera/color/camera_info'),
            ('/image', '/camera/color/image_raw'),
        ],
    )

    recognition_view = Node(
        package='image_view',
        executable='image_view',
        name='aruco_result_view',
        output='screen',
        remappings=[('/image', '/aruco_single/result')],
    )

    marker_file = PathJoinSubstitution([
        FindPackageShare('xarm_planner'),
        'worlds',
        'aruco_marker_board.sdf',
    ])
    marker_spawn_action = TimerAction(
        period=5.0,
        actions=[Node(
            package='ros_gz_sim',
            executable='create',
            name='spawn_aruco_marker',
            output='screen',
            arguments=[
                '-file', marker_file,
                '-name', 'aruco_marker',
                '-x', marker_x,
                '-y', marker_y,
                '-z', marker_z,
                '-X', marker_roll,
                '-Y', marker_pitch,
                '-Z', marker_yaw,
            ],
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(spawn_marker),
        )],
    )

    return LaunchDescription([
        DeclareLaunchArgument('dof', default_value='6'),
        DeclareLaunchArgument('robot_type', default_value='xarm'),
        DeclareLaunchArgument('gz_type', default_value='gz'),
        DeclareLaunchArgument('marker_size', default_value='0.15'),
        DeclareLaunchArgument('marker_id', default_value='0'),
        DeclareLaunchArgument('marker_x', default_value='-0.17'),
        DeclareLaunchArgument('marker_y', default_value='-0.83'),
        DeclareLaunchArgument('marker_z', default_value='1.016'),
        DeclareLaunchArgument('marker_roll', default_value='0'),
        DeclareLaunchArgument('marker_pitch', default_value='0'),
        DeclareLaunchArgument('marker_yaw', default_value='0'),
        DeclareLaunchArgument('spawn_marker', default_value='true'),
        robot_gazebo,
        aruco_detector,
        recognition_view,
        marker_spawn_action,
    ])
