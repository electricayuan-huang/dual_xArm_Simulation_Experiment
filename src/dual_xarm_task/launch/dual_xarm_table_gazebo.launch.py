#!/usr/bin/env python3

import os
import json

from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from uf_ros_lib.moveit_configs_builder import DualMoveItConfigsBuilder
from uf_ros_lib.uf_robot_utils import (
    generate_dual_ros2_control_params_temp_file,
    get_xacro_content,
)


def launch_setup(context, *args, **kwargs):
    dof_1 = LaunchConfiguration("dof_1")
    dof_2 = LaunchConfiguration("dof_2")
    robot_type_1 = LaunchConfiguration("robot_type_1")
    robot_type_2 = LaunchConfiguration("robot_type_2")
    prefix_1 = LaunchConfiguration("prefix_1")
    prefix_2 = LaunchConfiguration("prefix_2")
    hw_ns = LaunchConfiguration("hw_ns")
    limited = LaunchConfiguration("limited")
    add_gripper_1 = LaunchConfiguration("add_gripper_1")
    add_gripper_2 = LaunchConfiguration("add_gripper_2")
    add_other_geometry_1 = LaunchConfiguration("add_other_geometry_1")
    add_other_geometry_2 = LaunchConfiguration("add_other_geometry_2")
    geometry_type_1 = LaunchConfiguration("geometry_type_1")
    geometry_type_2 = LaunchConfiguration("geometry_type_2")
    geometry_mass_1 = LaunchConfiguration("geometry_mass_1")
    geometry_mass_2 = LaunchConfiguration("geometry_mass_2")
    geometry_height_1 = LaunchConfiguration("geometry_height_1")
    geometry_height_2 = LaunchConfiguration("geometry_height_2")
    geometry_radius_1 = LaunchConfiguration("geometry_radius_1")
    geometry_radius_2 = LaunchConfiguration("geometry_radius_2")
    geometry_length_1 = LaunchConfiguration("geometry_length_1")
    geometry_length_2 = LaunchConfiguration("geometry_length_2")
    geometry_width_1 = LaunchConfiguration("geometry_width_1")
    geometry_width_2 = LaunchConfiguration("geometry_width_2")
    base_xyz_1 = LaunchConfiguration("base_xyz_1")
    base_xyz_2 = LaunchConfiguration("base_xyz_2")
    base_rpy_1 = LaunchConfiguration("base_rpy_1")
    base_rpy_2 = LaunchConfiguration("base_rpy_2")
    add_fixed_camera = LaunchConfiguration("add_fixed_camera")
    fixed_camera_xyz = LaunchConfiguration("fixed_camera_xyz")
    fixed_camera_rpy = LaunchConfiguration("fixed_camera_rpy")
    fixed_camera_horizontal_fov = LaunchConfiguration("fixed_camera_horizontal_fov")
    use_camera_preset = LaunchConfiguration("use_camera_preset")
    camera_preset_file = LaunchConfiguration("camera_preset_file")
    fixed_camera_visualize = LaunchConfiguration("fixed_camera_visualize")
    bridge_camera_pointcloud = LaunchConfiguration("bridge_camera_pointcloud")
    gazebo_self_collide = LaunchConfiguration("gazebo_self_collide")
    show_rviz = LaunchConfiguration("show_rviz")
    rmw_implementation = LaunchConfiguration("rmw_implementation")
    robot_state_publisher_start_delay = LaunchConfiguration(
        "robot_state_publisher_start_delay"
    )
    world_file = LaunchConfiguration("world_file")
    spawn_charuco_board = LaunchConfiguration("spawn_charuco_board")
    charuco_board_xyz = LaunchConfiguration("charuco_board_xyz")
    charuco_board_rpy = LaunchConfiguration("charuco_board_rpy")
    gz_type = LaunchConfiguration("gz_type")

    dof_1_value = dof_1.perform(context)
    dof_2_value = dof_2.perform(context)
    robot_type_1_value = robot_type_1.perform(context)
    robot_type_2_value = robot_type_2.perform(context)
    prefix_1_value = prefix_1.perform(context)
    prefix_2_value = prefix_2.perform(context)
    add_gripper_1_value = add_gripper_1.perform(context).lower() == "true"
    add_gripper_2_value = add_gripper_2.perform(context).lower() == "true"
    gz_type_value = gz_type.perform(context).lower()
    fixed_camera_visualize_value = fixed_camera_visualize.perform(context)
    bridge_camera_pointcloud_value = bridge_camera_pointcloud.perform(context).lower() == "true"
    gazebo_self_collide_value = gazebo_self_collide.perform(context)
    spawn_charuco_board_value = spawn_charuco_board.perform(context).lower() == "true"
    if gz_type_value == "ign":
        gz_type_value = "ignition"

    ros2_control_plugin = {
        # Ignition Fortress and Gazebo Sim use the official GazeboSimSystem
        # backend. The gripper uses six explicitly commanded position joints;
        # native URDF mimic is disabled to avoid competing commands.
        "ignition": "gz_ros2_control/GazeboSimSystem",
        "gz": "gz_ros2_control/GazeboSimSystem",
        "gazebo": "gazebo_ros2_control/GazeboSystem",
    }.get(gz_type_value)
    if ros2_control_plugin is None:
        raise RuntimeError(
            "Unsupported gz_type '{}'; use ignition, gz, or gazebo".format(
                gz_type_value
            )
        )

    # The confirmed tuner preset is used for normal launches. Explicit tuner
    # launches disable this with use_camera_preset:=false.
    fixed_camera_xyz_value = fixed_camera_xyz.perform(context)
    fixed_camera_rpy_value = fixed_camera_rpy.perform(context)
    fixed_camera_horizontal_fov_value = fixed_camera_horizontal_fov.perform(context)
    if use_camera_preset.perform(context).lower() == "true":
        preset_path = camera_preset_file.perform(context)
        try:
            with open(preset_path, "r", encoding="utf-8") as preset_stream:
                preset = json.load(preset_stream)
            fixed_camera_xyz_value = preset.get("fixed_camera_xyz", fixed_camera_xyz_value)
            fixed_camera_rpy_value = preset.get("fixed_camera_rpy", fixed_camera_rpy_value)
            fixed_camera_horizontal_fov_value = preset.get(
                "fixed_camera_horizontal_fov",
                fixed_camera_horizontal_fov_value,
            )
            print(f"[dual_xarm_task] 已加载相机预设: {preset_path}")
        except FileNotFoundError:
            print(f"[dual_xarm_task] 未找到相机预设，使用 launch 参数: {preset_path}")
        except (OSError, ValueError) as exc:
            print(f"[dual_xarm_task] 相机预设读取失败，使用 launch 参数: {exc}")

    ros2_control_params = generate_dual_ros2_control_params_temp_file(
        os.path.join(
            get_package_share_directory("xarm_controller"),
            "config",
            f"{robot_type_1_value}{dof_1_value}_controllers.yaml",
        ),
        os.path.join(
            get_package_share_directory("xarm_controller"),
            "config",
            f"{robot_type_2_value}{dof_2_value}_controllers.yaml",
        ),
        prefix_1=prefix_1_value,
        prefix_2=prefix_2_value,
        add_gripper_1=add_gripper_1_value,
        add_gripper_2=add_gripper_2_value,
        ros_namespace="",
        # Keep the ros2_control update rate conservative for the gripper
        # follower position controllers.
        update_rate=100,
        use_sim_time=True,
        robot_type_1=robot_type_1_value,
        robot_type_2=robot_type_2_value,
    )

    moveit_config = DualMoveItConfigsBuilder(
        context=context,
        controllers_name="fake_controllers",
        dof_1=dof_1,
        dof_2=dof_2,
        robot_type_1=robot_type_1,
        robot_type_2=robot_type_2,
        prefix_1=prefix_1,
        prefix_2=prefix_2,
        hw_ns=hw_ns,
        limited=limited,
        ros2_control_plugin=ros2_control_plugin,
        ros2_control_params=ros2_control_params,
        add_gripper_1=add_gripper_1,
        add_gripper_2=add_gripper_2,
        add_other_geometry_1=add_other_geometry_1,
        add_other_geometry_2=add_other_geometry_2,
        geometry_type_1=geometry_type_1,
        geometry_type_2=geometry_type_2,
        geometry_mass_1=geometry_mass_1,
        geometry_mass_2=geometry_mass_2,
        geometry_height_1=geometry_height_1,
        geometry_height_2=geometry_height_2,
        geometry_radius_1=geometry_radius_1,
        geometry_radius_2=geometry_radius_2,
        geometry_length_1=geometry_length_1,
        geometry_length_2=geometry_length_2,
        geometry_width_1=geometry_width_1,
        geometry_width_2=geometry_width_2,
    ).to_moveit_configs()

    task_share = get_package_share_directory("dual_xarm_task")
    task_urdf = os.path.join(task_share, "urdf", "dual_xarm_table.urdf.xacro")
    robot_description = get_xacro_content(
        context,
        xacro_file=task_urdf,
        prefix_1=prefix_1,
        prefix_2=prefix_2,
        dof_1=dof_1,
        dof_2=dof_2,
        robot_type_1=robot_type_1,
        robot_type_2=robot_type_2,
        hw_ns=hw_ns,
        limited=limited,
        ros2_control_plugin=ros2_control_plugin,
        ros2_control_params=ros2_control_params,
        add_gripper_1=add_gripper_1,
        add_gripper_2=add_gripper_2,
        add_other_geometry_1=add_other_geometry_1,
        add_other_geometry_2=add_other_geometry_2,
        geometry_type_1=geometry_type_1,
        geometry_type_2=geometry_type_2,
        geometry_mass_1=geometry_mass_1,
        geometry_mass_2=geometry_mass_2,
        geometry_height_1=geometry_height_1,
        geometry_height_2=geometry_height_2,
        geometry_radius_1=geometry_radius_1,
        geometry_radius_2=geometry_radius_2,
        geometry_length_1=geometry_length_1,
        geometry_length_2=geometry_length_2,
        geometry_width_1=geometry_width_1,
        geometry_width_2=geometry_width_2,
        base_xyz_1=base_xyz_1,
        base_xyz_2=base_xyz_2,
        base_rpy_1=base_rpy_1,
        base_rpy_2=base_rpy_2,
        add_fixed_camera=add_fixed_camera,
        fixed_camera_xyz=fixed_camera_xyz_value,
        fixed_camera_rpy=fixed_camera_rpy_value,
        fixed_camera_horizontal_fov=fixed_camera_horizontal_fov_value,
        fixed_camera_visualize=fixed_camera_visualize_value,
        gazebo_self_collide=gazebo_self_collide_value,
    )
    moveit_config.robot_description = {"robot_description": robot_description}
    moveit_config_dict = moveit_config.to_dict()

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
    )

    world_file_value = world_file.perform(context)
    if gz_type_value == "ignition":
        gazebo_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("ros_ign_gazebo"),
                    "launch",
                    "ign_gazebo.launch.py",
                ])
            ),
            launch_arguments={
                "ign_args": "-r -v 3 {}".format(world_file_value),
            }.items(),
        )
        spawn_entity = Node(
            package="ros_ign_gazebo",
            executable="create",
            arguments=[
                "-topic",
                "robot_description",
                "-name",
                "dual_xarm_table_robot",
            ],
            output="screen",
        )
        clock_bridge = Node(
            package="ros_ign_bridge",
            executable="parameter_bridge",
            arguments=["/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock"],
            output="screen",
        )
    elif gz_type_value == "gz":
        gazebo_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("ros_gz_sim"),
                    "launch",
                    "gz_sim.launch.py",
                ])
            ),
            launch_arguments={
                "gz_args": "-r -v 3 {}".format(world_file_value),
            }.items(),
        )
        spawn_entity = Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-topic",
                "robot_description",
                "-name",
                "dual_xarm_table_robot",
            ],
            output="screen",
        )
        clock_bridge = Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
            output="screen",
        )
    else:
        gazebo_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("gazebo_ros"),
                    "launch",
                    "gazebo.launch.py",
                ])
            ),
            launch_arguments={
                "world": world_file,
                "verbose": "true",
            }.items(),
        )
        spawn_entity = Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            arguments=[
                "-topic",
                "robot_description",
                "-entity",
                "dual_xarm_table_robot",
            ],
            output="screen",
        )
        clock_bridge = None

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config_dict, {"use_sim_time": True}],
    )

    rviz_config = PathJoinSubstitution([
        FindPackageShare("dual_xarm_task"),
        "rviz",
        "dual_xarm_table.rviz",
    ])
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[moveit_config_dict, {"use_sim_time": True}],
        condition=IfCondition(show_rviz),
    )

    robot_state_publisher_start_delay_value = float(
        robot_state_publisher_start_delay.perform(context)
    )
    robot_launch_after_state_publisher = TimerAction(
        period=robot_state_publisher_start_delay_value,
        actions=[gazebo_launch, spawn_entity],
    )

    charuco_spawn_timer = None
    if spawn_charuco_board_value:
        charuco_xyz_value = charuco_board_xyz.perform(context).split()
        charuco_rpy_value = charuco_board_rpy.perform(context).split()
        if len(charuco_xyz_value) != 3 or len(charuco_rpy_value) != 3:
            raise RuntimeError(
                "charuco_board_xyz 和 charuco_board_rpy 必须各包含三个数值"
            )
        charuco_model_file = os.path.join(
            task_share, "models", "charuco_calibration_board", "model.sdf"
        )
        create_arguments = [
            "-file",
            charuco_model_file,
            "-name",
            "charuco_calibration_board",
            "-x",
            charuco_xyz_value[0],
            "-y",
            charuco_xyz_value[1],
            "-z",
            charuco_xyz_value[2],
            "-R",
            charuco_rpy_value[0],
            "-P",
            charuco_rpy_value[1],
            "-Y",
            charuco_rpy_value[2],
        ]
        if gz_type_value == "ignition":
            charuco_spawn = Node(
                package="ros_ign_gazebo",
                executable="create",
                arguments=create_arguments,
                output="screen",
            )
        elif gz_type_value == "gz":
            charuco_spawn = Node(
                package="ros_gz_sim",
                executable="create",
                arguments=create_arguments,
                output="screen",
            )
        else:
            charuco_spawn = Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
                arguments=[
                    "-entity",
                    "charuco_calibration_board",
                    "-file",
                    charuco_model_file,
                    "-x",
                    charuco_xyz_value[0],
                    "-y",
                    charuco_xyz_value[1],
                    "-z",
                    charuco_xyz_value[2],
                    "-R",
                    charuco_rpy_value[0],
                    "-P",
                    charuco_rpy_value[1],
                    "-Y",
                    charuco_rpy_value[2],
                ],
                output="screen",
            )
        # The world must be running before the create service is available.
        charuco_spawn_timer = TimerAction(
            period=max(5.0, robot_state_publisher_start_delay_value + 5.0),
            actions=[charuco_spawn],
        )

    camera_bridge = None
    camera_bridge_arguments = [
        "/camera/color/image_raw@sensor_msgs/msg/Image@ignition.msgs.Image",
        "/camera/color/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo",
        "/camera/depth/image@sensor_msgs/msg/Image@ignition.msgs.Image",
        "/camera/depth/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo",
        "/camera/depth/depth_image@sensor_msgs/msg/Image@ignition.msgs.Image",
    ]
    if bridge_camera_pointcloud_value:
        camera_bridge_arguments.append(
            "/camera/depth/points@sensor_msgs/msg/PointCloud2@ignition.msgs.PointCloudPacked"
        )
    if gz_type_value == "ignition":
        camera_bridge = Node(
            package="ros_ign_bridge",
            executable="parameter_bridge",
            name="fixed_camera_bridge",
            output="screen",
            arguments=camera_bridge_arguments,
        )
    elif gz_type_value == "gz":
        camera_bridge_arguments = [
            argument.replace("@ignition.msgs.", "@gz.msgs.")
            for argument in camera_bridge_arguments
        ]
        camera_bridge = Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="fixed_camera_bridge",
            output="screen",
            arguments=camera_bridge_arguments,
        )

    controllers = [
        "joint_state_broadcaster",
        f"{prefix_1_value}{robot_type_1_value}{dof_1_value}_traj_controller",
        f"{prefix_2_value}{robot_type_2_value}{dof_2_value}_traj_controller",
    ]
    gripper_controller_base_names = [
        "xarm_gripper_traj_controller",
    ]
    if add_gripper_1_value:
        controllers.extend(
            f"{prefix_1_value}{name}" for name in gripper_controller_base_names
        )
    if add_gripper_2_value:
        controllers.extend(
            f"{prefix_2_value}{name}" for name in gripper_controller_base_names
        )

    controller_nodes = [
        Node(
            package="controller_manager",
            executable="spawner",
            output="screen",
            arguments=[controller, "--controller-manager", "/controller_manager"],
            parameters=[{"use_sim_time": True}],
        )
        for controller in controllers
    ]

    # Start controllers in a deterministic order. The joint-state broadcaster
    # must be active before MoveIt clients try to read the current state.
    controller_chain_events = [
        RegisterEventHandler(
            OnProcessExit(
                target_action=current_controller,
                on_exit=[next_controller],
            )
        )
        for current_controller, next_controller in zip(
            controller_nodes, controller_nodes[1:]
        )
    ]
    controller_chain_start = controller_nodes[0]
    controller_chain_finish = RegisterEventHandler(
        OnProcessExit(
            target_action=controller_nodes[-1],
            on_exit=[rviz],
        )
    )

    # Resolve model://realsense2_description/... URIs emitted by the D435i xacro.
    realsense_resource_path = PathJoinSubstitution([
        FindPackageShare("realsense2_description"),
        "..",
    ])

    return [
        SetEnvironmentVariable(
            name="GAZEBO_MODEL_PATH",
            value=[
                PathJoinSubstitution([FindPackageShare("dual_xarm_task"), "models"]),
                ":",
                realsense_resource_path,
                ":",
                EnvironmentVariable("GAZEBO_MODEL_PATH", default_value=""),
            ],
        ),
        SetEnvironmentVariable(
            name="IGN_GAZEBO_RESOURCE_PATH",
            value=[
                PathJoinSubstitution([FindPackageShare("dual_xarm_task"), "models"]),
                ":",
                realsense_resource_path,
                ":",
                EnvironmentVariable("IGN_GAZEBO_RESOURCE_PATH", default_value=""),
            ],
        ),
        SetEnvironmentVariable(
            name="GZ_SIM_RESOURCE_PATH",
            value=[
                PathJoinSubstitution([FindPackageShare("dual_xarm_task"), "models"]),
                ":",
                realsense_resource_path,
                ":",
                EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value=""),
            ],
        ),
        RegisterEventHandler(
            OnProcessStart(
                target_action=robot_state_publisher,
                on_start=robot_launch_after_state_publisher,
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=spawn_entity,
                on_exit=[controller_chain_start],
            )
        ),
        *controller_chain_events,
        controller_chain_finish,
        robot_state_publisher,
        move_group,
        *([charuco_spawn_timer] if charuco_spawn_timer is not None else []),
        *([camera_bridge] if camera_bridge is not None else []),
        *([clock_bridge] if clock_bridge is not None else []),
    ]


def generate_launch_description():
    task_share = FindPackageShare("dual_xarm_task")
    return LaunchDescription([
        DeclareLaunchArgument("dof_1", default_value="6"),
        DeclareLaunchArgument("dof_2", default_value="6"),
        DeclareLaunchArgument("robot_type_1", default_value="xarm"),
        DeclareLaunchArgument("robot_type_2", default_value="xarm"),
        DeclareLaunchArgument("prefix_1", default_value="L_"),
        DeclareLaunchArgument("prefix_2", default_value="R_"),
        DeclareLaunchArgument("hw_ns", default_value="xarm"),
        DeclareLaunchArgument("limited", default_value="true"),
        DeclareLaunchArgument("add_gripper_1", default_value="false"),
        DeclareLaunchArgument("add_gripper_2", default_value="true"),
        DeclareLaunchArgument("add_other_geometry_1", default_value="true"),
        DeclareLaunchArgument("add_other_geometry_2", default_value="false"),
        DeclareLaunchArgument("geometry_type_1", default_value="cylinder"),
        DeclareLaunchArgument("geometry_type_2", default_value="box"),
        DeclareLaunchArgument("geometry_mass_1", default_value="0.12"),
        DeclareLaunchArgument("geometry_mass_2", default_value="0.1"),
        DeclareLaunchArgument("geometry_height_1", default_value="0.14"),
        DeclareLaunchArgument("geometry_height_2", default_value="0.1"),
        DeclareLaunchArgument("geometry_radius_1", default_value="0.008"),
        DeclareLaunchArgument("geometry_radius_2", default_value="0.1"),
        DeclareLaunchArgument("geometry_length_1", default_value="0.14"),
        DeclareLaunchArgument("geometry_length_2", default_value="0.1"),
        DeclareLaunchArgument("geometry_width_1", default_value="0.02"),
        DeclareLaunchArgument("geometry_width_2", default_value="0.1"),
        DeclareLaunchArgument("base_xyz_1", default_value="0.25 -0.25 1.03"),
        DeclareLaunchArgument("base_xyz_2", default_value="-0.25 -0.25 1.03"),
        DeclareLaunchArgument("base_rpy_1", default_value="0 0 1.5708"),
        DeclareLaunchArgument("base_rpy_2", default_value="0 0 1.5708"),
        DeclareLaunchArgument("add_fixed_camera", default_value="true"),
        DeclareLaunchArgument("fixed_camera_xyz", default_value="0 -0.42 1.015"),
        DeclareLaunchArgument("fixed_camera_rpy", default_value="0 0.70 1.5708"),
         DeclareLaunchArgument("fixed_camera_horizontal_fov", default_value="1.57"),
         DeclareLaunchArgument("fixed_camera_visualize", default_value="true"),
         DeclareLaunchArgument(
             "bridge_camera_pointcloud",
             default_value="true",
             description="是否桥接 RGB-D 点云；仅图像/深度校正时可关闭以降低仿真负载",
         ),
         DeclareLaunchArgument(
             "gazebo_self_collide",
             default_value="false",
             description="Gazebo 中是否启用 xArm6 各 link 的自碰撞；MoveIt 仍使用 SRDF 碰撞检查",
         ),
        DeclareLaunchArgument("use_camera_preset", default_value="true"),
        DeclareLaunchArgument(
            "camera_preset_file",
            default_value=PathJoinSubstitution(
                [task_share, "test_file", "camera_view_tuner_preset.json"]
            ),
        ),
          DeclareLaunchArgument(
              "world_file",
             default_value=PathJoinSubstitution(
                 [task_share, "worlds", "dual_xarm_table_ignition.world"]
              ),
          ),
          DeclareLaunchArgument(
              "spawn_charuco_board",
              default_value="false",
              description="启动后在标定板世界坐标生成 ChArUco 棋盘",
          ),
          DeclareLaunchArgument("charuco_board_xyz", default_value="0.0 0.30 1.42"),
          DeclareLaunchArgument("charuco_board_rpy", default_value="0.0 0.0 0.0"),
          DeclareLaunchArgument("gz_type", default_value="ignition"),
        DeclareLaunchArgument("show_rviz", default_value="true"),
         DeclareLaunchArgument(
             "rmw_implementation",
             default_value="rmw_fastrtps_cpp",
             description="所有 launch 子进程使用的 ROS 2 RMW 实现",
         ),
        DeclareLaunchArgument(
            "robot_state_publisher_start_delay",
            default_value="2.0",
            description=(
                "robot_state_publisher 启动后等待其参数服务可用，再生成 Gazebo 实体"
            ),
        ),
        # Set transport before the OpaqueFunction creates any ROS 2 actions.
        # This avoids Fast DDS shared-memory port conflicts after prior runs.
        SetEnvironmentVariable(
            name="RMW_IMPLEMENTATION",
            value=LaunchConfiguration("rmw_implementation"),
        ),
        SetEnvironmentVariable(name="RMW_FASTRTPS_USE_SHM", value="0"),
        SetEnvironmentVariable(name="FASTDDS_BUILTIN_TRANSPORTS", value="UDPv4"),
        OpaqueFunction(function=launch_setup),
    ])
