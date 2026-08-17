import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, OpaqueFunction, TimerAction, LogInfo, ExecuteProcess,
    RegisterEventHandler, Shutdown, EmitEvent,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown as ShutdownEvent
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder
from uf_ros_lib.uf_robot_utils import generate_ros2_control_params_temp_file


def launch_setup(context, *args, **kwargs):
    dof = LaunchConfiguration("dof").perform(context)
    robot_type = LaunchConfiguration("robot_type").perform(context)
    prefix = LaunchConfiguration("prefix").perform(context)
    gz_type = LaunchConfiguration("gz_type").perform(context)

    ros2_control_params = generate_ros2_control_params_temp_file(
        os.path.join(get_package_share_directory("xarm_controller"), "config",
            "{}{}_controllers.yaml".format(robot_type,
                dof if robot_type in ("xarm", "lite") else "")),
        prefix=prefix, add_gripper=True, add_bio_gripper=False,
        ros_namespace="", update_rate=1000, use_sim_time=True,
        robot_type=robot_type)

    moveit_config = MoveItConfigsBuilder(
        context=context, controllers_name="fake_controllers",
        dof=LaunchConfiguration("dof"), robot_type=LaunchConfiguration("robot_type"),
        prefix=LaunchConfiguration("prefix"), hw_ns="xarm",
        limited=False, effort_control=False, velocity_control=False,
        model1300=False, robot_sn="", attach_to="world",
        attach_xyz='"-0.2 -0.5 1.021"', attach_rpy='"0 0 -1.571"',
        mesh_suffix="stl", kinematics_suffix="",
        ros2_control_plugin="ign_ros2_control/IgnitionSystem",
        ros2_control_params=ros2_control_params, gripper_version="G1",
        add_gripper=LaunchConfiguration("add_gripper"),
        add_vacuum_gripper=False, add_bio_gripper=False,
        add_realsense_d435i=LaunchConfiguration("add_realsense_d435i"),
        add_d435i_links=True, add_other_geometry=False,
        geometry_type="box", geometry_mass=0.1, geometry_height=0.1,
        geometry_radius=0.1, geometry_length=0.1, geometry_width=0.1,
        geometry_mesh_filename="",
        geometry_mesh_origin_xyz='"0 0 0"', geometry_mesh_origin_rpy='"0 0 0"',
        geometry_mesh_tcp_xyz='"0 0 0"', geometry_mesh_tcp_rpy='"0 0 0"',
    ).to_moveit_configs()

    pkg_share = get_package_share_directory("xarm_planner")

    calib_file = os.path.expanduser("~/.ros2/handeye_img_samples_result.yaml")
    handeye_tf_publisher = ExecuteProcess(
        cmd=[
            "python3",
            os.path.join(pkg_share, "..", "..", "lib", "xarm_planner", "publish_handeye_tf.py"),
            "--ros-args",
            "-p", "use_sim_time:=True",
            "-p", "calib_file:=" + calib_file,
        ],
        output="screen",
        name="handeye_tf_publisher",
    )
    color_to_depth_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="color_to_depth_optical_tf",
        output="screen",
        arguments=[
            "--x", "0.015", "--y", "0.0", "--z", "0.0",
            "--roll", "0.0", "--pitch", "0.0", "--yaw", "0.0",
            "--frame-id", "camera_color_optical_frame",
            "--child-frame-id", "camera_depth_optical_frame",
        ],
        parameters=[{"use_sim_time": True}],
    )

    # --- Detector ---
    detector_cmd = ExecuteProcess(
        cmd=["python3", os.path.join(pkg_share, "..", "..", "lib", "xarm_planner", "opencv_detector.py"),
             "--ros-args", "-p", "use_sim_time:=True"],
        output="screen", name="color_detector")

    # --- Pickup node ---
    params = moveit_config.to_dict()
    params.update({
        "arm_group": prefix + "xarm" + dof,
        "grip_group": prefix + "xarm_gripper",
        "base_frame": prefix + "link_base",
        "eef_frame": prefix + "link_eef",
        "planning_eef_link": prefix + "link_tcp",
        "pre_z_offset": 0.10,           # 预抓取高度：物体上方多少米
        "lift_z_offset": 0.18,           # 夹取后抬升高度：运输前更高抬升
        "grasp_z_offset": 0.0,          # 参考 Python：检测点直接作为抓取高度
        # 放置区坐标，不是物体坐标；物体坐标全部来自 TF 检测。
        "place_z": 0.04,                 # 当前 table_gz.world 桌面相对高度
        "gripper_close": 0.55,          # 夹爪闭合位置 (0=全开, 1=全闭)
        "gripper_open": 0.0,            # 夹爪张开位置
        "velocity_scale": 0.8,          # 运动速度缩放 (0~1, 越小越慢)
        "home_joints": [0.0, 0.0, 0.0, 0.0, -1.5708, 0.0],  # HOME 关节角(rad)
        # 已验证可观察 table_gz.world 物体的姿态：
        # j1=0°, j2=-57°, j3=-14°, j4=0°, j5=74°, j6=0°
        "observe_joints": [0.0, -0.994838, -0.244346, 0.0, 1.291544, 0.0],
        "gripper_joints": [             # 夹爪关节名称列表
            "drive_joint", "left_finger_joint", "left_inner_knuckle_joint",
            "right_outer_knuckle_joint", "right_finger_joint", "right_inner_knuckle_joint",
        ],
        "use_sim_time": True,
    })

    pickup_node = Node(
        package="xarm_planner",
        executable="xarm_pickup_node",
        name="xarm_pickup_node",
        output="screen",
        emulate_tty=True,
        parameters=[params],
    )

    return [
        handeye_tf_publisher,
        color_to_depth_tf,
        TimerAction(period=3.0, actions=[LogInfo(msg="=== Starting detector ==="), detector_cmd]),
        TimerAction(period=8.0, actions=[LogInfo(msg="=== Starting pickup ==="), pickup_node]),
        RegisterEventHandler(
            OnProcessExit(
                target_action=pickup_node,
                on_exit=[LogInfo(msg="=== Pickup node exited, shutting down ==="),
                         EmitEvent(event=ShutdownEvent(reason="pick-and-place finished"))],
            )
        ),
    ]


def generate_launch_description():
    ld = LaunchDescription()
    for a, v in [("dof", "6"), ("robot_type", "xarm"), ("prefix", ""),
                 ("add_gripper", "true"), ("add_realsense_d435i", "true"),
                 ("gz_type", "ignition")]:
        ld.add_action(DeclareLaunchArgument(a, default_value=v))
    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld
