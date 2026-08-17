#!/usr/bin/env python3
"""Staged L-arm planning and insertion entry point.

The arm is never sent directly from its current state to a Cartesian insertion
path. It first plans through safe and ready joint configurations, then uses a
MoveGroup pose plan to reach the visual pre-alignment pose. Cartesian motion is
available only after that pose has been executed and verified.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import PoseStamped
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    MotionPlanRequest,
    OrientationConstraint,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import GetCartesianPath, GetPositionFK
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time as RosTime
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float32
import tf2_ros


def quaternion_to_matrix(quaternion: list[float]) -> np.ndarray:
    x, y, z, w = [float(value) for value in quaternion]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_to_axis(quaternion: Any) -> np.ndarray:
    return quaternion_to_matrix(
        [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    )[:, 2]


def transform_to_matrix(transform: Any) -> np.ndarray:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_matrix(
        [rotation.x, rotation.y, rotation.z, rotation.w]
    )
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


def angle_between(first: np.ndarray, second: np.ndarray) -> float:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    cosine = max(-1.0, min(1.0, float(np.dot(first, second))))
    return math.degrees(math.acos(cosine))


def parse_joint_values(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if len(values) != 6:
        raise argparse.ArgumentTypeError("L_ 关节目标必须包含六个弧度值")
    return values


class LArmInto(Node):
    """Execute or plan the staged L-arm insertion sequence."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("l_xarm_into")
        self.args = args
        self._instance_lock_file = None
        self._active_goal_handle = None
        self.latest_pose: PoseStamped | None = None
        self.latest_confidence = 0.0
        self.latest_joint_state: JointState | None = None
        self.latest_joint_state_monotonic: float | None = None
        self.pose_samples: list[PoseStamped] = []
        self.last_pose_stamp: tuple[int, int] | None = None
        self.acquire_instance_lock()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.move_group_client = ActionClient(self, MoveGroup, "/move_action")
        self.execute_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/L_xarm6_traj_controller/follow_joint_trajectory",
        )
        self.cartesian_client = self.create_client(
            GetCartesianPath, "/compute_cartesian_path"
        )
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")
        self.create_subscription(
            PoseStamped, args.entrance_topic, self.pose_callback, 10
        )
        self.create_subscription(
            Float32, args.confidence_topic, self.confidence_callback, 10
        )
        self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )
        self.stages: list[dict[str, Any]] = []

    def acquire_instance_lock(self) -> None:
        path = Path(self.args.lock_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = path.open("a+")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.close()
            raise RuntimeError(
                f"已有 L_xarm_into 实例运行，锁文件：{path}"
            ) from exc
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        self._instance_lock_file = lock_file

    def close(self) -> None:
        if self._instance_lock_file is None:
            return
        fcntl.flock(self._instance_lock_file.fileno(), fcntl.LOCK_UN)
        self._instance_lock_file.close()
        self._instance_lock_file = None

    def pose_callback(self, message: PoseStamped) -> None:
        self.latest_pose = message
        key = (int(message.header.stamp.sec), int(message.header.stamp.nanosec))
        if key != self.last_pose_stamp:
            self.pose_samples.append(message)
            self.pose_samples = self.pose_samples[-self.args.samples :]
            self.last_pose_stamp = key

    def confidence_callback(self, message: Float32) -> None:
        self.latest_confidence = float(message.data)

    def joint_state_callback(self, message: JointState) -> None:
        self.latest_joint_state = message
        self.latest_joint_state_monotonic = time.monotonic()

    def current_joint_positions(self) -> dict[str, float]:
        if self.latest_joint_state is None:
            raise RuntimeError("尚未收到最新 /joint_states")
        if (
            self.latest_joint_state_monotonic is None
            or time.monotonic() - self.latest_joint_state_monotonic
            > self.args.joint_state_max_age_sec
        ):
            raise RuntimeError(
                "/joint_states 数据过期，拒绝发送机械臂目标："
                f"age={time.monotonic() - (self.latest_joint_state_monotonic or 0.0):.3f}s"
            )
        return {
            name: float(position)
            for name, position in zip(
                self.latest_joint_state.name, self.latest_joint_state.position
            )
        }

    def wait_for_visual_target(self) -> PoseStamped:
        deadline = time.monotonic() + self.args.wait_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if len(self.pose_samples) < self.args.samples:
                continue
            if self.latest_confidence < self.args.min_confidence:
                continue
            positions = np.asarray(
                [
                    [
                        sample.pose.position.x,
                        sample.pose.position.y,
                        sample.pose.position.z,
                    ]
                    for sample in self.pose_samples
                ],
                dtype=np.float64,
            )
            axes = np.asarray(
                [quaternion_to_axis(sample.pose.orientation) for sample in self.pose_samples],
                dtype=np.float64,
            )
            position_std = float(np.linalg.norm(np.std(positions, axis=0)))
            axis_std = max(angle_between(axes[-1], axis) for axis in axes)
            if (
                position_std <= self.args.max_position_std_m
                and axis_std <= self.args.max_axis_std_deg
            ):
                self.get_logger().info(
                    "视觉目标稳定：samples=%d，position_std=%.3fmm，axis_std=%.3fdeg，confidence=%.3f"
                    % (
                        len(self.pose_samples),
                        position_std * 1000.0,
                        axis_std,
                        self.latest_confidence,
                    )
                )
                return self.pose_samples[-1]
        raise RuntimeError("未获得稳定的圆柱入口视觉目标")

    def current_state(self) -> RobotState:
        self.current_joint_positions()
        state = RobotState()
        state.joint_state = deepcopy(self.latest_joint_state)
        # Provide the freshest joint values while letting MoveIt merge fixed
        # joints and other state components from its monitored scene.
        state.is_diff = True
        return state

    def target_pose(self, entrance: PoseStamped, distance_m: float) -> PoseStamped:
        if entrance.header.frame_id != self.args.planning_frame:
            raise RuntimeError(
                f"视觉 frame={entrance.header.frame_id}，"
                f"规划 frame={self.args.planning_frame} 不一致"
            )
        axis = quaternion_to_axis(entrance.pose.orientation)
        axis /= np.linalg.norm(axis)
        target = PoseStamped()
        target.header = entrance.header
        target.header.frame_id = self.args.planning_frame
        target.pose = deepcopy(entrance.pose)
        target.pose.position.x -= float(axis[0]) * distance_m
        target.pose.position.y -= float(axis[1]) * distance_m
        target.pose.position.z -= float(axis[2]) * distance_m
        return target

    def joint_goal(self, positions: list[float]) -> MoveGroup.Goal:
        request = MotionPlanRequest()
        request.group_name = self.args.group_name
        request.start_state = self.current_state()
        request.num_planning_attempts = self.args.planning_attempts
        request.allowed_planning_time = self.args.planning_time_sec
        request.max_velocity_scaling_factor = self.args.velocity_scaling
        request.max_acceleration_scaling_factor = self.args.acceleration_scaling
        constraints = Constraints()
        constraints.joint_constraints = [
            JointConstraint(
                joint_name=f"L_joint{index + 1}",
                position=float(value),
                tolerance_above=self.args.joint_tolerance_rad,
                tolerance_below=self.args.joint_tolerance_rad,
                weight=1.0,
            )
            for index, value in enumerate(positions)
        ]
        request.goal_constraints = [constraints]
        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options.plan_only = True
        goal.planning_options.replan = False
        return goal

    def pose_goal(self, target: PoseStamped) -> MoveGroup.Goal:
        position = PositionConstraint()
        position.header = target.header
        position.link_name = self.args.tool_frame
        position.weight = 1.0
        position.constraint_region = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self.args.pose_position_tolerance_m]
        position.constraint_region.primitives = [sphere]
        position.constraint_region.primitive_poses = [target.pose]

        orientation = OrientationConstraint()
        orientation.header = target.header
        orientation.link_name = self.args.tool_frame
        orientation.orientation = target.pose.orientation
        orientation.absolute_x_axis_tolerance = self.args.pose_orientation_tolerance_rad
        orientation.absolute_y_axis_tolerance = self.args.pose_orientation_tolerance_rad
        orientation.absolute_z_axis_tolerance = self.args.pose_orientation_tolerance_rad
        orientation.weight = 1.0

        request = MotionPlanRequest()
        request.group_name = self.args.group_name
        request.start_state = self.current_state()
        request.num_planning_attempts = self.args.planning_attempts
        request.allowed_planning_time = self.args.planning_time_sec
        request.max_velocity_scaling_factor = self.args.velocity_scaling
        request.max_acceleration_scaling_factor = self.args.acceleration_scaling
        constraints = Constraints()
        constraints.position_constraints = [position]
        constraints.orientation_constraints = [orientation]
        request.goal_constraints = [constraints]
        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options.plan_only = True
        goal.planning_options.replan = False
        return goal

    def plan_move(self, goal: MoveGroup.Goal, label: str) -> Any:
        if not self.move_group_client.wait_for_server(timeout_sec=15.0):
            raise RuntimeError("MoveGroup /move_action 不可用")
        future = self.move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=20.0)
        if not future.done() or future.result() is None or not future.result().accepted:
            raise RuntimeError(f"{label} MoveGroup 目标被拒绝或超时")
        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=self.args.planning_time_sec + 30.0
        )
        if not result_future.done() or result_future.result() is None:
            raise RuntimeError(f"{label} MoveGroup 结果超时")
        result = result_future.result().result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"{label} MoveGroup 规划失败：{result.error_code.val}")
        trajectory = result.planned_trajectory
        if not trajectory.joint_trajectory.points:
            raise RuntimeError(f"{label} 返回空轨迹")
        self.get_logger().info(
            "%s 规划通过：points=%d，planning_time=%.3fs"
            % (
                label,
                len(trajectory.joint_trajectory.points),
                result.planning_time,
            )
        )
        return trajectory

    def retime(self, trajectory: Any, distance_m: float) -> None:
        points = trajectory.joint_trajectory.points
        if len(points) < 2:
            return
        total_duration = max(
            self.args.cartesian_min_duration,
            distance_m / max(self.args.cartesian_speed_mps, 1e-6),
        )
        for index, point in enumerate(points):
            seconds = total_duration * index / float(len(points) - 1)
            whole = int(seconds)
            nanoseconds = int(round((seconds - whole) * 1_000_000_000))
            if nanoseconds >= 1_000_000_000:
                whole += 1
                nanoseconds -= 1_000_000_000
            point.time_from_start = DurationMsg(sec=whole, nanosec=nanoseconds)
            # The controller commands position only.  MoveIt's derivatives use
            # the original timing and would be inconsistent after re-timing.
            point.velocities = []
            point.accelerations = []
            point.effort = []

    def plan_cartesian(self, target: PoseStamped, distance_m: float, label: str) -> Any:
        if not self.cartesian_client.wait_for_service(timeout_sec=15.0):
            raise RuntimeError("/compute_cartesian_path 不可用")
        request = GetCartesianPath.Request()
        request.header = target.header
        request.start_state = self.current_state()
        start_positions = {
            name: float(position)
            for name, position in zip(
                request.start_state.joint_state.name,
                request.start_state.joint_state.position,
            )
        }
        request.group_name = self.args.group_name
        request.link_name = self.args.tool_frame
        request.waypoints = [target.pose]
        request.max_step = self.args.cartesian_step_m
        request.jump_threshold = self.args.jump_threshold_rad
        request.revolute_jump_threshold = self.args.jump_threshold_rad
        request.prismatic_jump_threshold = self.args.prismatic_jump_threshold_m
        request.avoid_collisions = True
        future = self.cartesian_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"{label} CartesianPath 超时")
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"{label} CartesianPath 失败：{response.error_code.val}")
        if response.fraction < self.args.min_fraction:
            raise RuntimeError(
                f"{label} fraction={response.fraction:.4f} < {self.args.min_fraction:.4f}"
            )
        trajectory = response.solution
        self.validate_cartesian_joint_steps(trajectory, label, start_positions)
        self.validate_cartesian_endpoint(trajectory, target, label)
        self.retime(trajectory, distance_m)
        self.get_logger().info(
            "%s CartesianPath 通过：fraction=%.4f，points=%d，duration=%.3fs"
            % (
                label,
                response.fraction,
                len(trajectory.joint_trajectory.points),
                self.trajectory_duration(trajectory),
            )
        )
        return trajectory

    def validate_cartesian_joint_steps(
        self,
        trajectory: Any,
        label: str,
        start_positions: dict[str, float],
    ) -> None:
        points = trajectory.joint_trajectory.points
        names = trajectory.joint_trajectory.joint_names
        if not points or not names:
            raise RuntimeError(f"{label} CartesianPath 缺少关节轨迹")
        if len(points[0].positions) != len(names):
            raise RuntimeError(f"{label} CartesianPath 起点关节数量不一致")
        missing = [name for name in names if name not in start_positions]
        if missing:
            raise RuntimeError(f"{label} CartesianPath 缺少起点关节：{missing}")
        first = np.asarray(points[0].positions, dtype=np.float64)
        start = np.asarray([start_positions[name] for name in names])
        start_jump = float(np.max(np.abs(first - start)))
        if start_jump > self.args.max_joint_step_rad:
            raise RuntimeError(
                f"{label} CartesianPath 起点关节跳变："
                f"{start_jump:.3f}rad > {self.args.max_joint_step_rad:.3f}rad"
            )
        for previous, current_point in zip(points, points[1:]):
            if len(current_point.positions) != len(names):
                raise RuntimeError(f"{label} CartesianPath 关节数量不一致")
            step = float(
                np.max(
                    np.abs(
                        np.asarray(current_point.positions, dtype=np.float64)
                        - np.asarray(previous.positions, dtype=np.float64)
                    )
                )
            )
            if step > self.args.max_joint_step_rad:
                raise RuntimeError(
                    f"{label} CartesianPath 相邻点关节跳变："
                    f"{step:.3f}rad > {self.args.max_joint_step_rad:.3f}rad"
                )

    def validate_cartesian_endpoint(
        self, trajectory: Any, target: PoseStamped, label: str
    ) -> None:
        if not self.fk_client.wait_for_service(timeout_sec=15.0):
            raise RuntimeError("/compute_fk 不可用，拒绝执行 Cartesian 轨迹")
        points = trajectory.joint_trajectory.points
        names = trajectory.joint_trajectory.joint_names
        state = self.current_state()
        state.is_diff = False
        state_names = list(state.joint_state.name)
        state_positions = list(state.joint_state.position)
        state_indices = {name: index for index, name in enumerate(state_names)}
        for name, position in zip(names, points[-1].positions):
            index = state_indices.get(name)
            if index is None:
                raise RuntimeError(f"{label} FK 缺少关节：{name}")
            state_positions[index] = float(position)
        state.joint_state.position = state_positions

        request = GetPositionFK.Request()
        request.header = target.header
        request.fk_link_names = [self.args.tool_frame]
        request.robot_state = state
        future = self.fk_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"{label} FK 终点校验超时")
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"{label} FK 终点校验失败：{response.error_code.val}")
        if not response.pose_stamped:
            raise RuntimeError(f"{label} FK 未返回 TCP 位姿")
        actual = response.pose_stamped[0]
        actual_position = np.asarray(
            [actual.pose.position.x, actual.pose.position.y, actual.pose.position.z],
            dtype=np.float64,
        )
        target_position = np.asarray(
            [target.pose.position.x, target.pose.position.y, target.pose.position.z],
            dtype=np.float64,
        )
        position_error = float(np.linalg.norm(actual_position - target_position))
        axis_error = angle_between(
            quaternion_to_axis(actual.pose.orientation),
            quaternion_to_axis(target.pose.orientation),
        )
        if (
            position_error > self.args.insert_position_tolerance_m
            or axis_error > self.args.insert_axis_tolerance_deg
        ):
            raise RuntimeError(
                f"{label} FK 终点偏差：position={position_error * 1000.0:.3f}mm，"
                f"axis={axis_error:.3f}deg"
            )
        self.get_logger().info(
            "%s FK 终点通过：position=%.3fmm，axis=%.3fdeg"
            % (label, position_error * 1000.0, axis_error)
        )

    @staticmethod
    def trajectory_duration(trajectory: Any) -> float:
        if not trajectory.joint_trajectory.points:
            return 0.0
        stamp = trajectory.joint_trajectory.points[-1].time_from_start
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def execute(self, trajectory: Any, label: str) -> None:
        self.validate_trajectory_start(trajectory, label)
        if not self.execute_client.wait_for_server(timeout_sec=15.0):
            raise RuntimeError("L_xarm6_traj_controller action 不可用")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = deepcopy(trajectory.joint_trajectory)
        goal.goal_time_tolerance = DurationMsg(sec=5, nanosec=0)
        future = self.execute_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=20.0)
        if not future.done() or future.result() is None or not future.result().accepted:
            raise RuntimeError(f"{label} 执行目标被拒绝或超时")
        self._active_goal_handle = future.result()
        try:
            result_future = self._active_goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=120.0)
            if not result_future.done() or result_future.result() is None:
                self.cancel_active_goal(label)
                raise RuntimeError(f"{label} 执行结果超时，已请求取消")
            result = result_future.result().result
            if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
                raise RuntimeError(f"{label} 执行失败：{result.error_code}")
            self.get_logger().info(f"{label} 执行完成")
        except BaseException:
            self.cancel_active_goal(label)
            raise
        finally:
            self._active_goal_handle = None

    def validate_trajectory_start(self, trajectory: Any, label: str) -> None:
        points = trajectory.joint_trajectory.points
        names = trajectory.joint_trajectory.joint_names
        if not points or len(points[0].positions) != len(names):
            raise RuntimeError(f"{label} 轨迹起点无效")
        current = self.current_joint_positions()
        missing = [name for name in names if name not in current]
        if missing:
            raise RuntimeError(f"{label} 轨迹缺少当前关节：{missing}")
        start = np.asarray(points[0].positions, dtype=np.float64)
        actual = np.asarray([current[name] for name in names], dtype=np.float64)
        start_jump = float(np.max(np.abs(start - actual)))
        if start_jump > self.args.max_joint_step_rad:
            raise RuntimeError(
                f"{label} 执行前关节状态已变化："
                f"{start_jump:.3f}rad > {self.args.max_joint_step_rad:.3f}rad"
            )

    def cancel_active_goal(self, label: str) -> None:
        goal_handle = self._active_goal_handle
        if goal_handle is None:
            return
        try:
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)
            if cancel_future.done():
                self.get_logger().warning(f"{label} action 已请求取消")
            else:
                self.get_logger().error(f"{label} action 取消请求超时")
        except Exception as exc:
            self.get_logger().error(f"{label} action 取消失败：{exc}")
        finally:
            self._active_goal_handle = None

    def verify(self, target: PoseStamped, label: str, strict: bool) -> dict[str, Any]:
        deadline = time.monotonic() + self.args.settle_sec
        latest: dict[str, Any] | None = None
        position_limit = (
            self.args.insert_position_tolerance_m
            if strict
            else self.args.prealign_position_tolerance_m
        )
        axis_limit = (
            self.args.insert_axis_tolerance_deg
            if strict
            else self.args.prealign_axis_tolerance_deg
        )
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                transform = self.tf_buffer.lookup_transform(
                    target.header.frame_id,
                    self.args.tool_frame,
                    RosTime(),
                    timeout=Duration(seconds=0.3),
                )
            except tf2_ros.TransformException:
                continue
            matrix = transform_to_matrix(transform)
            actual_position = matrix[:3, 3]
            target_position = np.asarray(
                [target.pose.position.x, target.pose.position.y, target.pose.position.z],
                dtype=np.float64,
            )
            actual_axis = matrix[:3, :3][:, 2]
            target_axis = quaternion_to_axis(target.pose.orientation)
            latest = {
                "position_error_m": float(np.linalg.norm(actual_position - target_position)),
                "axis_error_deg": float(angle_between(actual_axis, target_axis)),
                "actual_position_m": actual_position.tolist(),
                "target_position_m": target_position.tolist(),
                "actual_axis": actual_axis.tolist(),
                "target_axis": target_axis.tolist(),
            }
            if (
                latest["position_error_m"] <= position_limit
                and latest["axis_error_deg"] <= axis_limit
            ):
                self.get_logger().info(
                    "%s 验证通过：position=%.3fmm，axis=%.3fdeg"
                    % (label, latest["position_error_m"] * 1000.0, latest["axis_error_deg"])
                )
                return latest
        if latest is None:
            raise RuntimeError(f"{label} 执行后无法读取 TCP TF")
        raise RuntimeError(
            "%s 验证失败：position=%.3fmm，axis=%.3fdeg"
            % (label, latest["position_error_m"] * 1000.0, latest["axis_error_deg"])
        )

    def record(self, label: str, status: str, **data: Any) -> None:
        self.stages.append({"label": label, "status": status, **data})

    def run_stage_move(self, label: str, goal: MoveGroup.Goal) -> None:
        trajectory = self.plan_move(goal, label)
        if not self.args.execute:
            self.record(label, "planned", points=len(trajectory.joint_trajectory.points))
            return
        self.execute(trajectory, label)
        self.record(label, "executed", points=len(trajectory.joint_trajectory.points))

    def run_pose_stage(
        self, label: str, target: PoseStamped, distance_m: float, strict: bool
    ) -> None:
        trajectory = self.plan_cartesian(target, distance_m, label)
        if not self.args.execute:
            self.record(
                label,
                "planned",
                fraction=1.0,
                points=len(trajectory.joint_trajectory.points),
                duration_sec=self.trajectory_duration(trajectory),
            )
            return
        self.execute(trajectory, label)
        verification = self.verify(target, label, strict)
        self.record(
            label,
            "executed",
            fraction=1.0,
            points=len(trajectory.joint_trajectory.points),
            duration_sec=self.trajectory_duration(trajectory),
            verification=verification,
        )

    def run_pose_joint_stage(
        self, label: str, target: PoseStamped, strict: bool
    ) -> None:
        trajectory = self.plan_move(self.pose_goal(target), label)
        if not self.args.execute:
            self.record(
                label,
                "planned",
                points=len(trajectory.joint_trajectory.points),
            )
            return
        self.execute(trajectory, label)
        verification = self.verify(target, label, strict)
        self.record(label, "executed", verification=verification)

    def run(self) -> int:
        try:
            entrance = self.wait_for_visual_target()
            self.run_stage_move("L_SAFE_JOINT", self.joint_goal(self.args.safe_joints))
            self.run_stage_move("L_READY_JOINT", self.joint_goal(self.args.ready_joints))

            prealign = self.target_pose(entrance, 0.04)
            pose_trajectory = self.plan_move(self.pose_goal(prealign), "L_VISUAL_PREALIGN")
            if not self.args.execute:
                self.record(
                    "L_VISUAL_PREALIGN",
                    "planned",
                    points=len(pose_trajectory.joint_trajectory.points),
                )
            else:
                self.execute(pose_trajectory, "L_VISUAL_PREALIGN")
                self.record(
                    "L_VISUAL_PREALIGN",
                    "executed",
                    verification=self.verify(prealign, "L_VISUAL_PREALIGN", False),
                )
            if self.args.phase == "prealign":
                return self.finish(None)
            if not self.args.execute:
                raise RuntimeError("approach/insert phase 必须使用 --execute --confirm-execute")

            approach = self.target_pose(entrance, 0.01)
            self.run_pose_joint_stage("L_APPROACH_JOINT", approach, strict=True)
            if self.args.phase == "approach":
                return self.finish(None)

            insert20 = self.target_pose(entrance, -0.02)
            self.run_pose_stage("L_INSERT_20MM", insert20, 0.03, strict=True)
            if self.args.phase == "insert20":
                return self.finish(None)

            insert70 = self.target_pose(entrance, -0.07)
            self.run_pose_stage("L_INSERT_70MM", insert70, 0.05, strict=True)
            if self.args.phase == "insert70":
                return self.finish(None)

            # Withdraw along the same axis before using joint-space return
            # motions.  The final positive-distance waypoint leaves a clear
            # gap between the rod tip and the cylinder entrance.
            for distance_m, segment_m, label in (
                (-0.05, 0.02, "L_WITHDRAW_50MM"),
                (-0.03, 0.02, "L_WITHDRAW_30MM"),
                (-0.01, 0.02, "L_WITHDRAW_10MM"),
                (0.01, 0.02, "L_WITHDRAW_OUTSIDE_10MM"),
                (0.04, 0.03, "L_WITHDRAW_CLEAR_40MM"),
            ):
                withdraw = self.target_pose(entrance, distance_m)
                self.run_pose_stage(label, withdraw, segment_m, strict=True)
            self.run_stage_move(
                "L_READY_RETURN", self.joint_goal(self.args.ready_joints)
            )
            self.run_stage_move(
                "L_SAFE_RETURN", self.joint_goal(self.args.safe_joints)
            )
            return self.finish(None)
        except Exception as exc:
            self.get_logger().error(str(exc))
            return self.finish(str(exc))

    def finish(self, error: str | None) -> int:
        payload = {
            "phase": self.args.phase,
            "execute": self.args.execute,
            "error": error,
            "stages": self.stages,
        }
        path = Path(self.args.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        self.get_logger().info(f"L_xarm_into 报告已写入 {path}")
        return 1 if error else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L_ 长杆分阶段规划、对准和穿入")
    parser.add_argument(
        "--phase",
        choices=["prealign", "approach", "insert20", "insert70", "through"],
        default="prealign",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-execute", action="store_true")
    parser.add_argument(
        "--safe-joints",
        type=parse_joint_values,
        default=[0.0, 0.0, 0.0, 0.0, -1.5708, 0.0],
    )
    parser.add_argument(
        "--ready-joints",
        type=parse_joint_values,
        default=[0.0, -0.40, 0.80, 0.0, -1.20, 0.0],
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--wait-sec", type=float, default=30.0)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--max-position-std-m", type=float, default=0.005)
    parser.add_argument("--max-axis-std-deg", type=float, default=3.0)
    parser.add_argument("--joint-tolerance-rad", type=float, default=0.02)
    parser.add_argument("--pose-position-tolerance-m", type=float, default=0.002)
    parser.add_argument("--pose-orientation-tolerance-rad", type=float, default=0.01)
    parser.add_argument("--prealign-position-tolerance-m", type=float, default=0.005)
    parser.add_argument("--prealign-axis-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--insert-position-tolerance-m", type=float, default=0.003)
    parser.add_argument("--insert-axis-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--settle-sec", type=float, default=10.0)
    parser.add_argument("--planning-time-sec", type=float, default=10.0)
    parser.add_argument("--planning-attempts", type=int, default=5)
    parser.add_argument("--velocity-scaling", type=float, default=0.10)
    parser.add_argument("--acceleration-scaling", type=float, default=0.10)
    parser.add_argument("--cartesian-step-m", type=float, default=0.001)
    parser.add_argument("--cartesian-speed-mps", type=float, default=0.005)
    parser.add_argument("--cartesian-min-duration", type=float, default=3.0)
    parser.add_argument("--min-fraction", type=float, default=0.98)
    # MoveIt's relative jump detector can reject a valid slow path near an IK
    # branch boundary; explicit joint-step and FK gates remain enabled.
    parser.add_argument("--jump-threshold-rad", type=float, default=0.0)
    parser.add_argument("--prismatic-jump-threshold-m", type=float, default=0.02)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.10)
    parser.add_argument("--joint-state-max-age-sec", type=float, default=0.50)
    parser.add_argument("--lock-path", default="/tmp/l_xarm_into.lock")
    parser.add_argument("--group-name", default="L_xarm6")
    parser.add_argument("--tool-frame", default="L_link_tcp")
    parser.add_argument("--planning-frame", default="world")
    parser.add_argument(
        "--entrance-topic", default="/cylinder_pose_estimator/entrance_pose"
    )
    parser.add_argument(
        "--confidence-topic", default="/cylinder_pose_estimator/confidence"
    )
    parser.add_argument("--report-path", default="/tmp/l_xarm_into_report.json")
    args = parser.parse_args(argv)
    if args.execute and not args.confirm_execute:
        parser.error("--execute requires --confirm-execute")
    if args.phase != "prealign" and not args.execute:
        parser.error("approach/insert phase 必须使用 --execute --confirm-execute")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rclpy.init(args=None)
    node: LArmInto | None = None
    try:
        node = LArmInto(args)
        return node.run()
    except Exception as exc:
        if node is not None:
            node.get_logger().error(str(exc))
        else:
            print(f"L_xarm_into 启动失败：{exc}")
        return 1
    finally:
        if node is not None:
            node.cancel_active_goal("L_xarm_into 退出")
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
