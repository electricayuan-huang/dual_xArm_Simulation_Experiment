#!/usr/bin/env python3
"""Estimate a hollow cylinder pose from a registered RGB-D image pair.

This node is deliberately perception-only. It publishes the estimated cylinder
center, the visible entrance center, and the axis pointing from the entrance
into the cylinder. It never sends a motion goal to either arm.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time as RosTime
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker
import tf2_ros


def stamp_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


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


def transform_to_matrix(transform: Any) -> np.ndarray:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_matrix(
        [rotation.x, rotation.y, rotation.z, rotation.w]
    )
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


def normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError("不能归一化零向量")
    return vector / length


def rotation_with_z_axis(axis: np.ndarray) -> np.ndarray:
    z_axis = normalize(axis)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, z_axis))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    x_axis = normalize(np.cross(reference, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation matrix to an xyzw quaternion."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def fit_cross_section_center(
    points: np.ndarray, axis: np.ndarray, origin: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Fit a circle in a plane normal to the cylinder axis."""
    if len(points) < 6:
        return np.zeros(3, dtype=np.float64), float("nan"), float("inf")
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, axis))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    basis_u = normalize(np.cross(axis, reference))
    basis_v = normalize(np.cross(axis, basis_u))
    relative = points - origin
    u = relative @ basis_u
    v = relative @ basis_v
    design = np.column_stack((2.0 * u, 2.0 * v, np.ones(len(points))))
    values = u * u + v * v
    try:
        solution, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(3, dtype=np.float64), float("nan"), float("inf")
    center_offset = basis_u * solution[0] + basis_v * solution[1]
    radius_sq = float(solution[2] + solution[0] ** 2 + solution[1] ** 2)
    if radius_sq <= 0.0:
        return center_offset, float("nan"), float("inf")
    radius = math.sqrt(radius_sq)
    residual = float(np.sqrt(np.mean((np.sqrt(values) - radius) ** 2)))
    return center_offset, radius, residual


def estimate_cylinder_geometry(
    points: np.ndarray,
    camera_origin: np.ndarray | None = None,
    known_height_m: float | None = None,
) -> dict[str, Any]:
    """Estimate axis and the camera-facing end from a colored point cloud."""
    if len(points) < 20:
        raise ValueError(f"有效三维点过少：{len(points)}")
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / max(1, len(points) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axis = normalize(eigenvectors[:, order[0]])
    projections = centered @ axis
    low = float(np.percentile(projections, 5.0))
    high = float(np.percentile(projections, 95.0))
    extent = high - low
    if extent <= 1e-6:
        raise ValueError("点云轴向跨度过小，无法估计圆柱轴")

    near_selection = (projections >= low) & (projections <= np.percentile(projections, 25.0))
    far_selection = (projections >= np.percentile(projections, 75.0)) & (projections <= high)
    near_offset, near_radius, near_residual = fit_cross_section_center(
        points[near_selection], axis, centroid
    )
    far_offset, far_radius, far_residual = fit_cross_section_center(
        points[far_selection], axis, centroid
    )
    near_endpoint = centroid + near_offset + axis * low
    far_endpoint = centroid + far_offset + axis * high

    if camera_origin is None:
        camera_origin = np.zeros(3, dtype=np.float64)
    if np.linalg.norm(near_endpoint - camera_origin) <= np.linalg.norm(
        far_endpoint - camera_origin
    ):
        entrance = near_endpoint
        exit_center = far_endpoint
        axis_into_cylinder = normalize(far_endpoint - near_endpoint)
    else:
        entrance = far_endpoint
        exit_center = near_endpoint
        axis_into_cylinder = normalize(near_endpoint - far_endpoint)

    if known_height_m is not None and known_height_m > 0.0:
        # The RGB mask may expose only the camera-facing part of the wall.
        # Use the measured entrance and the model height to recover the
        # physical center instead of treating the visible-points centroid as
        # the cylinder center.
        exit_center = entrance + axis_into_cylinder * known_height_m
        physical_center = entrance + axis_into_cylinder * (known_height_m / 2.0)
    else:
        physical_center = (entrance + exit_center) / 2.0

    line_points = centroid + np.outer(projections, axis)
    radial_distances = np.linalg.norm(points - line_points, axis=1)
    radius_median = float(np.median(radial_distances))
    radius_mad = float(np.median(np.abs(radial_distances - radius_median)))
    circle_residual = float(np.nanmean([near_residual, far_residual]))
    if not math.isfinite(circle_residual):
        circle_residual = float("inf")
    return {
        "center_m": physical_center.tolist(),
        "entrance_m": entrance.tolist(),
        "exit_m": exit_center.tolist(),
        "axis": axis_into_cylinder.tolist(),
        "axis_extent_m": float(extent),
        "known_height_m": float(known_height_m) if known_height_m else None,
        "point_count": int(len(points)),
        "radius_median_m": radius_median,
        "radius_mad_m": radius_mad,
        "near_radius_m": float(near_radius),
        "far_radius_m": float(far_radius),
        "circle_residual_m": circle_residual,
    }


class CylinderPoseEstimator(Node):
    """RGB-D cylinder estimator; no robot motion is performed here."""

    def __init__(self) -> None:
        super().__init__("cylinder_pose_estimator")
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/depth_image")
        self.declare_parameter("color_info_topic", "/camera/color/camera_info")
        self.declare_parameter("depth_info_topic", "/camera/depth/camera_info")
        self.declare_parameter("color_frame", "camera_color_optical_frame")
        self.declare_parameter("depth_frame", "camera_depth_optical_frame")
        self.declare_parameter("output_frame", "world")
        # The simulated D435 publishes color and depth from separate optical
        # frames. False uses the depth->color TF to associate pixels; true is
        # reserved for an explicitly registered depth image.
        self.declare_parameter("depth_aligned_to_color", False)
        self.declare_parameter("depth_sampling_stride", 2)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("depth_min_m", 0.10)
        self.declare_parameter("depth_max_m", 5.0)
        self.declare_parameter("hsv_lower", [25, 45, 20])
        self.declare_parameter("hsv_upper", [95, 255, 255])
        self.declare_parameter("min_mask_area_px", 100)
        self.declare_parameter("max_points", 3000)
        self.declare_parameter("min_axis_extent_m", 0.04)
        self.declare_parameter("max_axis_extent_m", 0.25)
        self.declare_parameter("outer_radius_m", 0.030)
        self.declare_parameter("outer_radius_tolerance_m", 0.020)
        self.declare_parameter("cylinder_height_m", 0.12)
        self.declare_parameter("max_sync_delta_sec", 0.10)
        self.declare_parameter("process_rate_hz", 10.0)
        self.declare_parameter("min_confidence", 0.50)
        self.declare_parameter("max_tracking_position_jump_m", 0.006)
        self.declare_parameter("max_tracking_axis_jump_deg", 3.0)
        self.declare_parameter("lock_pose", False)
        self.declare_parameter("lock_after_estimates", 10)
        self.declare_parameter("debug_image_topic", "/vision/cylinder/debug")
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("report_path", "/tmp/cylinder_pose_report.json")
        self.declare_parameter("duration_sec", 0.0)

        self.color_frame = str(self.get_parameter("color_frame").value)
        self.depth_frame = str(self.get_parameter("depth_frame").value)
        self.output_frame = str(self.get_parameter("output_frame").value)
        self.depth_aligned_to_color = bool(
            self.get_parameter("depth_aligned_to_color").value
        )
        self.depth_sampling_stride = max(
            1, int(self.get_parameter("depth_sampling_stride").value)
        )
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)
        self.hsv_lower = np.asarray(self.get_parameter("hsv_lower").value, dtype=np.uint8)
        self.hsv_upper = np.asarray(self.get_parameter("hsv_upper").value, dtype=np.uint8)
        self.min_mask_area_px = int(self.get_parameter("min_mask_area_px").value)
        self.max_points = int(self.get_parameter("max_points").value)
        self.min_axis_extent_m = float(self.get_parameter("min_axis_extent_m").value)
        self.max_axis_extent_m = float(self.get_parameter("max_axis_extent_m").value)
        self.outer_radius_m = float(self.get_parameter("outer_radius_m").value)
        self.outer_radius_tolerance_m = float(
            self.get_parameter("outer_radius_tolerance_m").value
        )
        self.cylinder_height_m = float(self.get_parameter("cylinder_height_m").value)
        self.max_sync_delta_sec = float(self.get_parameter("max_sync_delta_sec").value)
        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.max_tracking_position_jump_m = float(
            self.get_parameter("max_tracking_position_jump_m").value
        )
        self.max_tracking_axis_jump_deg = float(
            self.get_parameter("max_tracking_axis_jump_deg").value
        )
        self.lock_pose = bool(self.get_parameter("lock_pose").value)
        self.lock_after_estimates = max(
            1, int(self.get_parameter("lock_after_estimates").value)
        )
        self.publish_debug_image = bool(self.get_parameter("publish_debug_image").value)
        self.report_path = Path(str(self.get_parameter("report_path").value))
        self.duration_sec = float(self.get_parameter("duration_sec").value)

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.latest_color: tuple[Image, np.ndarray] | None = None
        self.latest_depth: tuple[Image, np.ndarray] | None = None
        self.color_info: CameraInfo | None = None
        self.depth_info: CameraInfo | None = None
        self.last_pair_key: tuple[float, float] | None = None
        self.last_estimate: dict[str, Any] | None = None
        self.locked_estimate: dict[str, Any] | None = None
        self.estimate_count = 0
        self.failure_count = 0
        self.started_monotonic = time.monotonic()
        self.last_warning_monotonic = 0.0

        self.center_publisher = self.create_publisher(
            PoseStamped, "~/cylinder_pose", 10
        )
        self.entrance_publisher = self.create_publisher(
            PoseStamped, "~/entrance_pose", 10
        )
        self.axis_publisher = self.create_publisher(Vector3Stamped, "~/axis", 10)
        self.confidence_publisher = self.create_publisher(Float32, "~/confidence", 10)
        self.marker_publisher = self.create_publisher(Marker, "~/marker", 10)
        self.debug_publisher = self.create_publisher(
            Image, str(self.get_parameter("debug_image_topic").value), 10
        )

        self.create_subscription(
            Image,
            str(self.get_parameter("color_topic").value),
            self.color_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("depth_topic").value),
            self.depth_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("color_info_topic").value),
            self.color_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("depth_info_topic").value),
            self.depth_info_callback,
            qos_profile_sensor_data,
        )
        self.process_timer = self.create_timer(
            1.0 / max(1.0, float(self.get_parameter("process_rate_hz").value)),
            self.process_latest_pair,
        )
        self.status_timer = self.create_timer(
            1.0,
            self.check_duration,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        self.get_logger().info(
            "RGB-D 圆柱位姿估计已启动；仅发布视觉结果，不执行机械臂运动。"
        )

    def warn_throttled(self, message: str, period_sec: float = 5.0) -> None:
        now = time.monotonic()
        if now - self.last_warning_monotonic >= period_sec:
            self.get_logger().warning(message)
            self.last_warning_monotonic = now

    def color_callback(self, message: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.warn_throttled(f"RGB 转换失败：{exc}")
            return
        self.latest_color = (message, np.asarray(image))

    def depth_callback(self, message: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self.warn_throttled(f"深度转换失败：{exc}")
            return
        depth_array = np.asarray(depth)
        if depth_array.dtype == np.uint16:
            depth_array = depth_array.astype(np.float32) * self.depth_scale
        else:
            depth_array = depth_array.astype(np.float32)
        self.latest_depth = (message, depth_array)

    def color_info_callback(self, message: CameraInfo) -> None:
        self.color_info = message

    def depth_info_callback(self, message: CameraInfo) -> None:
        self.depth_info = message

    def segment_target(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if components <= 1:
            return np.zeros_like(mask)
        candidates = [
            index
            for index in range(1, components)
            if int(stats[index, cv2.CC_STAT_AREA]) >= self.min_mask_area_px
        ]
        if not candidates:
            return np.zeros_like(mask)
        largest = max(candidates, key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
        return np.where(labels == largest, 255, 0).astype(np.uint8)

    def deproject(
        self, pixels: np.ndarray, depth_values: np.ndarray, info: CameraInfo
    ) -> np.ndarray:
        camera_matrix = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(info.d, dtype=np.float64)
        if distortion.size:
            normalized = cv2.undistortPoints(
                pixels.reshape(-1, 1, 2).astype(np.float64),
                camera_matrix,
                distortion,
            ).reshape(-1, 2)
            x_normalized = normalized[:, 0]
            y_normalized = normalized[:, 1]
        else:
            x_normalized = (pixels[:, 0] - camera_matrix[0, 2]) / camera_matrix[0, 0]
            y_normalized = (pixels[:, 1] - camera_matrix[1, 2]) / camera_matrix[1, 1]
        return np.column_stack(
            (
                x_normalized * depth_values,
                y_normalized * depth_values,
                depth_values,
            )
        )

    def project(self, points: np.ndarray, info: CameraInfo) -> np.ndarray:
        camera_matrix = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(info.d, dtype=np.float64)
        projected, _ = cv2.projectPoints(
            points.reshape(-1, 1, 3),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            camera_matrix,
            distortion,
        )
        return projected.reshape(-1, 2)

    def lookup_transform(self, source_frame: str, stamp: Any) -> np.ndarray:
        lookup_time = RosTime()
        if stamp.sec != 0 or stamp.nanosec != 0:
            lookup_time = RosTime.from_msg(stamp)
        transform = self.tf_buffer.lookup_transform(
            self.output_frame,
            source_frame,
            lookup_time,
            timeout=Duration(seconds=0.3),
        )
        return transform_to_matrix(transform)

    @staticmethod
    def ros_time_from_stamp(stamp: Any) -> RosTime:
        return RosTime.from_msg(stamp) if stamp.sec or stamp.nanosec else RosTime()

    def depth_points_under_color_mask(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        depth_info: CameraInfo,
        color_info: CameraInfo,
        stamp: Any,
    ) -> np.ndarray:
        """Return depth-frame points whose projection lands in the RGB mask."""
        if self.depth_aligned_to_color:
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                return np.empty((0, 3), dtype=np.float64)
            depth_values = depth[ys, xs]
            valid = (
                np.isfinite(depth_values)
                & (depth_values >= self.depth_min_m)
                & (depth_values <= self.depth_max_m)
            )
            pixels = np.column_stack((xs[valid], ys[valid])).astype(np.float64)
            return self.deproject(pixels, depth_values[valid], depth_info)

        height, width = depth.shape[:2]
        stride = self.depth_sampling_stride
        grid_y, grid_x = np.mgrid[0:height:stride, 0:width:stride]
        depth_pixels = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(
            np.float64
        )
        depth_values = depth[grid_y, grid_x].ravel()
        valid = (
            np.isfinite(depth_values)
            & (depth_values >= self.depth_min_m)
            & (depth_values <= self.depth_max_m)
        )
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float64)
        depth_pixels = depth_pixels[valid]
        depth_values = depth_values[valid]
        depth_points = self.deproject(depth_pixels, depth_values, depth_info)
        depth_to_color = transform_to_matrix(
            self.tf_buffer.lookup_transform(
                color_info.header.frame_id or self.color_frame,
                depth_info.header.frame_id or self.depth_frame,
                self.ros_time_from_stamp(stamp),
                timeout=Duration(seconds=0.3),
            )
        )
        color_points = (
            depth_to_color[:3, :3] @ depth_points.T
            + depth_to_color[:3, 3:4]
        ).T
        positive = color_points[:, 2] > 0.0
        if not np.any(positive):
            return np.empty((0, 3), dtype=np.float64)
        depth_points = depth_points[positive]
        color_points = color_points[positive]
        projected = self.project(color_points, color_info)
        projected_x = np.rint(projected[:, 0]).astype(np.int32)
        projected_y = np.rint(projected[:, 1]).astype(np.int32)
        inside = (
            (projected_x >= 0)
            & (projected_x < mask.shape[1])
            & (projected_y >= 0)
            & (projected_y < mask.shape[0])
        )
        if not np.any(inside):
            return np.empty((0, 3), dtype=np.float64)
        depth_points = depth_points[inside]
        projected_x = projected_x[inside]
        projected_y = projected_y[inside]
        selected = mask[projected_y, projected_x] > 0
        return depth_points[selected]

    def process_latest_pair(self) -> None:
        if (
            self.latest_color is None
            or self.latest_depth is None
            or self.color_info is None
            or self.depth_info is None
        ):
            return
        color_message, color = self.latest_color
        depth_message, depth = self.latest_depth
        color_stamp = stamp_seconds(color_message.header.stamp)
        depth_stamp = stamp_seconds(depth_message.header.stamp)
        pair_key = (color_stamp, depth_stamp)
        if self.last_pair_key == pair_key:
            return
        if abs(color_stamp - depth_stamp) > self.max_sync_delta_sec:
            self.warn_throttled(
                "RGB 与深度时间差超过阈值，跳过本帧："
                f"{abs(color_stamp - depth_stamp):.4f}s"
            )
            return
        self.last_pair_key = pair_key
        color_frame = self.color_info.header.frame_id or self.color_frame
        depth_frame = self.depth_info.header.frame_id or self.depth_frame
        lookup_time = self.ros_time_from_stamp(depth_message.header.stamp)
        if not self.tf_buffer.can_transform(
            color_frame, depth_frame, lookup_time, timeout=Duration(seconds=0.05)
        ) or not self.tf_buffer.can_transform(
            self.output_frame, depth_frame, lookup_time, timeout=Duration(seconds=0.05)
        ):
            self.last_pair_key = None
            return
        if (
            self.lock_pose
            and self.locked_estimate is not None
            and self.last_estimate is not None
        ):
            locked = deepcopy(self.locked_estimate)
            locked["stamp"] = {
                "sec": int(depth_message.header.stamp.sec),
                "nanosec": int(depth_message.header.stamp.nanosec),
            }
            locked["tracking_locked"] = True
            self.last_estimate = locked
            self.estimate_count += 1
            self.publish_estimate(locked, depth_message.header.stamp)
            return
        if color.shape[:2] != depth.shape[:2]:
            self.failure_count += 1
            self.warn_throttled(
                f"RGB/depth 分辨率不一致：{color.shape[:2]} vs {depth.shape[:2]}"
            )
            return
        mask = self.segment_target(color)
        if np.count_nonzero(mask) < self.min_mask_area_px:
            self.failure_count += 1
            self.publish_debug(color, mask, color_message)
            return
        try:
            points = self.depth_points_under_color_mask(
                mask,
                depth,
                self.depth_info,
                self.color_info,
                depth_message.header.stamp,
            )
        except (tf2_ros.TransformException, np.linalg.LinAlgError) as exc:
            self.failure_count += 1
            self.publish_debug(color, mask, color_message)
            self.warn_throttled(f"RGB/depth 像素配准失败：{exc}")
            return
        if len(points) < 20:
            self.failure_count += 1
            self.publish_debug(color, mask, color_message)
            return
        try:
            if len(points) > self.max_points:
                indices = np.linspace(0, len(points) - 1, self.max_points).astype(int)
                points = points[indices]
            geometry = estimate_cylinder_geometry(
                points, known_height_m=self.cylinder_height_m
            )
            if not (
                self.min_axis_extent_m
                <= geometry["axis_extent_m"]
                <= self.max_axis_extent_m
            ):
                raise ValueError(
                    "轴向跨度不符合圆柱模型："
                    f"{geometry['axis_extent_m']:.4f}m"
                )
            radius_error = abs(geometry["radius_median_m"] - self.outer_radius_m)
            if radius_error > self.outer_radius_tolerance_m:
                raise ValueError(
                    "拟合半径不符合圆柱模型："
                    f"{geometry['radius_median_m']:.4f}m"
                )
            transform = self.lookup_transform(
                self.depth_info.header.frame_id or self.depth_frame,
                depth_message.header.stamp,
            )
            center_source = np.asarray(geometry["center_m"], dtype=np.float64)
            entrance_source = np.asarray(geometry["entrance_m"], dtype=np.float64)
            axis_source = normalize(np.asarray(geometry["axis"], dtype=np.float64))
            center_output = transform[:3, :3] @ center_source + transform[:3, 3]
            entrance_output = transform[:3, :3] @ entrance_source + transform[:3, 3]
            axis_output = normalize(transform[:3, :3] @ axis_source)
            confidence = self.compute_confidence(
                len(points), geometry["axis_extent_m"], radius_error, geometry["circle_residual_m"]
            )
            estimate = {
                **geometry,
                "source_frame": self.depth_info.header.frame_id or self.depth_frame,
                "output_frame": self.output_frame,
                "center_output_m": center_output.tolist(),
                "entrance_output_m": entrance_output.tolist(),
                "axis_output": axis_output.tolist(),
                "mask_area_px": int(np.count_nonzero(mask)),
                "depth_point_count": int(len(points)),
                "confidence": confidence,
                "stamp": {
                    "sec": int(depth_message.header.stamp.sec),
                    "nanosec": int(depth_message.header.stamp.nanosec),
                },
            }
            if confidence < self.min_confidence:
                self.failure_count += 1
                self.publish_debug(color, mask, color_message)
                self.warn_throttled(
                    "圆柱估计置信度不足，丢弃本帧："
                    f"{confidence:.3f} < {self.min_confidence:.3f}"
                )
                return
            if self.last_estimate is not None:
                previous_entrance = np.asarray(
                    self.last_estimate["entrance_output_m"], dtype=np.float64
                )
                previous_axis = normalize(
                    np.asarray(self.last_estimate["axis_output"], dtype=np.float64)
                )
                entrance_jump = float(np.linalg.norm(entrance_output - previous_entrance))
                axis_jump = math.degrees(
                    math.acos(
                        max(-1.0, min(1.0, float(np.dot(axis_output, previous_axis))))
                    )
                )
                if (
                    entrance_jump > self.max_tracking_position_jump_m
                    or axis_jump > self.max_tracking_axis_jump_deg
                ):
                    self.failure_count += 1
                    self.publish_debug(color, mask, color_message)
                    self.warn_throttled(
                        "圆柱估计相对上一稳定帧跳变，丢弃本帧："
                        f"position={entrance_jump * 1000.0:.2f}mm, "
                        f"axis={axis_jump:.2f}deg"
                    )
                    return
            self.last_estimate = estimate
            if self.lock_pose and self.estimate_count + 1 >= self.lock_after_estimates:
                self.locked_estimate = deepcopy(estimate)
                self.get_logger().info(
                    "圆柱位姿已锁定：后续阶段假设 R_ 保持静止，"
                    f"lock_after_estimates={self.lock_after_estimates}"
                )
            self.estimate_count += 1
            self.publish_estimate(estimate, depth_message.header.stamp)
            self.publish_debug(color, mask, color_message, estimate)
            if self.estimate_count == 1 or self.estimate_count % 10 == 0:
                self.get_logger().info(
                    "圆柱视觉位姿：入口=(%.3f, %.3f, %.3f)m，轴=(%.3f, %.3f, %.3f)，置信度=%.2f"
                    % (
                        *estimate["entrance_output_m"],
                        *estimate["axis_output"],
                        confidence,
                    )
                )
        except (ValueError, tf2_ros.TransformException, np.linalg.LinAlgError) as exc:
            self.failure_count += 1
            self.publish_debug(color, mask, color_message)
            self.warn_throttled(f"圆柱位姿估计失败：{exc}")

    def compute_confidence(
        self, point_count: int, axis_extent: float, radius_error: float, circle_residual: float
    ) -> float:
        point_score = min(1.0, point_count / 600.0)
        extent_score = min(
            1.0,
            max(0.0, (axis_extent - self.min_axis_extent_m) / 0.06),
        )
        radius_score = max(
            0.0,
            1.0 - radius_error / max(self.outer_radius_tolerance_m, 1e-6),
        )
        residual_score = max(0.0, 1.0 - circle_residual / 0.01)
        return float(0.25 * point_score + 0.30 * extent_score + 0.25 * radius_score + 0.20 * residual_score)

    def publish_estimate(self, estimate: dict[str, Any], stamp: Any) -> None:
        quaternion = matrix_to_quaternion(
            rotation_with_z_axis(np.asarray(estimate["axis_output"], dtype=np.float64))
        )
        center_pose = PoseStamped()
        center_pose.header.stamp = stamp
        center_pose.header.frame_id = self.output_frame
        center_pose.pose.position.x, center_pose.pose.position.y, center_pose.pose.position.z = (
            estimate["center_output_m"]
        )
        center_pose.pose.orientation.x, center_pose.pose.orientation.y, center_pose.pose.orientation.z, center_pose.pose.orientation.w = quaternion
        entrance_pose = PoseStamped()
        entrance_pose.header = center_pose.header
        entrance_pose.pose = center_pose.pose
        entrance_pose.pose.position.x, entrance_pose.pose.position.y, entrance_pose.pose.position.z = (
            estimate["entrance_output_m"]
        )
        axis_message = Vector3Stamped()
        axis_message.header = center_pose.header
        axis_message.vector.x, axis_message.vector.y, axis_message.vector.z = estimate[
            "axis_output"
        ]
        confidence_message = Float32()
        confidence_message.data = float(estimate["confidence"])
        self.center_publisher.publish(center_pose)
        self.entrance_publisher.publish(entrance_pose)
        self.axis_publisher.publish(axis_message)
        self.confidence_publisher.publish(confidence_message)

        marker = Marker()
        marker.header = center_pose.header
        marker.ns = "cylinder_pose_estimator"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = entrance_pose.pose
        marker.scale.x = 0.10
        marker.scale.y = 0.004
        marker.scale.z = 0.004
        marker.color.r = 1.0
        marker.color.g = 0.1
        marker.color.b = 0.1
        marker.color.a = 0.9
        self.marker_publisher.publish(marker)

    def publish_debug(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        message: Image,
        estimate: dict[str, Any] | None = None,
    ) -> None:
        if not self.publish_debug_image:
            return
        debug = image.copy()
        debug[mask > 0] = (
            0.5 * debug[mask > 0] + np.array([0.0, 80.0, 0.0])
        ).astype(np.uint8)
        if estimate is not None:
            cv2.putText(
                debug,
                "cylinder confidence=%.2f" % estimate["confidence"],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        try:
            debug_message = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            debug_message.header = message.header
            self.debug_publisher.publish(debug_message)
        except CvBridgeError as exc:
            self.warn_throttled(f"调试图像发布失败：{exc}")

    def check_duration(self) -> None:
        if self.duration_sec > 0.0 and time.monotonic() - self.started_monotonic >= self.duration_sec:
            self.write_report()
            self.status_timer.cancel()
            self.process_timer.cancel()
            rclpy.shutdown()

    def write_report(self) -> None:
        payload = {
            "healthy": self.last_estimate is not None,
            "generated_at_unix": time.time(),
            "output_frame": self.output_frame,
            "depth_aligned_to_color": self.depth_aligned_to_color,
            "estimate_count": self.estimate_count,
            "failure_count": self.failure_count,
            "estimate": self.last_estimate,
            "tracking": {
                "lock_pose": self.lock_pose,
                "locked": self.locked_estimate is not None,
                "lock_after_estimates": self.lock_after_estimates,
            },
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.get_logger().info(
            f"圆柱视觉报告已写入 {self.report_path}，"
            f"估计帧={self.estimate_count}，失败帧={self.failure_count}"
        )


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = CylinderPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.write_report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
