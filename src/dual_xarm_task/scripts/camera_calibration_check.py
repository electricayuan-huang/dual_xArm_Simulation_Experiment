#!/usr/bin/env python3
"""固定外部相机健康检查和 ArUco 标定验证入口。

默认只检查相机数据、RGB/深度 CameraInfo 和固定 TF 链路。
启用 detect_charuco 后，会使用指定的 ChArUco 棋盘进行多角点
PnP 姿态估计，并输出图像中的标定板位置和多帧稳定性。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import tf2_ros


ARUCO_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
}


def stamp_to_dict(stamp: TimeMsg) -> dict[str, int]:
    return {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}


def transform_to_dict(transform: TransformStamped) -> dict[str, Any]:
    t = transform.transform.translation
    q = transform.transform.rotation
    return {
        "parent_frame": transform.header.frame_id,
        "child_frame": transform.child_frame_id,
        "translation_m": [t.x, t.y, t.z],
        "quaternion_xyzw": [q.x, q.y, q.z, q.w],
        "stamp": stamp_to_dict(transform.header.stamp),
    }


class CameraCalibrationCheck(Node):
    """采集固定相机状态并输出可复查的 JSON 报告。"""

    def __init__(self) -> None:
        super().__init__("camera_calibration_check")

        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/depth_image")
        self.declare_parameter("color_info_topic", "/camera/color/camera_info")
        self.declare_parameter("depth_info_topic", "/camera/depth/camera_info")
        self.declare_parameter("pointcloud_topic", "/camera/depth/points")
        self.declare_parameter("color_frame", "camera_color_optical_frame")
        self.declare_parameter("depth_frame", "camera_depth_optical_frame")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("gripper_base_frame", "R_link_base")
        self.declare_parameter("rod_base_frame", "L_link_base")
        self.declare_parameter("duration_sec", 10.0)
        self.declare_parameter(
            "report_path", "/tmp/dual_xarm_camera_calibration_report.json"
        )
        self.declare_parameter("detect_aruco", False)
        self.declare_parameter("aruco_dictionary", "DICT_4X4_50")
        self.declare_parameter("marker_id", 0)
        self.declare_parameter("marker_size_m", 0.0)
        self.declare_parameter("marker_frame", "")
        # PnP uses a centered ChArUco board frame at the physical board center.
        # This is the printed pattern plane center. The board visual is
        # offset by -6.5 mm along world Y from the board link center.
        self.declare_parameter("marker_world_xyz", [0.0, 0.2935, 1.42])
        self.declare_parameter("marker_world_rpy", [1.5707963, 0.0, 0.0])
        self.declare_parameter("detect_charuco", False)
        self.declare_parameter("charuco_squares_x", 5)
        self.declare_parameter("charuco_squares_y", 5)
        self.declare_parameter("charuco_square_size_m", 0.06)
        self.declare_parameter("charuco_marker_size_m", 0.042)
        self.declare_parameter("charuco_min_corners", 4)
        self.declare_parameter("charuco_report_path", "/tmp/charuco_detection.png")
        self.declare_parameter("aruco_detection_scale", 2.0)

        self.color_frame = str(self.get_parameter("color_frame").value)
        self.depth_frame = str(self.get_parameter("depth_frame").value)
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.report_path = Path(str(self.get_parameter("report_path").value))
        self.detect_aruco = bool(self.get_parameter("detect_aruco").value)
        self.marker_id = int(self.get_parameter("marker_id").value)
        self.marker_size_m = float(self.get_parameter("marker_size_m").value)
        self.marker_frame = str(self.get_parameter("marker_frame").value)
        self.marker_world_xyz = [
            float(value) for value in self.get_parameter("marker_world_xyz").value
        ]
        self.marker_world_rpy = [
            float(value) for value in self.get_parameter("marker_world_rpy").value
        ]
        self.detect_charuco = bool(self.get_parameter("detect_charuco").value)
        self.charuco_min_corners = int(self.get_parameter("charuco_min_corners").value)
        self.charuco_report_path = Path(
            str(self.get_parameter("charuco_report_path").value)
        )
        self.aruco_detection_scale = max(
            1.0, float(self.get_parameter("aruco_detection_scale").value)
        )

        dictionary_name = str(self.get_parameter("aruco_dictionary").value)
        if dictionary_name not in ARUCO_DICTIONARIES:
            raise ValueError(
                f"不支持的 ArUco 字典 {dictionary_name}，可选值："
                f"{', '.join(sorted(ARUCO_DICTIONARIES))}"
            )
        self.dictionary_name = dictionary_name
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(
            ARUCO_DICTIONARIES[dictionary_name]
        )
        self.charuco_board = cv2.aruco.CharucoBoard_create(
            int(self.get_parameter("charuco_squares_x").value),
            int(self.get_parameter("charuco_squares_y").value),
            float(self.get_parameter("charuco_square_size_m").value),
            float(self.get_parameter("charuco_marker_size_m").value),
            self.aruco_dictionary,
        )
        if hasattr(cv2.aruco, "DetectorParameters"):
            self.aruco_parameters = cv2.aruco.DetectorParameters()
        else:
            self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.started_monotonic = time.monotonic()
        self.last_color_monotonic = 0.0
        self.last_depth_monotonic = 0.0
        self.last_pointcloud_monotonic = 0.0
        self.color_count = 0
        self.depth_count = 0
        self.pointcloud_count = 0
        self.camera_info: CameraInfo | None = None
        self.depth_camera_info: CameraInfo | None = None
        self.last_color_frame_id = ""
        self.last_depth_frame_id = ""
        self.last_marker: dict[str, Any] | None = None
        self.last_charuco: dict[str, Any] | None = None
        self.charuco_samples: list[dict[str, Any]] = []
        self.last_tf: dict[str, dict[str, Any]] = {}
        self.tf_errors: dict[str, str] = {}

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
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("depth_info_topic").value),
            self.depth_camera_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("pointcloud_topic").value),
            self.pointcloud_callback,
            qos_profile_sensor_data,
        )

        self.timer = self.create_timer(
            1.0,
            self.check_status,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        self.get_logger().info(
            f"固定相机检查已启动：color={self.get_parameter('color_topic').value}, "
            f"depth={self.get_parameter('depth_topic').value}, "
            f"duration={self.duration_sec:.1f}s"
        )
        if self.detect_aruco:
            if self.marker_size_m <= 0.0:
                self.get_logger().warning(
                    "detect_aruco=true 但 marker_size_m <= 0，无法执行 PnP"
                )
            self.get_logger().info(
                f"ArUco 检查已开启：dictionary={self.dictionary_name}, "
                f"marker_id={self.marker_id}, size={self.marker_size_m:.4f}m"
            )
        if self.detect_charuco:
            self.get_logger().info(
                "ChArUco 检查已开启：5x5，square=%.3fm，marker=%.3fm，至少%d个角点"
                % (
                    float(self.get_parameter("charuco_square_size_m").value),
                    float(self.get_parameter("charuco_marker_size_m").value),
                    self.charuco_min_corners,
                )
            )

    def color_callback(self, message: Image) -> None:
        self.color_count += 1
        self.last_color_monotonic = time.monotonic()
        self.last_color_frame_id = message.header.frame_id
        if not self.detect_aruco or self.camera_info is None:
            if not self.detect_charuco or self.camera_info is None:
                return
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="mono8")
            if self.detect_charuco:
                # Keep the latest subscribed RGB frame even when detection fails.
                cv2.imwrite(
                    str(self.charuco_report_path), cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                )
            # Gazebo renders the 4x4 tags at a small pixel size. Detect on a
            # nearest-neighbor enlarged image, then return corners to the
            # original camera pixel coordinates for interpolation and PnP.
            detection_image = image
            scale = self.aruco_detection_scale
            if scale > 1.0:
                detection_image = cv2.resize(
                    image, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_NEAREST,
                )
            corners, ids, _ = cv2.aruco.detectMarkers(
                detection_image, self.aruco_dictionary, parameters=self.aruco_parameters
            )
            if scale > 1.0 and corners:
                corners = [corner / scale for corner in corners]
            if self.detect_charuco:
                self.process_charuco(image, corners, ids)
                return
            if ids is None or self.marker_size_m <= 0.0:
                return
            marker_indices = [
                index for index, marker_id in enumerate(ids.flatten())
                if int(marker_id) == self.marker_id
            ]
            if not marker_indices:
                return
            index = marker_indices[0]
            camera_matrix = np.asarray(self.camera_info.k, dtype=np.float64).reshape(3, 3)
            distortion = np.asarray(self.camera_info.d, dtype=np.float64)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[index]], self.marker_size_m, camera_matrix, distortion
            )
            rvec = rvecs[0, 0].tolist()
            tvec = tvecs[0, 0].tolist()
            self.last_marker = {
                "marker_id": self.marker_id,
                "frame_id": self.last_color_frame_id,
                "rvec_rad": rvec,
                "translation_m": tvec,
                "detected_monotonic": time.monotonic(),
            }
            self.last_marker.update(self.compute_extrinsic_comparison(rvec, tvec))
        except Exception as exc:  # OpenCV 编码和版本差异不能中断健康检查
            self.get_logger().warning(f"ArUco 图像处理失败：{exc}")

    def process_charuco(
        self, image: np.ndarray, corners: list[np.ndarray], ids: np.ndarray | None
    ) -> None:
        if ids is None or len(corners) == 0:
            return
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, image, self.charuco_board
        )
        if charuco_ids is None or len(charuco_ids) < self.charuco_min_corners:
            return

        camera_matrix = np.asarray(self.camera_info.k, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(self.camera_info.d, dtype=np.float64)
        refined = cv2.cornerSubPix(
            image,
            np.asarray(charuco_corners, dtype=np.float32),
            (5, 5),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        object_points = np.asarray(self.charuco_board.chessboardCorners)[
            np.asarray(charuco_ids).flatten()
        ].copy()
        # The physical board is 5 * square_size wide. The previous code
        # centered on the inner corner span (4 * square_size), offsetting
        # every PnP point by 30 mm from the known board center.
        square_size = float(self.get_parameter("charuco_square_size_m").value)
        board_size = self.charuco_board.getChessboardSize()
        object_points[:, 0] -= board_size[0] * square_size / 2.0
        object_points[:, 1] -= board_size[1] * square_size / 2.0
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            refined,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, distortion
        )
        reprojection_error = float(
            np.mean(np.linalg.norm(refined.reshape(-1, 2) - projected.reshape(-1, 2), axis=1))
        )
        marker_rotation, _ = cv2.Rodrigues(rvec)
        camera_board = np.eye(4, dtype=np.float64)
        camera_board[:3, :3] = marker_rotation
        camera_board[:3, 3] = tvec.flatten()
        board_world = np.eye(4, dtype=np.float64)
        board_world[:3, :3] = self.rpy_to_matrix(*self.marker_world_rpy)
        board_world[:3, 3] = np.asarray(self.marker_world_xyz, dtype=np.float64)
        world_camera = board_world @ np.linalg.inv(camera_board)

        pixel_points = refined.reshape(-1, 2)
        pixel_min = np.min(pixel_points, axis=0)
        pixel_max = np.max(pixel_points, axis=0)
        pixel_center = np.mean(pixel_points, axis=0)
        sample = {
            "world_camera": world_camera,
            "pixel_center": pixel_center.tolist(),
            "pixel_bbox": [
                float(pixel_min[0]), float(pixel_min[1]),
                float(pixel_max[0]), float(pixel_max[1]),
            ],
            "corners": int(len(charuco_ids)),
            "reprojection_error_px": reprojection_error,
        }
        sample.update(self.compute_extrinsic_comparison(rvec, tvec))
        self.charuco_samples.append(sample)
        self.charuco_samples = self.charuco_samples[-60:]
        self.last_charuco = {
            "corners": sample["corners"],
            "pixel_center": sample["pixel_center"],
            "pixel_bbox": sample["pixel_bbox"],
            "reprojection_error_px": reprojection_error,
            "world_camera": self.matrix_to_pose_dict(world_camera),
        }
        for key in ("nominal_world_camera", "translation_error_m", "rotation_error_deg"):
            if key in sample:
                self.last_charuco[key] = sample[key]
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        cv2.aruco.drawDetectedCornersCharuco(annotated, refined, charuco_ids)
        cv2.imwrite(str(self.charuco_report_path), annotated)
        if len(self.charuco_samples) == 1 or len(self.charuco_samples) % 10 == 0:
            self.get_logger().info(
                "ChArUco位置：像素中心=(%.1f, %.1f)，范围=[%.1f, %.1f]-[%.1f, %.1f]，角点=%d，重投影=%.3fpx"
                % (
                    sample["pixel_center"][0], sample["pixel_center"][1],
                    *sample["pixel_bbox"], sample["corners"], reprojection_error,
                )
            )

    def depth_callback(self, message: Image) -> None:
        self.depth_count += 1
        self.last_depth_monotonic = time.monotonic()
        self.last_depth_frame_id = message.header.frame_id

    def camera_info_callback(self, message: CameraInfo) -> None:
        self.camera_info = message

    def depth_camera_info_callback(self, message: CameraInfo) -> None:
        self.depth_camera_info = message

    def pointcloud_callback(self, _message: PointCloud2) -> None:
        self.pointcloud_count += 1
        self.last_pointcloud_monotonic = time.monotonic()

    @staticmethod
    def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ], dtype=np.float64)

    @staticmethod
    def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
        relative = first.T @ second
        cosine = max(-1.0, min(1.0, (np.trace(relative) - 1.0) / 2.0))
        return math.degrees(math.acos(cosine))

    def compute_extrinsic_comparison(
        self, rvec: list[float], tvec: list[float]
    ) -> dict[str, Any]:
        """Compare PnP-derived world->camera pose with the nominal TF."""
        rotation_camera_marker, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        camera_marker = np.eye(4, dtype=np.float64)
        camera_marker[:3, :3] = rotation_camera_marker
        camera_marker[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)

        marker_world = np.eye(4, dtype=np.float64)
        marker_world[:3, :3] = self.rpy_to_matrix(*self.marker_world_rpy)
        marker_world[:3, 3] = np.asarray(self.marker_world_xyz, dtype=np.float64)
        world_camera_from_pnp = marker_world @ np.linalg.inv(camera_marker)

        tf_data = self.last_tf.get(f"{self.world_frame}->{self.color_frame}")
        if tf_data is None:
            return {"pnp_world_camera": self.matrix_to_pose_dict(world_camera_from_pnp)}
        tf_matrix = np.eye(4, dtype=np.float64)
        tf_matrix[:3, :3] = self.quaternion_to_matrix(tf_data["quaternion_xyzw"])
        tf_matrix[:3, 3] = np.asarray(tf_data["translation_m"], dtype=np.float64)
        return {
            "pnp_world_camera": self.matrix_to_pose_dict(world_camera_from_pnp),
            "nominal_world_camera": self.matrix_to_pose_dict(tf_matrix),
            "translation_error_m": float(
                np.linalg.norm(world_camera_from_pnp[:3, 3] - tf_matrix[:3, 3])
            ),
            "rotation_error_deg": self.rotation_error_deg(
                world_camera_from_pnp[:3, :3], tf_matrix[:3, :3]
            ),
        }

    @staticmethod
    def quaternion_to_matrix(quaternion: list[float]) -> np.ndarray:
        x, y, z, w = [float(value) for value in quaternion]
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    @staticmethod
    def matrix_to_pose_dict(matrix: np.ndarray) -> dict[str, Any]:
        return {
            "translation_m": matrix[:3, 3].tolist(),
            "rotation_matrix": matrix[:3, :3].tolist(),
        }

    def lookup_tf(self, source_frame: str) -> None:
        key = f"{self.world_frame}->{source_frame}"
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.3),
            )
            self.last_tf[key] = transform_to_dict(transform)
            self.tf_errors.pop(key, None)
        except tf2_ros.TransformException as exc:
            self.tf_errors[key] = str(exc)

    @staticmethod
    def camera_info_is_valid(
        camera_info: CameraInfo | None, expected_frame: str
    ) -> tuple[bool, str]:
        if camera_info is None:
            return False, "未收到 CameraInfo"
        matrix = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
        valid = (
            camera_info.width > 0
            and camera_info.height > 0
            and matrix[0, 0] > 0
            and matrix[1, 1] > 0
            and camera_info.header.frame_id == expected_frame
        )
        return valid, "有效" if valid else "尺寸、内参或 frame_id 与预期不一致"

    def check_status(self) -> None:
        self.lookup_tf(self.color_frame)
        self.lookup_tf(self.depth_frame)
        self.lookup_tf(str(self.get_parameter("gripper_base_frame").value))
        self.lookup_tf(str(self.get_parameter("rod_base_frame").value))

        age = time.monotonic() - self.started_monotonic
        if self.color_count and self.depth_count and self.pointcloud_count:
            self.get_logger().info(
                f"相机状态：RGB={self.color_count}, depth={self.depth_count}, "
                f"pointcloud={self.pointcloud_count}, "
                f"CameraInfo={'ok' if self.depth_camera_info else 'pending'}, "
                f"TF={len(self.last_tf)}/4"
            )
        if self.duration_sec > 0.0 and age >= self.duration_sec:
            self.write_report()
            self.timer.cancel()
            rclpy.shutdown()

    def report(self) -> dict[str, Any]:
        now = time.monotonic()
        camera_info_valid, camera_info_reason = self.camera_info_is_valid(
            self.camera_info, self.color_frame
        )
        depth_camera_info_valid, depth_camera_info_reason = self.camera_info_is_valid(
            self.depth_camera_info, self.depth_frame
        )

        stream_ok = all(
            count > 0
            for count in (self.color_count, self.depth_count, self.pointcloud_count)
        )
        tf_ok = len(self.last_tf) == 4 and not self.tf_errors
        marker_ok = not self.detect_aruco or self.last_marker is not None
        charuco_ok = not self.detect_charuco or self.last_charuco is not None
        healthy = (
            stream_ok
            and camera_info_valid
            and depth_camera_info_valid
            and tf_ok
            and marker_ok
            and charuco_ok
        )

        charuco_summary: dict[str, Any] = {
            "enabled": self.detect_charuco,
            "board": {
                "squares_x": int(self.get_parameter("charuco_squares_x").value),
                "squares_y": int(self.get_parameter("charuco_squares_y").value),
                "square_size_m": float(self.get_parameter("charuco_square_size_m").value),
                "marker_size_m": float(self.get_parameter("charuco_marker_size_m").value),
            },
            "samples": len(self.charuco_samples),
            "detection": self.last_charuco,
            "annotated_image_path": str(self.charuco_report_path),
        }
        if self.charuco_samples:
            translations = np.asarray([
                sample["world_camera"][:3, 3] for sample in self.charuco_samples
            ])
            charuco_summary["translation_mean_m"] = np.mean(translations, axis=0).tolist()
            charuco_summary["translation_std_m"] = np.std(translations, axis=0).tolist()
            charuco_summary["translation_std_norm_m"] = float(np.linalg.norm(
                np.std(translations, axis=0)
            ))
            translation_errors = [
                sample["translation_error_m"]
                for sample in self.charuco_samples
                if "translation_error_m" in sample
            ]
            rotation_errors = [
                sample["rotation_error_deg"]
                for sample in self.charuco_samples
                if "rotation_error_deg" in sample
            ]
            if translation_errors:
                charuco_summary["translation_error_mean_m"] = float(
                    np.mean(translation_errors)
                )
                charuco_summary["translation_error_max_m"] = float(
                    np.max(translation_errors)
                )
            if rotation_errors:
                charuco_summary["rotation_error_mean_deg"] = float(
                    np.mean(rotation_errors)
                )
                charuco_summary["rotation_error_max_deg"] = float(
                    np.max(rotation_errors)
                )

        return {
            "healthy": healthy,
            "generated_at_unix": time.time(),
            "frames": {
                "world": self.world_frame,
                "color": self.color_frame,
                "depth": self.depth_frame,
                "color_message_frame_id": self.last_color_frame_id,
                "depth_message_frame_id": self.last_depth_frame_id,
            },
            "streams": {
                "color_messages": self.color_count,
                "depth_messages": self.depth_count,
                "pointcloud_messages": self.pointcloud_count,
                "color_recent": now - self.last_color_monotonic < 2.0,
                "depth_recent": now - self.last_depth_monotonic < 2.0,
                "pointcloud_recent": now - self.last_pointcloud_monotonic < 2.0,
            },
            "camera_info": {
                "valid": camera_info_valid,
                "reason": camera_info_reason,
                "width": self.camera_info.width if self.camera_info else 0,
                "height": self.camera_info.height if self.camera_info else 0,
                "frame_id": self.camera_info.header.frame_id if self.camera_info else "",
            },
            "depth_camera_info": {
                "valid": depth_camera_info_valid,
                "reason": depth_camera_info_reason,
                "width": self.depth_camera_info.width
                if self.depth_camera_info
                else 0,
                "height": self.depth_camera_info.height
                if self.depth_camera_info
                else 0,
                "frame_id": self.depth_camera_info.header.frame_id
                if self.depth_camera_info
                else "",
            },
            "tf": {"transforms": self.last_tf, "errors": self.tf_errors},
            "aruco": {
                "enabled": self.detect_aruco,
                "dictionary": self.dictionary_name,
                "marker_id": self.marker_id,
                "marker_size_m": self.marker_size_m,
                "detection": self.last_marker,
                "marker_frame": self.marker_frame,
            },
            "charuco": charuco_summary,
        }

    def write_report(self) -> None:
        payload = self.report()
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.get_logger().info(
            f"相机检查结束：{'通过' if payload['healthy'] else '未通过'}，"
            f"报告已写入 {self.report_path}"
        )


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = CameraCalibrationCheck()
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
