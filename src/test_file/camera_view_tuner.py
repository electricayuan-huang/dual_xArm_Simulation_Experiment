#!/usr/bin/env python3
"""双 xArm 仿真中的 RGB 相机视角与视场角调试器。

相机属于生成的机器人模型，修改固定关节或视场角需要重新启动 Gazebo。
滑块和“平视回位”只修改待确认参数，点击“确认并保存”后才会应用。

请在已加载 ROS 2 环境的终端中运行：

    python3 src/test_file/camera_view_tuner.py

操作方式：
    鼠标     点击预览中的“确认并保存”或“平视回位”按钮
    c/回车   确认、应用并保存当前参数
    l        将横滚角和俯仰角重置为 0 度（平视）
    q/Esc    退出并停止仿真
    p        打印当前启动参数
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


WINDOW_NAME = "双 xArm 相机视角调试器"
CONFIRM_BUTTON = "确认并保存"
LEVEL_BUTTON = "平视回位"
CHINESE_FONT_PATHS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)


@dataclass
class CameraSettings:
    """传递给 dual_xarm_table_gazebo.launch.py 的参数。"""

    x: float = 0.0
    y: float = -0.42
    z: float = 1.015
    roll_deg: float = 0.0
    pitch_deg: float = math.degrees(0.70)
    yaw_deg: float = math.degrees(1.5708)
    fov_deg: float = math.degrees(1.57)

    def xyz_arg(self) -> str:
        return f"{self.x:.4f} {self.y:.4f} {self.z:.4f}"

    def rpy_arg(self) -> str:
        return " ".join(
            f"{math.radians(angle):.6f}"
            for angle in (self.roll_deg, self.pitch_deg, self.yaw_deg)
        )

    def fov_rad(self) -> float:
        return math.radians(self.fov_deg)

    def launch_arguments(self) -> list[str]:
        return [
            f"fixed_camera_xyz:={self.xyz_arg()}",
            f"fixed_camera_rpy:={self.rpy_arg()}",
            f"fixed_camera_horizontal_fov:={self.fov_rad():.6f}",
        ]

    def focal_length_px(self, image_width: int = 640) -> float:
        return image_width / (2.0 * math.tan(self.fov_rad() / 2.0))

    def update_from_preset(self, payload: dict[str, object]) -> None:
        """Load the persisted degree values, with launch fields as fallback."""
        xyz = str(payload.get("fixed_camera_xyz", "")).split()
        if len(xyz) == 3:
            self.x, self.y, self.z = (float(value) for value in xyz)

        rpy = str(payload.get("fixed_camera_rpy", "")).split()
        if len(rpy) == 3:
            self.roll_deg, self.pitch_deg, self.yaw_deg = (
                math.degrees(float(value)) for value in rpy
            )

        if "roll_deg" in payload:
            self.roll_deg = float(payload["roll_deg"])
        if "pitch_deg" in payload:
            self.pitch_deg = float(payload["pitch_deg"])
        if "yaw_deg" in payload:
            self.yaw_deg = float(payload["yaw_deg"])

        if "horizontal_fov_deg" in payload:
            self.fov_deg = float(payload["horizontal_fov_deg"])
        elif "fixed_camera_horizontal_fov" in payload:
            self.fov_deg = math.degrees(float(payload["fixed_camera_horizontal_fov"]))


class CameraImageSubscriber(Node):
    """仅保留最新 RGB 帧，用于 OpenCV 预览。"""

    def __init__(self, topic: str) -> None:
        super().__init__("camera_view_tuner")
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._last_error = ""
        self.create_subscription(
            Image,
            topic,
            self._image_callback,
            qos_profile_sensor_data,
        )

    def _image_callback(self, message: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            with self._lock:
                self._latest_frame = frame
                self._last_error = ""
        except Exception as exc:  # pragma: no cover - 取决于图像消息编码
            with self._lock:
                self._last_error = str(exc)

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def status_error(self) -> str:
        with self._lock:
            return self._last_error


class SimulationProcess:
    """以一个进程组启动和停止完整任务仿真。"""

    def __init__(self, settings: CameraSettings, gz_type: str) -> None:
        self.settings = settings
        self.gz_type = gz_type
        self.process: Optional[subprocess.Popen[bytes]] = None

    def command(self) -> list[str]:
        return [
            "ros2",
            "launch",
            "dual_xarm_task",
            "dual_xarm_table_gazebo.launch.py",
            *self.settings.launch_arguments(),
            f"gz_type:={self.gz_type}",
            "use_camera_preset:=false",
            "show_rviz:=false",
            "no_gui_ctrl:=true",
        ]

    def restart(self) -> None:
        self.stop()
        print("\n正在使用以下参数启动仿真:")
        print("  " + " ".join(self.command()))
        try:
            self.process = subprocess.Popen(
                self.command(),
                env=os.environ.copy(),
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "未找到 ros2。请先加载 /opt/ros/humble/setup.bash 和 "
                "/home/ayuan/dual_xarm_ros2_ws/install/setup.bash。"
            ) from exc

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return

        process_group = os.getpgid(self.process.pid)
        try:
            os.killpg(process_group, signal.SIGINT)
            self.process.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            os.killpg(process_group, signal.SIGTERM)
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                os.killpg(process_group, signal.SIGKILL)
                self.process.wait()
        finally:
            self.process = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None


class CameraViewTuner:
    """OpenCV 滑块界面与确认重启仿真控制。"""

    def __init__(
        self,
        subscriber: CameraImageSubscriber,
        settings: CameraSettings,
        simulation: SimulationProcess,
        preset_path: Path,
    ) -> None:
        self.subscriber = subscriber
        self.settings = settings
        self.simulation = simulation
        self.preset_path = preset_path
        self._pending_changes = False
        self._button_rects: dict[str, tuple[int, int, int, int]] = {}
        self._last_message = ""
        self._text_font = self._load_chinese_font(16)
        self._button_font = self._load_chinese_font(18)
        self._load_preset()

    def _load_preset(self) -> None:
        if not self.preset_path.exists():
            self._last_message = f"未找到预设，使用默认参数：{self.preset_path}"
            return
        try:
            payload = json.loads(self.preset_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("预设内容不是 JSON 对象")
            self.settings.update_from_preset(payload)
            self._last_message = f"已加载上次保存的预设：{self.preset_path}"
            print(f"已加载相机预设：{self.preset_path}")
            self.print_settings()
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._last_message = f"预设加载失败，使用默认参数：{exc}"
            print(self._last_message, file=sys.stderr)

    def _preset_targets(self) -> list[Path]:
        targets = [self.preset_path]
        default_source = Path(__file__).with_name("camera_view_tuner_preset.json")
        workspace_root = Path(__file__).resolve().parents[2]
        installed = (
            workspace_root
            / "install/dual_xarm_task/share/dual_xarm_task/test_file"
            / "camera_view_tuner_preset.json"
        )
        if self.preset_path.resolve() == default_source.resolve():
            targets.append(installed)
        return list(dict.fromkeys(targets))

    @staticmethod
    def _load_chinese_font(size: int) -> ImageFont.FreeTypeFont:
        for font_path in CHINESE_FONT_PATHS:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
        raise RuntimeError("未找到支持中文的字体，请安装 Noto Sans CJK 字体。")

    def _trackbar_changed(self, _value: int) -> None:
        self._pending_changes = True

    def _set_level_view(self) -> None:
        self.settings.roll_deg = 0.0
        self.settings.pitch_deg = 0.0
        cv2.setTrackbarPos("横滚角（度）", WINDOW_NAME, 180)
        cv2.setTrackbarPos("俯仰角（度）", WINDOW_NAME, 90)
        self._last_message = "已选择平视：横滚角=0 度，俯仰角=0 度"
        self._pending_changes = True

    def _save_preset(self) -> None:
        payload = {
            "fixed_camera_xyz": self.settings.xyz_arg(),
            "fixed_camera_rpy": self.settings.rpy_arg(),
            "fixed_camera_horizontal_fov": f"{self.settings.fov_rad():.6f}",
            "roll_deg": round(self.settings.roll_deg, 3),
            "pitch_deg": round(self.settings.pitch_deg, 3),
            "yaw_deg": round(self.settings.yaw_deg, 3),
            "horizontal_fov_deg": round(self.settings.fov_deg, 3),
        }
        serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        for target in self._preset_targets():
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(serialized, encoding="utf-8")
                print(f"已保存相机预设：{target}")
            except OSError as exc:
                if target == self.preset_path:
                    raise RuntimeError(f"无法保存相机预设 {target}: {exc}") from exc
                print(f"警告：无法同步安装目录预设 {target}: {exc}", file=sys.stderr)

    def _confirm(self) -> None:
        self._read_trackbars()
        try:
            self._save_preset()
            self.simulation.restart()
        except RuntimeError as exc:
            self._last_message = str(exc)
            print(self._last_message, file=sys.stderr)
            return
        self._pending_changes = False
        self._last_message = "已确认、保存并重启仿真"
        self.print_settings()

    def _mouse_callback(
        self,
        event: int,
        x: int,
        y: int,
        _flags: int,
        _data: object,
    ) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return
        for name, (left, top, right, bottom) in self._button_rects.items():
            if left <= x <= right and top <= y <= bottom:
                if name == CONFIRM_BUTTON:
                    self._confirm()
                elif name == LEVEL_BUTTON:
                    self._set_level_view()
                return

    def _create_trackbars(self) -> None:
        cv2.createTrackbar("支架 X (毫米)", WINDOW_NAME, 1000, 2000, self._trackbar_changed)
        cv2.createTrackbar("支架 Y (毫米)", WINDOW_NAME, 580, 1500, self._trackbar_changed)
        cv2.createTrackbar("支架 Z (毫米)", WINDOW_NAME, 1015, 1600, self._trackbar_changed)
        cv2.createTrackbar("横滚角（度）", WINDOW_NAME, 180, 360, self._trackbar_changed)
        cv2.createTrackbar("俯仰角（度）", WINDOW_NAME, int(round(self.settings.pitch_deg + 90)), 180, self._trackbar_changed)
        cv2.createTrackbar("偏航角（度）", WINDOW_NAME, int(round(self.settings.yaw_deg + 180)), 360, self._trackbar_changed)
        cv2.createTrackbar("水平视场角（度）", WINDOW_NAME, int(round(self.settings.fov_deg)), 140, self._trackbar_changed)

    def _read_trackbars(self) -> None:
        self.settings.x = (cv2.getTrackbarPos("支架 X (毫米)", WINDOW_NAME) - 1000) / 1000.0
        self.settings.y = (cv2.getTrackbarPos("支架 Y (毫米)", WINDOW_NAME) - 1000) / 1000.0
        self.settings.z = cv2.getTrackbarPos("支架 Z (毫米)", WINDOW_NAME) / 1000.0
        self.settings.roll_deg = cv2.getTrackbarPos("横滚角（度）", WINDOW_NAME) - 180
        self.settings.pitch_deg = cv2.getTrackbarPos("俯仰角（度）", WINDOW_NAME) - 90
        self.settings.yaw_deg = cv2.getTrackbarPos("偏航角（度）", WINDOW_NAME) - 180
        self.settings.fov_deg = max(
            40.0,
            float(cv2.getTrackbarPos("水平视场角（度）", WINDOW_NAME)),
        )

    def _draw_preview(self, frame: Optional[np.ndarray]) -> np.ndarray:
        if frame is None:
            preview = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            preview = frame

        preview_width = preview.shape[1]
        button_width = 210
        button_height = 42
        button_left = preview_width - button_width - 16
        self._button_rects = {
            CONFIRM_BUTTON: (button_left, 12, preview_width - 16, 12 + button_height),
            LEVEL_BUTTON: (button_left, 64, preview_width - 16, 64 + button_height),
        }

        lines = [
            *(["等待 /camera/color/image_raw 图像..."] if frame is None else []),
            f"支架 xyz：{self.settings.xyz_arg()} 米",
            f"姿态角：横滚 {self.settings.roll_deg:.1f}，俯仰 {self.settings.pitch_deg:.1f}，偏航 {self.settings.yaw_deg:.1f} 度",
            f"水平视场角：{self.settings.fov_deg:.1f} 度，等效焦距≈{self.settings.focal_length_px():.1f} 像素",
            "点击按钮 | c/回车确认 | l 平视回位 | q/Esc 退出",
        ]
        if self._pending_changes:
            lines.append("参数待确认：点击“确认并保存”后才会应用")
        elif not self.simulation.is_running():
            lines.append("仿真未运行")
        if self._last_message:
            lines.append(self._last_message)

        # OpenCV 的 Hershey 字体不支持中文，使用 Pillow 的 CJK 字体绘制叠加层。
        pil_image = PILImage.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        self._draw_button(draw, CONFIRM_BUTTON, self._button_rects[CONFIRM_BUTTON])
        self._draw_button(draw, LEVEL_BUTTON, self._button_rects[LEVEL_BUTTON])
        for index, line in enumerate(lines):
            draw.text(
                (12, 8 + index * 24),
                line,
                font=self._text_font,
                fill=(0, 255, 0),
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
        return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    def _draw_button(
        self,
        draw: ImageDraw.ImageDraw,
        label: str,
        rect: tuple[int, int, int, int],
    ) -> None:
        left, top, right, bottom = rect
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=4,
            fill=(35, 35, 35),
            outline=(0, 220, 255),
            width=2,
        )
        text_box = draw.textbbox((0, 0), label, font=self._button_font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x = left + (right - left - text_width) // 2 - text_box[0]
        text_y = top + (bottom - top - text_height) // 2 - text_box[1]
        draw.text(
            (text_x, text_y),
            label,
            font=self._button_font,
            fill=(255, 255, 255),
        )

    def print_settings(self) -> None:
        print("\n当前相机启动参数：")
        for argument in self.settings.launch_arguments():
            print(f"  {argument}")
        print(f"  640 像素宽度下的估算焦距：{self.settings.focal_length_px():.2f} 像素")

    def run(self) -> None:
        if os.name != "posix":
            raise RuntimeError("当前调试器需要在 POSIX 环境中运行。")
        if shutil.which("ros2") is None:
            raise RuntimeError(
                "未找到 ros2。请先加载 ROS 2 和双臂工作空间的 setup 文件。"
            )

        # 保持传感器图像尺寸，确保按钮区域与 OpenCV 鼠标坐标一致。
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)
        self._create_trackbars()
        self._pending_changes = False
        self.simulation.restart()

        try:
            while rclpy.ok():
                self._read_trackbars()

                preview = self._draw_preview(self.subscriber.latest_frame())
                cv2.imshow(WINDOW_NAME, preview)
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    break
                if key in (ord("c"), 13):
                    self._confirm()
                elif key == ord("l"):
                    self._set_level_view()
                elif key == ord("p"):
                    self.print_settings()
        finally:
            self.simulation.stop()
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        default="/camera/color/image_raw",
        help="预览窗口显示的彩色图像话题",
    )
    parser.add_argument(
        "--gz-type",
        choices=("ignition", "gz", "gazebo"),
        default="ignition",
        help="任务启动使用的仿真后端",
    )
    parser.add_argument(
        "--preset",
        type=Path,
        default=Path(__file__).with_name("camera_view_tuner_preset.json"),
        help="确认操作写入的预设文件路径",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    subscriber = CameraImageSubscriber(args.topic)
    spin_thread = threading.Thread(target=rclpy.spin, args=(subscriber,), daemon=True)
    spin_thread.start()

    try:
        settings = CameraSettings()
        tuner = CameraViewTuner(
            subscriber=subscriber,
            settings=settings,
            simulation=SimulationProcess(settings, args.gz_type),
            preset_path=args.preset,
        )
        tuner.run()
    except RuntimeError as exc:
        print(f"相机视角调试器：{exc}", file=sys.stderr)
        return 2
    finally:
        subscriber.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
