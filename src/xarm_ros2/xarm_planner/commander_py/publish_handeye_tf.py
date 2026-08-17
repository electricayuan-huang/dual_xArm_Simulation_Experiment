#!/usr/bin/env python3
"""
将手眼标定结果发布为静态 TF: link_eef → camera_color_optical_frame

使用方式：
    python3 publish_handeye_tf.py --ros-args -p calib_file:=<path>

默认读取 ~/.ros2/handeye_samples_result.yaml
"""

import sys
import os
import math
import yaml

import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped


class HandEyeTFPublisher(Node):
    def __init__(self):
        super().__init__("handeye_tf_publisher")

        self.declare_parameter("calib_file", os.path.expanduser("~/.ros2/handeye_samples_result.yaml"))
        calib_path = self.get_parameter("calib_file").value

        if not os.path.exists(calib_path):
            self.get_logger().error(f"Calibration file not found: {calib_path}")
            raise FileNotFoundError(calib_path)

        self.get_logger().info(f"Loading calibration from: {calib_path}")

        with open(calib_path, "r") as f:
            calib = yaml.safe_load(f)

        parent_frame = calib["parent_frame"]
        child_frame = calib["child_frame"]
        t = calib["transform"]
        x, y, z = t["x"], t["y"], t["z"]
        roll, pitch, yaw = t["roll"], t["pitch"], t["yaw"]

        # RPY to quaternion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        self._broadcaster = StaticTransformBroadcaster(self)
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = parent_frame
        msg.child_frame_id = child_frame
        msg.transform.translation.x = x
        msg.transform.translation.y = y
        msg.transform.translation.z = z
        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw

        self._broadcaster.sendTransform(msg)

        self.get_logger().info("=" * 50)
        self.get_logger().info(f"  Static TF published: {parent_frame} -> {child_frame}")
        self.get_logger().info(f"  trans: ({x:.5f}, {y:.5f}, {z:.5f})")
        self.get_logger().info(f"  RPY (deg): ({math.degrees(roll):.2f}, "
                               f"{math.degrees(pitch):.2f}, {math.degrees(yaw):.2f})")
        self.get_logger().info(f"  algorithm: {calib.get('algorithm', 'unknown')}")
        self.get_logger().info("=" * 50)
        self.get_logger().info("TF chain: link_base -> link_eef -> camera_color_optical_frame -> "
                               "camera_depth_optical_frame -> obj_xxx")
        self.get_logger().info("Node will keep running to maintain the static TF. Ctrl+C to stop.")


def main():
    rclpy.init()
    try:
        node = HandEyeTFPublisher()
        rclpy.spin(node)
    except FileNotFoundError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
