#!/usr/bin/env python3
"""Simple deterministic R-arm pickup sequence.

The gripper is commanded with six explicit position joints.  Once the close
move has reached the workpiece, the Fortress DetachableJoint system is used
to keep the workpiece attached during lifting and reorientation.  This avoids
using a trajectory action's success status as a substitute for grasping force.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import re
import subprocess
import sys
import time
from typing import Optional

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_JOINTS = tuple(f"R_joint{index}" for index in range(1, 7))
GRIPPER_JOINTS = (
    "R_drive_joint",
    "R_left_finger_joint",
    "R_left_inner_knuckle_joint",
    "R_right_outer_knuckle_joint",
    "R_right_finger_joint",
    "R_right_inner_knuckle_joint",
)

ARM_ACTION = "/R_xarm6_traj_controller/follow_joint_trajectory"
GRIPPER_ACTION = "/R_xarm_gripper_traj_controller/follow_joint_trajectory"

ATTACH_TOPIC = "/R_hollow_cylinder/attach"
DETACH_TOPIC = "/R_hollow_cylinder/detach"
ATTACH_STATE_TOPIC = "/R_hollow_cylinder/state"
POSE_TOPIC = "/world/dual_xarm_table_ignition/pose/info"
MIN_FINAL_Z = 1.25
MIN_FINAL_ROTATION_RAD = 0.5

# This is the measured 60 mm fit for the current URDF collision boxes.
DEFAULT_CLOSE_RAD = 0.25
MIN_GRIPPER_RAD = 0.0
MAX_GRIPPER_RAD = 0.85


@dataclass(frozen=True)
class ArmStage:
    name: str
    joint_degrees: tuple[float, ...]
    duration_sec: float


STAGES = (
    ArmStage("01_pregrasp_open", (38.0, 16.0, -62.0, -64.0, 115.0, 140.0), 4.0),
    ArmStage("02_approach_open", (-2.0, -7.0, -33.0, -93.0, 88.0, 140.0), 4.0),
    ArmStage("03_grasp_pose", (-2.0, 23.0, -38.0, -93.0, 90.0, 164.0), 4.0),
    ArmStage("04_lift_hold", (-2.0, -24.0, -41.0, -92.0, 86.0, 115.0), 6.0),
    ArmStage("05_reorient_hold", (0.0, 6.0, -71.0, -92.0, 87.0, 115.0), 6.0),
    ArmStage("06_retreat", (0.0, 6.0, -71.0, -92.0, 87.0, 25.0), 6.0),
)


def duration_message(seconds: float) -> Duration:
    whole = int(seconds)
    nanoseconds = int(round((seconds - whole) * 1_000_000_000))
    if nanoseconds >= 1_000_000_000:
        whole += 1
        nanoseconds -= 1_000_000_000
    return Duration(sec=whole, nanosec=nanoseconds)


def trajectory_goal(
    joint_names: tuple[str, ...],
    positions: list[float],
    duration_sec: float,
    goal_time_tolerance_sec: float = 5.0,
) -> FollowJointTrajectory.Goal:
    point = JointTrajectoryPoint()
    point.positions = positions
    point.time_from_start = duration_message(duration_sec)

    trajectory = JointTrajectory()
    trajectory.joint_names = list(joint_names)
    trajectory.points = [point]

    goal = FollowJointTrajectory.Goal()
    goal.trajectory = trajectory
    goal.goal_time_tolerance = duration_message(goal_time_tolerance_sec)
    return goal


class Pickup:
    def __init__(
        self,
        node: Node,
        execute: bool,
        close_rad: float,
        gripper_duration_sec: float,
        pause_sec: float,
        use_attachment: bool,
    ) -> None:
        self.node = node
        self.execute = execute
        self.close_rad = close_rad
        self.gripper_duration_sec = gripper_duration_sec
        self.pause_sec = pause_sec
        self.use_attachment = use_attachment
        self.arm_client = ActionClient(node, FollowJointTrajectory, ARM_ACTION)
        self.gripper_client = ActionClient(node, FollowJointTrajectory, GRIPPER_ACTION)
        self.joint_positions: dict[str, float] = {}
        node.create_subscription(JointState, "/joint_states", self._joint_state_callback, 10)

    def _joint_state_callback(self, message: JointState) -> None:
        self.joint_positions.update(dict(zip(message.name, message.position)))

    def wait_for_controllers(self) -> bool:
        if not self.arm_client.wait_for_server(timeout_sec=15.0):
            self.node.get_logger().error("R arm trajectory action is unavailable")
            return False
        if not self.gripper_client.wait_for_server(timeout_sec=15.0):
            self.node.get_logger().error("R gripper trajectory action is unavailable")
            return False
        return True

    def send_goal(
        self,
        client: ActionClient,
        goal: FollowJointTrajectory.Goal,
        label: str,
        timeout_sec: float,
    ) -> bool:
        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, send_future, timeout_sec=15.0)
        if not send_future.done():
            self.node.get_logger().error(f"{label}: action send timed out")
            return False

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.node.get_logger().error(f"{label}: action rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=timeout_sec)
        if not result_future.done():
            self.node.get_logger().error(f"{label}: action timed out")
            goal_handle.cancel_goal_async()
            return False

        wrapped = result_future.result()
        result = wrapped.result if wrapped is not None else None
        if result is None or result.error_code != 0:
            error_code = None if result is None else result.error_code
            error_string = "" if result is None else result.error_string
            self.node.get_logger().warning(
                f"{label}: action ended with error_code={error_code} {error_string}"
            )
            return False

        self.node.get_logger().info(f"{label}: completed")
        return True

    def send_arm_stage(self, stage: ArmStage) -> bool:
        positions = [math.radians(value) for value in stage.joint_degrees]
        self.node.get_logger().info(
            f"{stage.name}: " + ", ".join(f"{value:.1f}" for value in stage.joint_degrees)
        )
        if not self.execute:
            return True
        success = self.send_goal(
            self.arm_client,
            trajectory_goal(ARM_JOINTS, positions, stage.duration_sec),
            stage.name,
            timeout_sec=60.0,
        )
        if success and self.pause_sec > 0.0:
            time.sleep(self.pause_sec)
        return success

    def send_gripper(self, angle_rad: float, label: str) -> bool:
        self.node.get_logger().info(
            f"{label}: six joints -> {math.degrees(angle_rad):.2f} deg"
        )
        if not self.execute:
            return True
        success = self.send_goal(
            self.gripper_client,
            trajectory_goal(
                GRIPPER_JOINTS,
                [angle_rad] * len(GRIPPER_JOINTS),
                self.gripper_duration_sec,
            ),
            label,
            timeout_sec=max(30.0, self.gripper_duration_sec + 12.0),
        )
        rclpy.spin_once(self.node, timeout_sec=0.2)
        actual = [self.joint_positions.get(name) for name in GRIPPER_JOINTS]
        if all(value is not None for value in actual):
            self.node.get_logger().info(
                f"{label}: actual="
                + ", ".join(f"{math.degrees(value):.2f} deg" for value in actual)
            )
        return success

    @staticmethod
    def publish_empty(topic: str) -> bool:
        try:
            completed = subprocess.run(
                [
                    "ign",
                    "topic",
                    "-t",
                    topic,
                    "-m",
                    "ignition.msgs.Empty",
                    "-p",
                    "{}",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"ign topic failed for {topic}: {exc}", file=sys.stderr)
            return False
        if completed.returncode != 0:
            print(completed.stderr.strip(), file=sys.stderr)
            return False
        return True

    def detach_workpiece(self) -> bool:
        if not self.execute or not self.use_attachment:
            return True
        self.node.get_logger().info("Resetting workpiece attachment")
        success = self.publish_empty(DETACH_TOPIC)
        time.sleep(0.3)
        return success

    def attach_workpiece(self) -> bool:
        if not self.execute or not self.use_attachment:
            return True

        # Listen before publishing because the DetachableJoint state is an
        # event, not a latched topic.
        for attempt in range(1, 4):
            listener = subprocess.Popen(
                [
                    "ign",
                    "topic",
                    "-e",
                    "-t",
                    ATTACH_STATE_TOPIC,
                    "-d",
                    "3",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                time.sleep(0.2)
                if not self.publish_empty(ATTACH_TOPIC):
                    listener.kill()
                    return False
                stdout, _ = listener.communicate(timeout=5.0)
            except (OSError, subprocess.TimeoutExpired) as exc:
                listener.kill()
                self.node.get_logger().warning(f"attachment listener failed: {exc}")
                continue

            if "attached" in stdout.lower():
                self.node.get_logger().info("Workpiece attachment confirmed")
                return True
            self.node.get_logger().warning(
                f"Attachment state not received, retry {attempt}/3"
            )

        self.node.get_logger().error("Workpiece attachment was not confirmed")
        return False

    @staticmethod
    def read_workpiece_pose() -> Optional[tuple[float, ...]]:
        try:
            completed = subprocess.run(
                ["ign", "topic", "-e", "-t", POSE_TOPIC, "-n", "1"],
                check=False,
                capture_output=True,
                text=True,
                timeout=8.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        match = re.search(
            r'name:\s*"hollow_cylinder_workpiece".*?'
            r'position\s*\{\s*'
            r'x:\s*([^\n]+)\s*y:\s*([^\n]+)\s*z:\s*([^\n]+).*?'
            r'orientation\s*\{\s*'
            r'x:\s*([^\n]+)\s*y:\s*([^\n]+)\s*z:\s*([^\n]+)\s*'
            r'w:\s*([^\n]+)',
            completed.stdout,
            re.DOTALL,
        )
        if match is None:
            return None
        return tuple(float(value.strip()) for value in match.groups())

    def verify_workpiece_motion(self) -> bool:
        if not self.execute or not self.use_attachment:
            return True
        pose = self.read_workpiece_pose()
        if pose is None:
            self.node.get_logger().error("Could not read final workpiece pose")
            return False
        x, y, z, qx, qy, qz, qw = pose
        rotation = 2.0 * math.acos(min(1.0, abs(qw)))
        self.node.get_logger().info(
            f"Final workpiece pose: x={x:.3f}, y={y:.3f}, z={z:.3f}, "
            f"rotation={math.degrees(rotation):.1f} deg"
        )
        if z < MIN_FINAL_Z or rotation < MIN_FINAL_ROTATION_RAD:
            self.node.get_logger().error(
                "Workpiece attachment did not produce the required lift and rotation"
            )
            return False
        return True

    def run(self) -> bool:
        if self.execute and not self.wait_for_controllers():
            return False

        self.node.get_logger().info(
            f"Starting simple pickup sequence, close_rad={self.close_rad:.4f}"
        )
        if not self.detach_workpiece():
            return False
        if not self.send_gripper(MIN_GRIPPER_RAD, "00_open_gripper"):
            return False

        for stage in STAGES[:3]:
            if not self.send_arm_stage(stage):
                return False

        # Contact can make a position trajectory report an error.  The actual
        # joint state is logged, then the deterministic attachment is requested.
        self.send_gripper(self.close_rad, "03_close_gripper")
        if not self.attach_workpiece():
            return False

        for stage in STAGES[3:]:
            if not self.send_arm_stage(stage):
                return False

        if not self.verify_workpiece_motion():
            return False
        self.node.get_logger().info("Pickup, lift and reorientation completed")
        return True


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple R-arm cylinder pickup")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-execute", action="store_true")
    parser.add_argument("--close-rad", type=float, default=DEFAULT_CLOSE_RAD)
    parser.add_argument("--gripper-duration", type=float, default=4.0)
    parser.add_argument("--pause-after", type=float, default=1.0)
    parser.add_argument("--no-attachment", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and not args.confirm_execute:
        parser.error("--execute requires --confirm-execute")
    if not MIN_GRIPPER_RAD <= args.close_rad <= MAX_GRIPPER_RAD:
        parser.error("--close-rad is outside the URDF joint limit")
    if args.gripper_duration <= 0.0 or args.pause_after < 0.0:
        parser.error("invalid duration or pause")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rclpy.init(args=[])
    node = Node("r_xarm_pickup")
    try:
        pickup = Pickup(
            node=node,
            execute=args.execute,
            close_rad=args.close_rad,
            gripper_duration_sec=args.gripper_duration,
            pause_sec=args.pause_after,
            use_attachment=not args.no_attachment,
        )
        return 0 if pickup.run() else 1
    except KeyboardInterrupt:
        node.get_logger().warning("Interrupted")
        return 130
    except Exception as exc:
        node.get_logger().error(f"Pickup failed: {exc}")
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
