#!/usr/bin/env python3
"""
Set initial joint position for UF_ROBOT in Ignition Gazebo.
Called after spawn to move joint5 to hold-up pose (-1.5708 rad).
Communicates via Ignition Transport directly to the Gazebo server.
"""
import sys
import time
import argparse

try:
    from ignition.msgs.joint_cmd_pb2 import JointCmd
    from ignition.transport import Node
    HAS_IGNITION = True
except ImportError:
    HAS_IGNITION = False


def set_joint_via_cmd_pos(joint_name, position, timeout=15):
    """Publish joint position command via Ignition topic."""
    node = Node()
    topic = f"/model/UF_ROBOT/joint/{joint_name}/cmd_pos"
    pub = node.advertise(topic, JointCmd)
    
    msg = JointCmd()
    msg.name = joint_name
    msg.position = float(position)
    
    start = time.time()
    while time.time() - start < timeout:
        pub.publish(msg)
        print(f"Sent: {joint_name} -> {position}")
        time.sleep(0.5)
        return True
    
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint', default='joint5')
    parser.add_argument('--position', type=float, default=-1.5708)
    parser.add_argument('--timeout', type=int, default=15)
    args = parser.parse_args()

    if not HAS_IGNITION:
        print("Ignition Transport Python bindings not available, exiting")
        return

    print(f"Setting {args.joint} to {args.position}...")
    set_joint_via_cmd_pos(args.joint, args.position, args.timeout)


if __name__ == '__main__':
    main()
