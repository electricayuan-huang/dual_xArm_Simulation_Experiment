#!/usr/bin/env python3
"""
Offline hand-eye calibration computation.

Reads sample data from the recorder output file and solves AX=XB
using OpenCV's calibrateHandEye.

Usage:
    python3 compute_handeye.py <sample_file>

Sample file format (one line per sample):
    eef_x eef_y eef_z eef_qx eef_qy eef_qz eef_qw  marker_x marker_y marker_z marker_qx marker_qy marker_qz marker_qw

Output:
    eye_in_hand transform: eef -> camera_color_optical_frame
"""

import sys
import os
import numpy as np
import cv2


def quat_to_rvec_tvec(x, y, z, qx, qy, qz, qw):
    """Convert quaternion + translation to Rodrigues vector + translation."""
    # quaternion to rotation matrix
    rmat = np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float64)
    rvec = cv2.Rodrigues(rmat)[0]
    tvec = np.array([x, y, z], dtype=np.float64).reshape(3, 1)
    return rvec, tvec


def rvec_tvec_to_pose(rvec, tvec):
    """Convert Rodrigues vector + translation to (x,y,z, qx,qy,qz,qw)."""
    rmat = cv2.Rodrigues(rvec)[0]
    q = cv2.Rodrigues(rmat)[0]
    # q from Rodrigues is (w, x, y, z) -> convert to (x, y, z, w)
    return (float(tvec[0]), float(tvec[1]), float(tvec[2]),
            float(q[1]), float(q[2]), float(q[3]), float(q[0]))


def compute_rpy(rmat):
    """Compute roll-pitch-yaw from rotation matrix."""
    sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rmat[2, 1], rmat[2, 2])
        pitch = np.arctan2(-rmat[2, 0], sy)
        yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        roll = np.arctan2(-rmat[1, 2], rmat[1, 1])
        pitch = np.arctan2(-rmat[2, 0], sy)
        yaw = 0
    return roll, pitch, yaw


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compute_handeye.py <sample_file>")
        sys.exit(1)

    filepath = os.path.expanduser(sys.argv[1])
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 14:
                print(f"  SKIP invalid line: {line[:60]}...")
                continue
            vals = [float(x) for x in parts]

            # EEF in base frame (gripper2base)
            rv_gb, tv_gb = quat_to_rvec_tvec(*vals[0:7])
            R_gripper2base.append(rv_gb)
            t_gripper2base.append(tv_gb)

            # Marker in camera frame (target2cam)
            rv_tc, tv_tc = quat_to_rvec_tvec(*vals[7:14])
            R_target2cam.append(rv_tc)
            t_target2cam.append(tv_tc)

    n = len(R_gripper2base)
    if n < 3:
        print(f"ERROR: Need >= 3 samples, got {n}")
        sys.exit(1)

    print(f"Loaded {n} samples from {filepath}\n")

    # ------------------------------------------------------------------
    # Solve: calibrateHandEye(method=Tsai)
    #   A(i) = gripper2base  (eef in base frame)
    #   B(i) = target2cam    (marker in camera frame)
    #   Output X = cam2gripper (camera in eef frame)
    # ------------------------------------------------------------------
    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "Tsai-Lenz"),
        (cv2.CALIB_HAND_EYE_PARK, "Park-Martin"),
        (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
        (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
        (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis"),
    ]

    solutions = []
    for method, name in methods:
        try:
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam,   t_target2cam,
                method=method)
        except cv2.error as e:
            print(f"  [{name}] FAILED: {e}")
            continue

        # Select the solution using the same fixed-marker closure criterion
        # used by evaluate_handeye.py, rather than assuming Tsai is best.
        base_to_marker = []
        for i in range(n):
            r_gripper2base = cv2.Rodrigues(R_gripper2base[i])[0]
            r_target2cam = cv2.Rodrigues(R_target2cam[i])[0]
            t_base_to_marker = t_gripper2base[i].reshape(3) + r_gripper2base @ (
                t_cam2gripper.reshape(3) + R_cam2gripper @ t_target2cam[i].reshape(3))
            base_to_marker.append(t_base_to_marker)
        closure_std = float(np.linalg.norm(np.std(np.asarray(base_to_marker), axis=0)))
        solutions.append((closure_std, name, R_cam2gripper, t_cam2gripper))

        # cam2gripper = camera -> eef
        rmat_c2g = R_cam2gripper
        roll, pitch, yaw = compute_rpy(rmat_c2g)
        print(f"  [{name}] cam_to_eef:")
        print(f"    trans (m):  x={t_cam2gripper[0][0]:.6f}  y={t_cam2gripper[1][0]:.6f}  z={t_cam2gripper[2][0]:.6f}")
        print(f"    rot RPY (rad):  roll={roll:.6f}  pitch={pitch:.6f}  yaw={yaw:.6f}")
        print(f"    rot RPY (deg):  roll={np.degrees(roll):.3f}  pitch={np.degrees(pitch):.3f}  yaw={np.degrees(yaw):.3f}")

        # Inverse: eef -> camera (for TF publishing)
        rmat_e2c = rmat_c2g.T
        t_e2c = -rmat_c2g.T @ t_cam2gripper
        roll2, pitch2, yaw2 = compute_rpy(rmat_e2c)
        print(f"  [{name}] eef_to_camera (publish as static TF eef->camera):")
        print(f"    trans (m):  x={t_e2c[0][0]:.6f}  y={t_e2c[1][0]:.6f}  z={t_e2c[2][0]:.6f}")
        print(f"    rot RPY (rad):  roll={roll2:.6f}  pitch={pitch2:.6f}  yaw={yaw2:.6f}")
        print(f"    rot RPY (deg):  roll={np.degrees(roll2):.3f}  pitch={np.degrees(pitch2):.3f}  yaw={np.degrees(yaw2):.3f}")
        print(f"    closure std (m): {closure_std:.6f}")
        print()

    if not solutions:
        print("ERROR: All hand-eye calibration methods failed")
        sys.exit(1)

    # Save the method with the smallest fixed-marker closure error.
    closure_std, selected_name, R_cam2gripper, t_cam2gripper = min(
        solutions, key=lambda solution: solution[0])
    print(f"Selected method: {selected_name} (closure std {closure_std:.6f} m)")

    try:
        rmat_c2g = R_cam2gripper
        roll, pitch, yaw = compute_rpy(rmat_c2g)

        outpath = filepath.rsplit('.', 1)[0] + '_result.yaml'
        with open(outpath, 'w') as f:
            f.write(f"eye_in_hand: true\n")
            f.write(f"parent_frame: link_eef\n")
            f.write(f"child_frame: camera_color_optical_frame\n")
            f.write(f"transform:\n")
            f.write(f"  x: {t_cam2gripper[0][0]:.8f}\n")
            f.write(f"  y: {t_cam2gripper[1][0]:.8f}\n")
            f.write(f"  z: {t_cam2gripper[2][0]:.8f}\n")
            f.write(f"  roll: {roll:.8f}\n")
            f.write(f"  pitch: {pitch:.8f}\n")
            f.write(f"  yaw: {yaw:.8f}\n")
            f.write(f"num_samples: {n}\n")
            f.write(f"algorithm: {selected_name}\n")
            f.write(f"closure_std_m: {closure_std:.8f}\n")
        print(f"Result saved to {outpath}")
    except cv2.error as e:
        print(f"Save failed: {e}")


if __name__ == '__main__':
    main()
