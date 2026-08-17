#!/usr/bin/env python3
"""
手眼标定质量评估脚本（借鉴 fishros/handeye-calib 设计）

原理：标定板固定不动，通过各组样本 + 标定矩阵反算 marker 在 base 系下的位置，
      若标定准确，各组算出的 base→marker 应一致，标准差越小质量越好。

使用方法：
    python3 evaluate_handeye.py <result_yaml> <sample_file>

示例：
    python3 evaluate_handeye.py ~/.ros2/handeye_samples_result.yaml ~/.ros2/handeye_samples.txt
"""

import sys
import os
import numpy as np
import cv2


def quat_to_rmat(w, x, y, z):
    rmat = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)
    return rmat


def euler_to_rmat(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def rmat_to_rpy(rmat):
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


def load_calib(filepath):
    result = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            val = parts[1].strip()
            try:
                result[key] = float(val)
            except ValueError:
                result[key] = val
    rmat = euler_to_rmat(result['roll'], result['pitch'], result['yaw'])
    tvec = np.array([result['x'], result['y'], result['z']], dtype=np.float64)
    return rmat, tvec, result


def transform_point(rmat, tvec, pt):
    return rmat @ pt + tvec


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 evaluate_handeye.py <result_yaml> <sample_file>")
        print("Example: python3 evaluate_handeye.py ~/.ros2/handeye_samples_result.yaml"
              " ~/.ros2/handeye_samples.txt")
        sys.exit(1)

    calib_path = os.path.expanduser(sys.argv[1])
    sample_path = os.path.expanduser(sys.argv[2])

    eef2cam_rmat, eef2cam_tvec, calib_info = load_calib(calib_path)
    print(f"\n{'='*60}")
    print(f"  手眼标定质量评估")
    print(f"{'='*60}")
    print(f"  标定文件 : {calib_path}")
    print(f"  样本文件 : {sample_path}")
    print(f"  算法     : {calib_info.get('algorithm', 'unknown')}")
    print(f"  样本数   : {int(calib_info.get('num_samples', 0))}")
    print(f"\n  eef -> camera (标定结果):")
    print(f"    trans (m):  x={eef2cam_tvec[0]:.6f}  y={eef2cam_tvec[1]:.6f}  z={eef2cam_tvec[2]:.6f}")
    r, p, y = rmat_to_rpy(eef2cam_rmat)
    print(f"    rot RPY (deg):  roll={np.degrees(r):.3f}  pitch={np.degrees(p):.3f}  yaw={np.degrees(y):.3f}")

    # Parse samples
    base2eef_rmats = []
    base2eef_tvecs = []
    cam2marker_rmats = []
    cam2marker_tvecs = []

    with open(sample_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 14:
                continue
            vals = [float(x) for x in parts]

            # EEF in base (eef_x eef_y eef_z eef_qx eef_qy eef_qz eef_qw)
            rmat_b2e = quat_to_rmat(vals[6], vals[3], vals[4], vals[5])
            tvec_b2e = np.array(vals[0:3], dtype=np.float64)
            base2eef_rmats.append(rmat_b2e)
            base2eef_tvecs.append(tvec_b2e)

            # Marker in camera (marker_x marker_y marker_z marker_qx marker_qy marker_qz marker_qw)
            rmat_c2m = quat_to_rmat(vals[13], vals[10], vals[11], vals[12])
            tvec_c2m = np.array(vals[7:10], dtype=np.float64)
            cam2marker_rmats.append(rmat_c2m)
            cam2marker_tvecs.append(tvec_c2m)

    n = len(base2eef_rmats)
    if n == 0:
        print("\n  错误：未找到有效样本数据\n")
        sys.exit(1)

    # Compute base -> marker for each sample
    # base_to_marker = base_to_eef * eef_to_camera * camera_to_marker
    markers = []
    for i in range(n):
        b2m_tvec = base2eef_tvecs[i] + base2eef_rmats[i] @ (
            eef2cam_tvec + eef2cam_rmat @ cam2marker_tvecs[i])
        b2m_rmat = base2eef_rmats[i] @ eef2cam_rmat @ cam2marker_rmats[i]
        markers.append(b2m_tvec)

    markers = np.array(markers)

    # Statistics
    mean_xyz = np.mean(markers, axis=0)
    std_xyz = np.std(markers, axis=0)
    var_xyz = np.var(markers, axis=0)

    print(f"\n{'='*60}")
    print(f"  base -> marker 计算结果（各组采样）")
    print(f"{'='*60}")
    for i in range(n):
        dist = np.linalg.norm(markers[i] - mean_xyz)
        outlier = " *** OUTLIER" if dist > 2.0 * np.linalg.norm(std_xyz) else ""
        print(f"  #{i:2d}: ({markers[i][0]:.4f}, {markers[i][1]:.4f}, {markers[i][2]:.4f}) m"
              f"  dist_to_mean={dist:.4f}{outlier}")

    print(f"\n{'='*60}")
    print(f"  统计")
    print(f"{'='*60}")
    print(f"  {'':>8s} {'x (m)':>12s} {'y (m)':>12s} {'z (m)':>12s}")
    print(f"  {'mean':>8s} {mean_xyz[0]:12.6f} {mean_xyz[1]:12.6f} {mean_xyz[2]:12.6f}")
    print(f"  {'var':>8s} {var_xyz[0]:12.8f} {var_xyz[1]:12.8f} {var_xyz[2]:12.8f}")
    print(f"  {'std':>8s} {std_xyz[0]:12.6f} {std_xyz[1]:12.6f} {std_xyz[2]:12.6f}")

    # Overall quality
    total_std = np.linalg.norm(std_xyz)
    print(f"\n  {'综合偏差':>8s} {total_std:12.6f} m")

    if total_std < 0.005:
        verdict = "EXCELLENT - 标定质量极高，可直接用于抓取"
    elif total_std < 0.015:
        verdict = "GOOD - 标定质量良好"
    elif total_std < 0.030:
        verdict = "ACCEPTABLE - 基本可用"
    else:
        verdict = "POOR - 建议增加样本或重新标定"

    print(f"\n  评价: {verdict}")

    # Outlier summary
    outliers = [i for i in range(n) if np.linalg.norm(markers[i] - mean_xyz) > 2.0 * total_std]
    if outliers:
        print(f"  异常样本 (偏离 > 2σ): {outliers}")
        print(f"  建议剔除后重新运行 compute_handeye.py 求解")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
