#!/usr/bin/env python3
"""Validate one independent ChArUco report against project acceptance limits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="验收 ChArUco 独立复核报告")
    parser.add_argument("report", type=Path, help="camera_calibration_check JSON 报告")
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--min-corners", type=int, default=6)
    parser.add_argument("--max-reprojection-px", type=float, default=0.5)
    parser.add_argument("--max-std-mm", type=float, default=1.0)
    parser.add_argument("--max-position-mm", type=float, default=2.0)
    parser.add_argument("--max-angle-deg", type=float, default=1.0)
    args = parser.parse_args()

    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"读取报告失败: {exc}", file=sys.stderr)
        return 2

    charuco = report.get("charuco", {})
    detection = charuco.get("detection") or {}
    samples = int(charuco.get("samples", 0))
    corners = int(detection.get("corners", 0))
    reprojection = float(detection.get("reprojection_error_px", float("inf")))
    std_mm = 1000.0 * float(charuco.get("translation_std_norm_m", float("inf")))
    position_mm = 1000.0 * float(
        detection.get("translation_error_m", float("inf"))
    )
    angle_deg = float(detection.get("rotation_error_deg", float("inf")))

    checks = [
        ("health", bool(report.get("healthy", False)), "相机流、CameraInfo、TF 和角点检测通过"),
        ("samples", samples >= args.min_samples, f"samples={samples} >= {args.min_samples}"),
        ("corners", corners >= args.min_corners, f"corners={corners} >= {args.min_corners}"),
        ("reprojection", reprojection <= args.max_reprojection_px,
         f"reprojection={reprojection:.4f}px <= {args.max_reprojection_px:.4f}px"),
        ("stability", std_mm <= args.max_std_mm,
         f"translation_std={std_mm:.4f}mm <= {args.max_std_mm:.4f}mm"),
        ("position", position_mm <= args.max_position_mm,
         f"position_error={position_mm:.4f}mm <= {args.max_position_mm:.4f}mm"),
        ("orientation", angle_deg <= args.max_angle_deg,
         f"rotation_error={angle_deg:.4f}deg <= {args.max_angle_deg:.4f}deg"),
    ]

    print(f"报告: {args.report}")
    for name, passed, detail in checks:
        print(f"[{'通过' if passed else '失败'}] {name}: {detail}")
    return 0 if all(passed for _, passed, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
