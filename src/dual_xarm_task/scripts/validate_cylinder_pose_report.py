#!/usr/bin/env python3
"""Validate one cylinder RGB-D estimate against the held-pose reference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


DEFAULT_CENTER = np.asarray([-0.016463, 0.262657, 1.467054], dtype=float)
DEFAULT_ENTRANCE = np.asarray([-0.013536, 0.202737, 1.466051], dtype=float)
DEFAULT_AXIS = np.asarray([-0.048783, 0.998670, 0.016713], dtype=float)


def read_vector(values: str) -> np.ndarray:
    try:
        vector = np.asarray([float(value) for value in values.split(",")], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if vector.shape != (3,):
        raise argparse.ArgumentTypeError("向量必须包含三个逗号分隔的数值")
    return vector


def angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    cosine = max(-1.0, min(1.0, float(np.dot(first, second))))
    return math.degrees(math.acos(cosine))


def main() -> int:
    parser = argparse.ArgumentParser(description="验收 RGB-D 圆柱位姿报告")
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-center", type=read_vector)
    parser.add_argument("--expected-entrance", type=read_vector)
    parser.add_argument("--expected-axis", type=read_vector)
    parser.add_argument("--max-center-mm", type=float, default=5.0)
    parser.add_argument("--max-entrance-mm", type=float, default=5.0)
    parser.add_argument("--max-axis-deg", type=float, default=3.0)
    parser.add_argument("--min-points", type=int, default=50)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    args = parser.parse_args()

    try:
        payload = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"读取报告失败: {exc}", file=sys.stderr)
        return 2

    estimate = payload.get("estimate") or {}
    if not estimate:
        print("[失败] 报告中没有有效圆柱估计")
        return 1
    expected_center = args.expected_center if args.expected_center is not None else DEFAULT_CENTER
    expected_entrance = (
        args.expected_entrance if args.expected_entrance is not None else DEFAULT_ENTRANCE
    )
    expected_axis = args.expected_axis if args.expected_axis is not None else DEFAULT_AXIS
    try:
        center = np.asarray(estimate["center_output_m"], dtype=float)
        entrance = np.asarray(estimate["entrance_output_m"], dtype=float)
        axis = np.asarray(estimate["axis_output"], dtype=float)
        point_count = int(estimate["depth_point_count"])
        confidence = float(estimate["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        print(f"[失败] 报告字段不完整: {exc}")
        return 1

    center_error_mm = 1000.0 * float(np.linalg.norm(center - expected_center))
    entrance_error_mm = 1000.0 * float(np.linalg.norm(entrance - expected_entrance))
    axis_error_deg = angle_deg(axis, expected_axis)
    checks = [
        ("healthy", bool(payload.get("healthy", False)), "报告包含有效估计"),
        ("points", point_count >= args.min_points, f"points={point_count} >= {args.min_points}"),
        (
            "confidence",
            confidence >= args.min_confidence,
            f"confidence={confidence:.3f} >= {args.min_confidence:.3f}",
        ),
        (
            "center",
            center_error_mm <= args.max_center_mm,
            f"center_error={center_error_mm:.3f}mm <= {args.max_center_mm:.3f}mm",
        ),
        (
            "entrance",
            entrance_error_mm <= args.max_entrance_mm,
            f"entrance_error={entrance_error_mm:.3f}mm <= {args.max_entrance_mm:.3f}mm",
        ),
        (
            "axis",
            axis_error_deg <= args.max_axis_deg,
            f"axis_error={axis_error_deg:.3f}deg <= {args.max_axis_deg:.3f}deg",
        ),
    ]
    print(f"报告: {args.report}")
    for name, passed, detail in checks:
        print(f"[{'通过' if passed else '失败'}] {name}: {detail}")
    return 0 if all(passed for _, passed, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
