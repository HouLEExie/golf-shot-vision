from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from tracking.trajectory_cleaner import TrajectoryPoint


@dataclass
class TrajectoryValidation:
    is_valid: bool
    is_stable: bool
    confidence_multiplier: float
    reasons: List[str]
    warnings: List[str]


def _direction_reversal_count(vectors: np.ndarray, direction: str) -> int:
    if len(vectors) == 0:
        return 0
    if direction == "右到左":
        return int(np.sum(vectors[:, 0] > 1.0))
    if direction == "向上":
        return int(np.sum(vectors[:, 1] > 1.0))
    return int(np.sum(vectors[:, 0] < -1.0))


def validate_trajectory(
    points: List[TrajectoryPoint],
    fps: int,
    flight_direction: str,
    min_step_px: float,
    max_step_px: float,
) -> TrajectoryValidation:
    reasons: List[str] = []
    warnings: List[str] = []

    if len(points) < 5:
        return TrajectoryValidation(
            is_valid=False,
            is_stable=False,
            confidence_multiplier=0.0,
            reasons=["轨迹点少于 5 个"],
            warnings=["当前视频未能稳定识别高尔夫球轨迹。建议使用半自动识别模式，手动设置起始帧、起始球位置和搜索区域。"],
        )

    coords = np.array([[point.x, point.y] for point in points], dtype=float)
    vectors = np.diff(coords, axis=0)
    distances = np.linalg.norm(vectors, axis=1)
    large_jump_count = int(np.sum(distances > max_step_px * 1.12))
    tiny_step_count = int(np.sum(distances < max(0.0, min_step_px * 0.65)))
    reversal_count = _direction_reversal_count(vectors, flight_direction)
    median_step = float(np.median(distances)) if len(distances) else 0.0
    jump_ratio = float(np.max(distances) / max(1.0, median_step)) if len(distances) else 0.0
    vertical_jumps = int(np.sum(np.abs(vectors[:, 1]) > max_step_px * 0.85)) if len(vectors) else 0
    y_span = float(np.max(coords[:, 1]) - np.min(coords[:, 1]))
    x_span = float(np.max(coords[:, 0]) - np.min(coords[:, 0]))

    multiplier = 1.0
    if large_jump_count:
        reasons.append(f"存在 {large_jump_count} 次异常大跳变")
        multiplier *= 0.62
    if reversal_count:
        reasons.append(f"存在 {reversal_count} 次方向反转")
        multiplier *= 0.68
    if jump_ratio > 3.5:
        reasons.append("轨迹速度变化过大")
        multiplier *= 0.66
    if vertical_jumps >= 2:
        reasons.append("相邻帧垂直跳变过大")
        multiplier *= 0.72
    if tiny_step_count > max(2, len(distances) // 2):
        reasons.append("轨迹长期贴近原地或地面亮点")
        multiplier *= 0.72
    if x_span < 18 and y_span < 18:
        reasons.append("轨迹位移过小")
        multiplier *= 0.35

    if fps < 60:
        warnings.append("当前视频帧率较低，球飞行过程中帧间位移过大，系统无法稳定追踪。建议使用 iPhone 慢动作 120fps 或 240fps 重新拍摄。")
        multiplier *= 0.72
    elif fps == 60:
        warnings.append("当前为普通帧率视频，可进行粗略分析。如需更稳定识别，建议使用 120fps 或 240fps 慢动作视频。")
        multiplier *= 0.88

    is_valid = "轨迹点少于 5 个" not in reasons and "轨迹位移过小" not in reasons
    is_stable = is_valid and multiplier >= (0.56 if fps < 60 else 0.64)
    if not is_stable:
        warnings.append("轨迹不够稳定，系统不会输出高可信度距离结果。")

    return TrajectoryValidation(
        is_valid=is_valid,
        is_stable=is_stable,
        confidence_multiplier=max(0.0, min(1.0, multiplier)),
        reasons=reasons or ["轨迹连续性通过基础校验"],
        warnings=warnings,
    )
