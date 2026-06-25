from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ConfidenceSummary:
    percent: float
    level: str
    reasons: List[str]
    should_retake_slow_motion: bool
    distance_reliable: bool


def summarize_confidence(
    base_confidence: float,
    validation_multiplier: float,
    fps: int,
    point_count: int,
    recognition_mode: str,
    validation_reasons: List[str],
) -> ConfidenceSummary:
    percent = max(0.0, min(98.0, base_confidence * validation_multiplier))
    reasons: List[str] = []
    should_retake = False

    if fps < 60:
        percent = min(percent, 42.0)
        reasons.append("视频帧率低于 60fps，帧间位移大，距离估算仅供参考。")
        should_retake = True
    elif fps == 60:
        percent = min(percent, 64.0)
        reasons.append("60fps 可粗略分析，但稳定性不如 120fps 或 240fps 慢动作。")
    else:
        reasons.append("慢动作帧率更适合追踪高尔夫球飞行。")

    if point_count < 8:
        percent = min(percent, 48.0)
        reasons.append("有效轨迹点偏少。")
    if recognition_mode.startswith("自动"):
        percent = min(percent, 72.0)
        reasons.append("自动识别未使用手动起点和 ROI，误识别风险较高。")
    else:
        reasons.append("半自动识别已启用，更有利于避开人体和背景亮点。")

    for reason in validation_reasons:
        if reason not in reasons:
            reasons.append(reason)

    if percent >= 75:
        level = "高"
    elif percent >= 52:
        level = "中"
    elif fps < 60:
        level = "低，仅供参考"
    else:
        level = "低"

    distance_reliable = percent >= 52 and point_count >= 5
    if fps < 60:
        distance_reliable = point_count >= 5 and percent >= 28

    should_retake = should_retake or percent < 52
    return ConfidenceSummary(
        percent=round(percent, 1),
        level=level,
        reasons=reasons,
        should_retake_slow_motion=should_retake,
        distance_reliable=distance_reliable,
    )
