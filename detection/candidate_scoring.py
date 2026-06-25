from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ScoredCandidate:
    frame_index: int
    x: float
    y: float
    radius: float
    area: float
    brightness: float
    score: float
    brightness_score: float
    area_score: float
    prediction_score: float
    direction_score: float
    smoothness_score: float
    velocity_score: float
    blur_score: float
    accepted: bool = False
    reason: str = ""


def direction_vector(direction: str) -> np.ndarray:
    if direction == "右到左":
        return np.array([-1.0, 0.0])
    if direction == "向上":
        return np.array([0.0, -1.0])
    return np.array([1.0, 0.0])


def _norm(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return max(0.0, min(1.0, (value - lower) / (upper - lower)))


def _area_score(area: float, low_fps: bool) -> float:
    ideal = 42.0 if not low_fps else 84.0
    tolerance = 90.0 if not low_fps else 220.0
    return max(0.0, 1.0 - abs(area - ideal) / tolerance)


def _prediction_score(
    x: float,
    y: float,
    predicted_x: Optional[float],
    predicted_y: Optional[float],
    max_step_px: float,
) -> float:
    if predicted_x is None or predicted_y is None:
        return 0.55
    distance = math.hypot(x - predicted_x, y - predicted_y)
    return max(0.0, 1.0 - distance / max(1.0, max_step_px))


def _direction_score(
    x: float,
    y: float,
    previous_x: Optional[float],
    previous_y: Optional[float],
    direction: str,
) -> float:
    if previous_x is None or previous_y is None:
        return 0.65

    movement = np.array([x - previous_x, y - previous_y], dtype=float)
    distance = float(np.linalg.norm(movement))
    if distance <= 0:
        return 0.0

    expected = direction_vector(direction)
    cosine = float(np.dot(movement / distance, expected))
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def _smoothness_score(
    x: float,
    y: float,
    predicted_x: Optional[float],
    predicted_y: Optional[float],
    smoothness: float,
    max_step_px: float,
) -> float:
    if predicted_x is None or predicted_y is None:
        return 0.65
    distance = math.hypot(x - predicted_x, y - predicted_y)
    tolerance = max(2.0, max_step_px * max(0.1, smoothness))
    return max(0.0, 1.0 - distance / tolerance)


def _velocity_score(
    x: float,
    y: float,
    previous_x: Optional[float],
    previous_y: Optional[float],
    previous_velocity: Optional[np.ndarray],
    min_step_px: float,
    max_step_px: float,
) -> float:
    if previous_x is None or previous_y is None:
        return 0.65

    movement = np.array([x - previous_x, y - previous_y], dtype=float)
    distance = float(np.linalg.norm(movement))
    if distance > max_step_px:
        return 0.0
    if distance < min_step_px:
        return 0.12

    range_score = 1.0 - abs(distance - ((min_step_px + max_step_px) / 2.0)) / max(1.0, max_step_px)
    range_score = max(0.15, min(1.0, range_score))

    if previous_velocity is None or float(np.linalg.norm(previous_velocity)) < 0.1:
        return range_score

    current_norm = float(np.linalg.norm(movement))
    previous_norm = float(np.linalg.norm(previous_velocity))
    acceleration_ratio = abs(current_norm - previous_norm) / max(1.0, previous_norm)
    continuity = max(0.0, 1.0 - acceleration_ratio / 1.5)
    return 0.45 * range_score + 0.55 * continuity


def _blur_score(blur_angle_deg: Optional[float], direction: str, low_fps: bool) -> float:
    if not low_fps:
        return 0.55
    if blur_angle_deg is None:
        return 0.35

    if direction in {"左到右", "右到左"}:
        expected = 0.0
    else:
        expected = 90.0

    delta = abs((blur_angle_deg - expected + 90.0) % 180.0 - 90.0)
    return max(0.0, 1.0 - delta / 90.0)


def reject_reason(
    x: float,
    y: float,
    previous_x: Optional[float],
    previous_y: Optional[float],
    max_step_px: float,
    min_step_px: float,
    direction_score: float,
) -> str:
    if previous_x is None or previous_y is None:
        return ""
    distance = math.hypot(x - previous_x, y - previous_y)
    if distance > max_step_px:
        return "超过最大单帧位移"
    if distance < min_step_px:
        return "低于最小单帧位移"
    if direction_score < 0.42:
        return "飞行方向不一致"
    return ""


def score_candidate(
    frame_index: int,
    x: float,
    y: float,
    radius: float,
    area: float,
    brightness: float,
    predicted_x: Optional[float],
    predicted_y: Optional[float],
    previous_x: Optional[float],
    previous_y: Optional[float],
    previous_velocity: Optional[np.ndarray],
    direction: str,
    min_step_px: float,
    max_step_px: float,
    smoothness: float,
    low_fps: bool,
    blur_angle_deg: Optional[float],
) -> ScoredCandidate:
    brightness_score = _norm(brightness, 150.0, 255.0)
    area_score = _area_score(area, low_fps=low_fps)
    prediction_score = _prediction_score(x, y, predicted_x, predicted_y, max_step_px=max_step_px)
    direction_score = _direction_score(x, y, previous_x, previous_y, direction)
    smoothness_score = _smoothness_score(
        x,
        y,
        predicted_x,
        predicted_y,
        smoothness=smoothness,
        max_step_px=max_step_px,
    )
    velocity_score = _velocity_score(
        x,
        y,
        previous_x,
        previous_y,
        previous_velocity=previous_velocity,
        min_step_px=min_step_px,
        max_step_px=max_step_px,
    )
    blur_score = _blur_score(blur_angle_deg, direction=direction, low_fps=low_fps)

    score = (
        0.20 * brightness_score
        + 0.17 * area_score
        + 0.19 * prediction_score
        + 0.16 * direction_score
        + 0.12 * smoothness_score
        + 0.11 * velocity_score
        + 0.05 * blur_score
    )
    if low_fps:
        score = (
            0.16 * brightness_score
            + 0.13 * area_score
            + 0.18 * prediction_score
            + 0.20 * direction_score
            + 0.10 * smoothness_score
            + 0.10 * velocity_score
            + 0.13 * blur_score
        )

    return ScoredCandidate(
        frame_index=frame_index,
        x=x,
        y=y,
        radius=radius,
        area=area,
        brightness=brightness,
        score=float(score),
        brightness_score=float(brightness_score),
        area_score=float(area_score),
        prediction_score=float(prediction_score),
        direction_score=float(direction_score),
        smoothness_score=float(smoothness_score),
        velocity_score=float(velocity_score),
        blur_score=float(blur_score),
        reason=reject_reason(x, y, previous_x, previous_y, max_step_px, min_step_px, direction_score),
    )
