from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from detection.ball_detector import Detection
from tracking.kalman_tracker import KalmanTracker


@dataclass
class TrajectoryPoint:
    frame_index: int
    x: float
    y: float
    observed_x: float
    observed_y: float
    score: float


def _moving_average(values: List[float], radius: int = 2) -> List[float]:
    if len(values) <= 2:
        return values
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed.append(float(np.mean(values[start:end])))
    return smoothed


def smooth_trajectory(detections: List[Detection]) -> List[TrajectoryPoint]:
    if not detections:
        return []

    ordered = sorted(detections, key=lambda item: item.frame_index)
    tracker = KalmanTracker()
    first = ordered[0]
    tracker.initialize(first.x, first.y)

    points = [
        TrajectoryPoint(
            frame_index=first.frame_index,
            x=first.x,
            y=first.y,
            observed_x=first.x,
            observed_y=first.y,
            score=first.score,
        )
    ]
    previous_frame = first.frame_index

    for detection in ordered[1:]:
        gap = max(1, detection.frame_index - previous_frame)
        tracker.predict(dt=float(gap))
        estimated_x, estimated_y = tracker.update(detection.x, detection.y)
        points.append(
            TrajectoryPoint(
                frame_index=detection.frame_index,
                x=estimated_x,
                y=estimated_y,
                observed_x=detection.x,
                observed_y=detection.y,
                score=detection.score,
            )
        )
        previous_frame = detection.frame_index

    averaged_x = _moving_average([point.x for point in points])
    averaged_y = _moving_average([point.y for point in points])
    for point, x_value, y_value in zip(points, averaged_x, averaged_y):
        point.x = x_value
        point.y = y_value
    return points


def compute_confidence(
    points: List[TrajectoryPoint],
    detections: List[Detection],
    camera_angle: str,
    processed_frames: int,
) -> float:
    if len(points) < 2:
        return 0.0

    positions = np.array([[point.x, point.y] for point in points], dtype=float)
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    segment_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    mean_speed = float(np.mean(segment_distances)) if len(segment_distances) else 0.0
    speed_std = float(np.std(segment_distances)) if len(segment_distances) else 0.0

    length_score = min(1.0, len(points) / 24.0)
    detection_score = float(np.mean([item.score for item in detections])) if detections else 0.0
    avg_radius = float(np.mean([item.radius for item in detections])) if detections else 12.0
    radius_std = float(np.std([item.radius for item in detections])) if detections else 4.0
    avg_area = float(np.mean([item.area for item in detections])) if detections else 200.0
    consistency_score = 1.0 / (1.0 + speed_std / max(1.0, mean_speed))
    displacement_score = min(1.0, displacement / 240.0)
    duration_score = min(1.0, (points[-1].frame_index - points[0].frame_index + 1) / max(1, processed_frames * 0.12))
    small_target_score = max(0.0, 1.0 - abs(avg_radius - 4.5) / 10.0)
    size_stability_score = 1.0 / (1.0 + radius_std / max(1.0, avg_radius))
    area_score = max(0.0, 1.0 - min(1.0, avg_area / 260.0))
    view_factor = 0.82 if camera_angle in {"Rear View", "后方拍摄"} else 1.0

    confidence = (
        0.20 * length_score
        + 0.20 * detection_score
        + 0.18 * consistency_score
        + 0.14 * displacement_score
        + 0.08 * duration_score
        + 0.09 * small_target_score
        + 0.07 * size_stability_score
        + 0.04 * area_score
    )
    return round(max(0.0, min(0.98, confidence * view_factor)) * 100.0, 1)
