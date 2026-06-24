from __future__ import annotations

from typing import List, Optional

import numpy as np

from tracking.trajectory_cleaner import TrajectoryPoint


def estimate_initial_ball_speed(
    points: List[TrajectoryPoint],
    fps: float,
    pixel_to_meter: float,
    window_points: int = 8,
) -> Optional[float]:
    if len(points) < 2 or fps <= 0 or pixel_to_meter <= 0:
        return None

    sample = points[: min(len(points), window_points)]
    velocities = []
    for previous, current in zip(sample[:-1], sample[1:]):
        frame_delta = max(1, current.frame_index - previous.frame_index)
        pixel_distance = float(np.linalg.norm([current.x - previous.x, current.y - previous.y]))
        meters_per_second = pixel_distance * pixel_to_meter * fps / frame_delta
        if meters_per_second > 0:
            velocities.append(meters_per_second)

    if not velocities:
        return None

    # Median dampens frame-to-frame detection noise better than a raw maximum.
    return round(float(np.median(velocities)), 2)
