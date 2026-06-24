from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from tracking.trajectory_cleaner import TrajectoryPoint


def estimate_launch_angle(points: List[TrajectoryPoint], camera_angle: str = "侧面拍摄") -> Optional[float]:
    if len(points) < 3:
        return None

    sample = points[: min(len(points), 10)]
    x_values = np.array([point.x - sample[0].x for point in sample], dtype=float)
    y_up_values = np.array([sample[0].y - point.y for point in sample], dtype=float)

    horizontal_span = float(np.max(x_values) - np.min(x_values))
    vertical_span = float(np.max(y_up_values) - np.min(y_up_values))
    if horizontal_span < 2.0 and vertical_span < 2.0:
        return None

    if horizontal_span < 2.0:
        angle = 80.0 if vertical_span > 0 else 0.0
    else:
        slope, _ = np.polyfit(x_values, y_up_values, 1)
        angle = math.degrees(math.atan(abs(float(slope))))

    if camera_angle in {"Rear View", "后方拍摄"}:
        angle *= 0.82

    return round(max(-5.0, min(60.0, angle)), 1)
