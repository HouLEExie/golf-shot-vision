from __future__ import annotations

import math
from typing import Optional


CLUB_LAUNCH_FALLBACK = {
    "Driver": 12.0,
    "一号木": 12.0,
    "Wood": 14.0,
    "球道木": 14.0,
    "Hybrid": 17.0,
    "混合杆": 17.0,
    "Iron": 21.0,
    "铁杆": 21.0,
    "Wedge": 31.0,
    "挖起杆": 31.0,
}

CLUB_DRAG_FACTOR = {
    "Driver": 0.62,
    "一号木": 0.62,
    "Wood": 0.60,
    "球道木": 0.60,
    "Hybrid": 0.58,
    "混合杆": 0.58,
    "Iron": 0.55,
    "铁杆": 0.55,
    "Wedge": 0.46,
    "挖起杆": 0.46,
}


def estimate_carry(
    speed_mps: Optional[float],
    launch_angle_deg: Optional[float],
    club_type: str = "一号木",
    camera_angle: str = "侧面拍摄",
) -> Optional[float]:
    if speed_mps is None or speed_mps <= 0:
        return None

    angle = launch_angle_deg
    if angle is None or angle <= 0:
        angle = CLUB_LAUNCH_FALLBACK.get(club_type, 14.0)

    angle = max(3.0, min(45.0, float(angle)))
    radians = math.radians(angle)
    ideal_range = (speed_mps * speed_mps * math.sin(2.0 * radians)) / 9.80665
    drag_factor = CLUB_DRAG_FACTOR.get(club_type, 0.56)
    if camera_angle in {"Rear View", "后方拍摄"}:
        drag_factor *= 0.82

    carry = ideal_range * drag_factor
    return round(max(0.0, carry), 1)
