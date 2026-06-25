from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


def save_debug_overlay_image(
    video_path: Path,
    output_path: Path,
    debug_rows: List[Dict[str, object]],
    roi: Optional[Tuple[int, int, int, int]],
    start_frame: int,
) -> Optional[Path]:
    if not debug_rows:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None

    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 255), 2)

    accepted_points = []
    rejected_points = []
    for row in debug_rows:
        point = (int(float(row["x"])), int(float(row["y"])))
        if bool(row.get("accepted")):
            accepted_points.append(point)
        else:
            rejected_points.append(point)

    for point in rejected_points[:160]:
        cv2.circle(frame, point, 4, (60, 60, 255), 1, lineType=cv2.LINE_AA)
    for point in accepted_points:
        cv2.circle(frame, point, 6, (57, 255, 136), 2, lineType=cv2.LINE_AA)

    if len(accepted_points) >= 2:
        for previous, current in zip(accepted_points[:-1], accepted_points[1:]):
            cv2.line(frame, previous, current, (57, 255, 136), 2, lineType=cv2.LINE_AA)

    cv2.putText(
        frame,
        "green=selected red=rejected yellow=ROI",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), frame)
    return output_path
