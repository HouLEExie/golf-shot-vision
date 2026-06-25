from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tracking.trajectory_cleaner import TrajectoryPoint
from video.loader import VideoLoadError


def _points_until_frame(trajectory: List[TrajectoryPoint], frame_index: int) -> List[Tuple[int, int]]:
    return [
        (int(round(point.x)), int(round(point.y)))
        for point in trajectory
        if point.frame_index <= frame_index
    ]


def _draw_polyline(
    frame: np.ndarray,
    points: List[Tuple[int, int]],
    color_bgr: Tuple[int, int, int],
    thickness: int,
    glow: bool,
) -> None:
    if len(points) < 2:
        return

    point_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    if glow:
        for multiplier, alpha in [(5, 0.18), (3, 0.26)]:
            overlay = frame.copy()
            cv2.polylines(
                overlay,
                [point_array],
                isClosed=False,
                color=color_bgr,
                thickness=max(thickness * multiplier, thickness + 4),
                lineType=cv2.LINE_AA,
            )
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)

    cv2.polylines(
        frame,
        [point_array],
        isClosed=False,
        color=color_bgr,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def _draw_points(
    frame: np.ndarray,
    points: List[Tuple[int, int]],
    color_bgr: Tuple[int, int, int],
    show_points: bool,
    show_markers: bool,
) -> None:
    if show_points:
        for point in points:
            cv2.circle(frame, point, 3, color_bgr, -1, lineType=cv2.LINE_AA)

    if show_markers and points:
        cv2.circle(frame, points[0], 8, (57, 255, 136), 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, points[-1], 8, (63, 210, 255), 2, lineType=cv2.LINE_AA)


def render_traced_video(
    input_path: Path,
    output_path: Path,
    trajectory: List[TrajectoryPoint],
    color_bgr: Tuple[int, int, int],
    thickness: int = 4,
    glow: bool = True,
    show_points: bool = True,
    show_markers: bool = True,
    max_frames: Optional[int] = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise VideoLoadError("无法重新打开输入视频用于渲染。")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise VideoLoadError("视频尺寸无效。")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise VideoLoadError("无法创建轨迹输出视频。")

    frame_index = 0
    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            current_points = _points_until_frame(trajectory, frame_index)
            _draw_polyline(frame, current_points, color_bgr=color_bgr, thickness=thickness, glow=glow)
            _draw_points(
                frame,
                current_points,
                color_bgr=color_bgr,
                show_points=show_points,
                show_markers=show_markers,
            )
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    return output_path
