from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2


class VideoLoadError(RuntimeError):
    """Raised when OpenCV cannot open or read a video file."""


@dataclass
class VideoMetadata:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float


def ensure_output_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return safe or "uploaded_video.mp4"


def save_uploaded_file(uploaded_file, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    original_name = sanitize_filename(uploaded_file.name)
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix.lower() or ".mp4"
    output_path = upload_dir / f"{stem}_{int(time.time())}{suffix}"
    output_path.write_bytes(uploaded_file.getbuffer())
    return output_path


def get_video_metadata(video_path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoLoadError("无法打开该视频文件。请尝试转换为 mp4 后重新上传。")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()

    if fps <= 1:
        fps = 30.0
    duration = frame_count / fps if frame_count > 0 else 0.0
    return VideoMetadata(
        path=str(video_path),
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_seconds=duration,
    )


def iter_video_frames(
    video_path: Path,
    max_frames: Optional[int] = None,
) -> Iterable[Tuple[int, object]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoLoadError("无法打开该视频文件。请尝试更换视频。")

    frame_index = 0
    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            yield frame_index, frame
            frame_index += 1
    finally:
        capture.release()
