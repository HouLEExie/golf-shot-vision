from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from detection.ball_detector import Detection
from detection.candidate_scoring import ScoredCandidate, score_candidate
from video.loader import VideoLoadError


@dataclass
class TrackerConfig:
    recognition_mode: str
    video_quality_mode: str
    fps: int
    start_frame: int
    analyze_frames: int
    start_x: Optional[float]
    start_y: Optional[float]
    flight_direction: str
    roi: Optional[Tuple[int, int, int, int]]
    max_step_px: float
    min_step_px: float
    smoothness: float
    debug_mode: bool = False


@dataclass
class ROITrackingResult:
    detections: List[Detection]
    debug_rows: List[Dict[str, object]]
    candidate_counts: List[Dict[str, int]]
    warnings: List[str]
    stopped_reason: str
    roi_used: Optional[Tuple[int, int, int, int]]


def clamp_roi(
    roi: Optional[Tuple[int, int, int, int]],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    if roi is None:
        return None
    x1, y1, x2, y2 = roi
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _window_around(
    center_x: float,
    center_y: float,
    radius: float,
    bounds: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bounds
    return (
        max(x1, int(round(center_x - radius))),
        max(y1, int(round(center_y - radius))),
        min(x2, int(round(center_x + radius))),
        min(y2, int(round(center_y + radius))),
    )


def _contour_angle(contour: np.ndarray) -> Optional[float]:
    if len(contour) < 5:
        return None
    rect = cv2.minAreaRect(contour)
    (_, _), (width, height), angle = rect
    if width <= 0 or height <= 0:
        return None
    if width < height:
        angle += 90.0
    return float(angle % 180.0)


def _candidate_mask(frame: np.ndarray, previous_frame: Optional[np.ndarray], low_fps: bool) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = int(max(145 if low_fps else 165, min(246, np.percentile(blurred, 98.7))))
    bright_mask = cv2.inRange(blurred, threshold, 255)

    if previous_frame is None:
        return bright_mask

    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    previous_blurred = cv2.GaussianBlur(previous_gray, (5, 5), 0)
    diff = cv2.absdiff(blurred, previous_blurred)
    _, motion_mask = cv2.threshold(diff, 8 if low_fps else 12, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.dilate(motion_mask, np.ones((3, 3), np.uint8), iterations=1)

    if low_fps:
        return cv2.bitwise_and(cv2.bitwise_or(bright_mask, motion_mask), cv2.dilate(bright_mask, np.ones((5, 5), np.uint8)))
    return cv2.bitwise_and(bright_mask, motion_mask)


class ROITracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self.low_fps = config.fps < 60

    def _extract_candidates(
        self,
        frame: np.ndarray,
        previous_frame: Optional[np.ndarray],
        frame_index: int,
        search_window: Tuple[int, int, int, int],
        predicted: Tuple[Optional[float], Optional[float]],
        previous_position: Tuple[Optional[float], Optional[float]],
        previous_velocity: Optional[np.ndarray],
    ) -> List[ScoredCandidate]:
        x1, y1, x2, y2 = search_window
        if x2 <= x1 or y2 <= y1:
            return []

        crop = frame[y1:y2, x1:x2]
        previous_crop = previous_frame[y1:y2, x1:x2] if previous_frame is not None else None
        mask = _candidate_mask(crop, previous_crop, low_fps=self.low_fps)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        if self.low_fps:
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        candidates: List[ScoredCandidate] = []
        min_area = 2.0 if not self.low_fps else 2.0
        max_area = 460.0 if not self.low_fps else 1500.0

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue

            rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
            aspect = max(rect_w, rect_h) / max(1.0, float(min(rect_w, rect_h)))
            if aspect > (8.0 if self.low_fps else 2.7):
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            local_x = float(moments["m10"] / moments["m00"])
            local_y = float(moments["m01"] / moments["m00"])
            radius = float(max(rect_w, rect_h) / 2.0)
            global_x = local_x + x1
            global_y = local_y + y1

            mask_for_contour = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask_for_contour, [contour], -1, 255, thickness=-1)
            brightness = float(cv2.mean(gray, mask=mask_for_contour)[0])
            if brightness < (130 if self.low_fps else 155):
                continue

            scored = score_candidate(
                frame_index=frame_index,
                x=global_x,
                y=global_y,
                radius=radius,
                area=area,
                brightness=brightness,
                predicted_x=predicted[0],
                predicted_y=predicted[1],
                previous_x=previous_position[0],
                previous_y=previous_position[1],
                previous_velocity=previous_velocity,
                direction=self.config.flight_direction,
                min_step_px=self.config.min_step_px,
                max_step_px=self.config.max_step_px,
                smoothness=self.config.smoothness,
                low_fps=self.low_fps,
                blur_angle_deg=_contour_angle(contour),
            )
            if scored.reason:
                scored.accepted = False
            candidates.append(scored)

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def track(self, video_path: Path) -> ROITrackingResult:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise VideoLoadError("无法打开该视频文件。请尝试更换视频。")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            capture.release()
            raise VideoLoadError("无法读取视频尺寸。请更换视频或重新导出。")

        roi = clamp_roi(self.config.roi, width=width, height=height)
        bounds = roi or (0, 0, width, height)
        warnings: List[str] = []
        if roi is None:
            warnings.append("建议设置搜索区域 ROI，避免程序误识别球杆反光、地面白点或背景亮点。")
        if self.low_fps:
            warnings.append("30fps 视频仅适合粗略参考，不建议用于准确距离估算。")

        start_frame = max(0, int(self.config.start_frame))
        end_frame = start_frame + max(1, int(self.config.analyze_frames))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - 1))
        previous_frame = None
        if start_frame > 0:
            ok, previous_frame = capture.read()
            if not ok:
                previous_frame = None

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        detections: List[Detection] = []
        debug_rows: List[Dict[str, object]] = []
        candidate_counts: List[Dict[str, int]] = []
        missed = 0
        max_missed = 5 if self.low_fps else 4
        previous_velocity: Optional[np.ndarray] = None
        predicted_x = self.config.start_x
        predicted_y = self.config.start_y
        stopped_reason = "分析帧数结束"

        for frame_index in range(start_frame, end_frame):
            ok, frame = capture.read()
            if not ok:
                stopped_reason = "视频读取结束"
                break

            previous_position = (
                detections[-1].x if detections else self.config.start_x,
                detections[-1].y if detections else self.config.start_y,
            )

            if detections:
                if previous_velocity is not None:
                    predicted_x = detections[-1].x + previous_velocity[0] * max(1, missed + 1)
                    predicted_y = detections[-1].y + previous_velocity[1] * max(1, missed + 1)
                else:
                    predicted_x = detections[-1].x
                    predicted_y = detections[-1].y

            if predicted_x is not None and predicted_y is not None:
                radius = self.config.max_step_px * (1.5 + 0.45 * missed)
                search_window = _window_around(predicted_x, predicted_y, radius=radius, bounds=bounds)
            else:
                search_window = bounds

            candidates = self._extract_candidates(
                frame=frame,
                previous_frame=previous_frame,
                frame_index=frame_index,
                search_window=search_window,
                predicted=(predicted_x, predicted_y),
                previous_position=previous_position,
                previous_velocity=previous_velocity,
            )

            accepted: Optional[ScoredCandidate] = None
            for candidate in candidates:
                if candidate.reason:
                    continue
                if candidate.score >= (0.34 if self.low_fps else 0.42):
                    accepted = candidate
                    break

            if accepted is not None:
                accepted.accepted = True
                accepted.reason = "已选择"
                new_detection = Detection(
                    frame_index=accepted.frame_index,
                    x=accepted.x,
                    y=accepted.y,
                    radius=accepted.radius,
                    area=accepted.area,
                    score=accepted.score,
                )
                if detections:
                    frame_delta = max(1, new_detection.frame_index - detections[-1].frame_index)
                    current_velocity = np.array(
                        [
                            (new_detection.x - detections[-1].x) / frame_delta,
                            (new_detection.y - detections[-1].y) / frame_delta,
                        ],
                        dtype=float,
                    )
                    if previous_velocity is None:
                        previous_velocity = current_velocity
                    else:
                        alpha = max(0.05, min(0.85, 1.0 - self.config.smoothness))
                        previous_velocity = alpha * current_velocity + (1.0 - alpha) * previous_velocity
                detections.append(new_detection)
                missed = 0
            else:
                missed += 1
                if missed > max_missed and detections:
                    stopped_reason = "连续丢失目标过多，停止追踪"
                    previous_frame = frame
                    break

            candidate_counts.append(
                {
                    "frame_index": frame_index,
                    "candidate_count": len(candidates),
                    "accepted_count": 1 if accepted is not None else 0,
                }
            )
            if self.config.debug_mode:
                for candidate in candidates:
                    debug_rows.append(
                        {
                            "frame_index": candidate.frame_index,
                            "x": round(candidate.x, 2),
                            "y": round(candidate.y, 2),
                            "score": round(candidate.score, 4),
                            "brightness_score": round(candidate.brightness_score, 4),
                            "area_score": round(candidate.area_score, 4),
                            "prediction_score": round(candidate.prediction_score, 4),
                            "direction_score": round(candidate.direction_score, 4),
                            "smoothness_score": round(candidate.smoothness_score, 4),
                            "velocity_score": round(candidate.velocity_score, 4),
                            "blur_score": round(candidate.blur_score, 4),
                            "accepted": bool(candidate.accepted),
                            "reason": candidate.reason or "候选未入选",
                        }
                    )

            previous_frame = frame

        capture.release()
        return ROITrackingResult(
            detections=detections,
            debug_rows=debug_rows,
            candidate_counts=candidate_counts,
            warnings=warnings,
            stopped_reason=stopped_reason,
            roi_used=roi,
        )
