from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class Detection:
    frame_index: int
    x: float
    y: float
    radius: float
    area: float
    score: float


class BallDetector:
    """Detect small bright moving targets that could be a golf ball."""

    def __init__(
        self,
        min_radius: float = 1.3,
        max_radius: float = 14.0,
        max_candidates_per_frame: int = 18,
    ) -> None:
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.max_candidates_per_frame = max_candidates_per_frame

    def detect_candidates(
        self,
        frame: np.ndarray,
        frame_index: int,
        previous_frame: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        percentile_threshold = np.percentile(blurred, 99.45)
        brightness_threshold = int(max(175, min(245, percentile_threshold)))
        bright_mask = cv2.inRange(blurred, brightness_threshold, 255)

        if previous_frame is None:
            return []

        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        previous_blurred = cv2.GaussianBlur(previous_gray, (5, 5), 0)
        diff = cv2.absdiff(blurred, previous_blurred)
        _, motion_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.dilate(motion_mask, np.ones((3, 3), np.uint8), iterations=1)
        candidate_mask = cv2.bitwise_and(bright_mask, motion_mask)
        if cv2.countNonZero(candidate_mask) < 2:
            return []

        candidate_mask = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_OPEN,
            np.ones((2, 2), np.uint8),
            iterations=1,
        )

        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []
        frame_area = frame.shape[0] * frame.shape[1]
        max_area = max(32.0, min(520.0, frame_area * 0.00028))

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 2.0 or area > max_area:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < self.min_radius or radius > self.max_radius:
                continue

            rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
            aspect_ratio = max(rect_w, rect_h) / max(1.0, float(min(rect_w, rect_h)))
            if aspect_ratio > 2.1:
                continue

            enclosing_area = np.pi * radius * radius
            fill_ratio = area / max(1.0, enclosing_area)
            if fill_ratio < 0.18:
                continue

            x_i = int(round(x))
            y_i = int(round(y))
            if not (0 <= x_i < gray.shape[1] and 0 <= y_i < gray.shape[0]):
                continue

            brightness_score = float(gray[y_i, x_i]) / 255.0
            size_score = 1.0 - min(1.0, abs(radius - 4.5) / 10.0)
            circularity = 0.0
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = min(1.0, (4.0 * np.pi * area) / (perimeter * perimeter))
            if circularity < 0.24:
                continue

            previous_value = float(previous_gray[y_i, x_i])
            motion_score = min(1.0, abs(float(gray[y_i, x_i]) - previous_value) / 45.0)
            if motion_score < 0.12:
                continue

            compactness_score = min(1.0, fill_ratio / 0.55)
            score = (
                (0.34 * brightness_score)
                + (0.23 * size_score)
                + (0.20 * circularity)
                + (0.15 * motion_score)
                + (0.08 * compactness_score)
            )
            detections.append(
                Detection(
                    frame_index=frame_index,
                    x=float(x),
                    y=float(y),
                    radius=float(radius),
                    area=area,
                    score=float(score),
                )
            )

        detections.sort(key=lambda item: item.score, reverse=True)
        return detections[: self.max_candidates_per_frame]
