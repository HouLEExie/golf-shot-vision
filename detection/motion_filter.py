from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from detection.ball_detector import Detection


@dataclass
class _Track:
    detections: List[Detection]
    score: float
    missed: int = 0

    @property
    def last(self) -> Detection:
        return self.detections[-1]

    def velocity(self) -> np.ndarray:
        if len(self.detections) < 2:
            return np.array([0.0, 0.0])
        current = self.detections[-1]
        previous = self.detections[-2]
        frame_delta = max(1, current.frame_index - previous.frame_index)
        return np.array([(current.x - previous.x) / frame_delta, (current.y - previous.y) / frame_delta])

    def predicted_position(self, frame_index: int) -> np.ndarray:
        gap = max(1, frame_index - self.last.frame_index)
        last_position = np.array([self.last.x, self.last.y])
        return last_position + self.velocity() * gap

    def extended(self, detection: Detection, link_score: float) -> "_Track":
        return _Track(
            detections=[*self.detections, detection],
            score=self.score + detection.score + link_score,
            missed=0,
        )


def _link_score(track: _Track, detection: Detection, max_distance: float) -> Optional[float]:
    predicted = track.predicted_position(detection.frame_index)
    current = np.array([detection.x, detection.y])
    distance = float(np.linalg.norm(current - predicted))
    if distance > max_distance:
        return None

    velocity = track.velocity()
    speed = float(np.linalg.norm(velocity))
    distance_score = 1.0 - (distance / max_distance)
    speed_score = min(1.0, speed / 14.0)
    return 0.72 * distance_score + 0.28 * speed_score


def _is_rear_view(camera_angle: str) -> bool:
    return camera_angle in {"Rear View", "后方拍摄"}


def _is_in_start_region(detection: Detection, frame_width: int, frame_height: int, start_region: str) -> bool:
    x_ratio = detection.x / max(1.0, float(frame_width))
    y_ratio = detection.y / max(1.0, float(frame_height))

    if start_region in {"全画面", "Full frame"}:
        return True
    if start_region in {"左下区域", "Lower left"}:
        return x_ratio <= 0.48 and y_ratio >= 0.42
    if start_region in {"中下区域", "Lower center"}:
        return 0.22 <= x_ratio <= 0.78 and y_ratio >= 0.42
    if start_region in {"右下区域", "Lower right"}:
        return x_ratio >= 0.52 and y_ratio >= 0.42
    return y_ratio >= 0.42


def _track_quality(
    track: _Track,
    frame_width: int,
    frame_height: int,
    camera_angle: str = "侧面拍摄",
    start_region: str = "画面下半部（推荐）",
) -> float:
    detections = track.detections
    if len(detections) < 2:
        return track.score

    start = np.array([detections[0].x, detections[0].y])
    end = np.array([detections[-1].x, detections[-1].y])
    displacement = float(np.linalg.norm(end - start))
    min_dimension = max(1, min(frame_width, frame_height))

    speeds = []
    for previous, current in zip(detections[:-1], detections[1:]):
        frame_delta = max(1, current.frame_index - previous.frame_index)
        speeds.append(np.linalg.norm([current.x - previous.x, current.y - previous.y]) / frame_delta)

    speed_mean = float(np.mean(speeds)) if speeds else 0.0
    speed_std = float(np.std(speeds)) if speeds else 0.0
    consistency = 1.0 / (1.0 + speed_std / max(1.0, speed_mean))
    displacement_score = min(1.0, displacement / (min_dimension * 0.18))
    length_score = min(1.0, len(detections) / 24.0)
    avg_score = float(np.mean([item.score for item in detections]))
    avg_radius = float(np.mean([item.radius for item in detections]))
    radius_std = float(np.std([item.radius for item in detections]))
    avg_area = float(np.mean([item.area for item in detections]))
    small_target_score = max(0.0, 1.0 - abs(avg_radius - 4.5) / 10.0)
    size_stability_score = 1.0 / (1.0 + radius_std / max(1.0, avg_radius))
    area_penalty = min(3.0, avg_area / 120.0)

    start_region_score = 1.0 if _is_in_start_region(detections[0], frame_width, frame_height, start_region) else 0.0
    vertical_travel = detections[0].y - detections[-1].y
    upward_score = 0.5
    if not _is_rear_view(camera_angle):
        upward_score = max(0.0, min(1.0, (vertical_travel / max(1.0, min_dimension * 0.10)) * 0.5 + 0.5))
    speed_score = min(1.0, speed_mean / 10.0)

    return (
        track.score
        + 8.0 * displacement_score
        + 5.5 * consistency
        + 4.0 * length_score
        + 3.0 * small_target_score
        + 2.4 * size_stability_score
        + 2.2 * speed_score
        + 2.0 * avg_score
        + 2.0 * start_region_score
        + 1.5 * upward_score
        - area_penalty
    )


def select_best_trajectory(
    candidate_frames: List[List[Detection]],
    frame_width: int,
    frame_height: int,
    min_points: int = 5,
    max_gap: int = 4,
    camera_angle: str = "侧面拍摄",
    start_region: str = "画面下半部（推荐）",
) -> List[Detection]:
    """Link per-frame detections into the most plausible moving ball trajectory."""
    min_dimension = max(1, min(frame_width, frame_height))
    base_link_distance = max(32.0, min_dimension * 0.055)
    active_tracks: List[_Track] = []
    completed_tracks: List[_Track] = []

    for frame_candidates in candidate_frames:
        if not frame_candidates:
            still_active = []
            for track in active_tracks:
                track.missed += 1
                if track.missed <= max_gap:
                    still_active.append(track)
                else:
                    completed_tracks.append(track)
            active_tracks = still_active
            continue

        frame_index = frame_candidates[0].frame_index
        next_tracks: List[_Track] = []

        for track in active_tracks:
            gap = frame_index - track.last.frame_index
            if gap > max_gap + 1:
                completed_tracks.append(track)
                continue

            max_distance = base_link_distance * max(1.0, gap * 1.25)
            linked_options = []
            for detection in frame_candidates:
                score = _link_score(track, detection, max_distance=max_distance)
                if score is not None:
                    linked_options.append((score, detection))

            linked_options.sort(key=lambda item: item[0] + item[1].score, reverse=True)
            for score, detection in linked_options[:2]:
                next_tracks.append(track.extended(detection, score))

            if not linked_options:
                track.missed += 1
                if track.missed <= max_gap:
                    next_tracks.append(track)
                else:
                    completed_tracks.append(track)

        for detection in frame_candidates[:8]:
            if _is_in_start_region(detection, frame_width, frame_height, start_region):
                next_tracks.append(_Track(detections=[detection], score=detection.score + 0.6))

        next_tracks.sort(
            key=lambda track: _track_quality(
                track,
                frame_width=frame_width,
                frame_height=frame_height,
                camera_angle=camera_angle,
                start_region=start_region,
            ),
            reverse=True,
        )
        active_tracks = next_tracks[:80]

    completed_tracks.extend(active_tracks)
    qualified = [track for track in completed_tracks if len(track.detections) >= min_points]
    if not qualified:
        return []

    qualified.sort(
        key=lambda track: _track_quality(
            track,
            frame_width=frame_width,
            frame_height=frame_height,
            camera_angle=camera_angle,
            start_region=start_region,
        ),
        reverse=True,
    )
    best = qualified[0]

    start = np.array([best.detections[0].x, best.detections[0].y])
    end = np.array([best.detections[-1].x, best.detections[-1].y])
    displacement = float(np.linalg.norm(end - start))
    if displacement < max(18.0, min_dimension * 0.018):
        return []

    if not _is_rear_view(camera_angle):
        vertical_travel = best.detections[0].y - best.detections[-1].y
        if vertical_travel < -min_dimension * 0.08:
            return []

    return best.detections
