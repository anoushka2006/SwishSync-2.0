"""Draw measured + gap-predicted continuity tracks."""

from __future__ import annotations

import cv2
import numpy as np

from swishsync_cv.data import ShotCandidate, SparseBallDetection, effective_point_source
from swishsync_cv.tracking.gap_recovery import continuity_track

COLLECTING_POINT_COLOR = (200, 200, 255)
COLLECTING_PATH_COLOR = (160, 160, 220)
GAP_PREDICTED_COLOR = (60, 180, 255)
DEFAULT_REACQUISITION_GAP_FRAMES = 15


def draw_continuity_track(
    frame: np.ndarray,
    shot: ShotCandidate,
    *,
    measured_color: tuple[int, int, int] = COLLECTING_POINT_COLOR,
    path_color: tuple[int, int, int] = COLLECTING_PATH_COLOR,
    gap_color: tuple[int, int, int] = GAP_PREDICTED_COLOR,
) -> None:
    points = continuity_track(shot)
    if not points:
        return

    for point in points:
        center = (int(round(point.x)), int(round(point.y)))
        if effective_point_source(point) == "gap_predicted":
            cv2.circle(frame, center, 4, gap_color, 2)
        else:
            cv2.circle(frame, center, 4, measured_color, -1)

    if len(points) < 2:
        return

    for start_point, end_point in zip(points, points[1:]):
        start = (int(round(start_point.x)), int(round(start_point.y)))
        end = (int(round(end_point.x)), int(round(end_point.y)))
        segment_color = path_color
        if _segment_uses_gap_prediction(start_point, end_point):
            _draw_dotted_line(frame, start, end, gap_color)
        elif _measured_segment_gap_ok(start_point, end_point):
            _draw_dotted_line(frame, start, end, segment_color)


def _segment_uses_gap_prediction(
    start_point: SparseBallDetection,
    end_point: SparseBallDetection,
) -> bool:
    return (
        effective_point_source(start_point) == "gap_predicted"
        or effective_point_source(end_point) == "gap_predicted"
    )


def _measured_segment_gap_ok(
    start_point: SparseBallDetection,
    end_point: SparseBallDetection,
    max_gap_frames: int = DEFAULT_REACQUISITION_GAP_FRAMES,
) -> bool:
    if _segment_uses_gap_prediction(start_point, end_point):
        return True
    return end_point.frame_index - start_point.frame_index <= max_gap_frames


def _draw_dotted_line(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    if length <= 0:
        return
    steps = max(length // 6, 1)
    for step in range(0, steps, 2):
        t0 = step / steps
        t1 = min((step + 1) / steps, 1.0)
        p0 = (int(x1 + (x2 - x1) * t0), int(y1 + (y2 - y1) * t0))
        p1 = (int(x1 + (x2 - x1) * t1), int(y1 + (y2 - y1) * t1))
        cv2.line(frame, p0, p1, color, 1, cv2.LINE_AA)
