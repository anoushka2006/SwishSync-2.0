"""OpenCV drawing helpers for detections and ball trajectory overlays."""

from __future__ import annotations

import cv2
import numpy as np

from swishsync_cv.config import VideoOutputConfig
from swishsync_cv.data import DetectionRecord, TrajectoryPoint

BASKETBALL_COLOR = (0, 140, 255)
HOOP_COLOR = (255, 80, 80)
TRAJECTORY_COLOR = (0, 255, 255)
TEXT_COLOR = (255, 255, 255)


def annotate_frame(
    frame: np.ndarray,
    detections: list[DetectionRecord],
    trajectory: list[TrajectoryPoint],
    config: VideoOutputConfig,
    frame_index: int | None = None,
) -> np.ndarray:
    """Return a copy of ``frame`` with boxes, labels, and trajectory drawn."""

    annotated = frame.copy()
    _draw_detections(annotated, detections, draw_confidence=config.draw_confidence)
    _draw_trajectory(
        annotated,
        trajectory,
        max_points=config.trajectory_max_points,
    )
    if config.draw_frame_index and frame_index is not None:
        cv2.putText(
            annotated,
            f"frame={frame_index}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
    return annotated


def _draw_detections(
    frame: np.ndarray,
    detections: list[DetectionRecord],
    draw_confidence: bool,
) -> None:
    for detection in detections:
        color = BASKETBALL_COLOR if detection.label == "basketball" else HOOP_COLOR
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox_xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = detection.label
        if draw_confidence:
            label = f"{label} {detection.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(16, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_trajectory(
    frame: np.ndarray,
    trajectory: list[TrajectoryPoint],
    max_points: int | None,
) -> None:
    points = trajectory[-max_points:] if max_points is not None else trajectory
    if not points:
        return

    pixel_points = [(int(round(point.x)), int(round(point.y))) for point in points]
    for point in pixel_points:
        cv2.circle(frame, point, 3, TRAJECTORY_COLOR, -1)

    if len(pixel_points) >= 2:
        cv2.polylines(
            frame,
            [np.asarray(pixel_points, dtype=np.int32)],
            isClosed=False,
            color=TRAJECTORY_COLOR,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
