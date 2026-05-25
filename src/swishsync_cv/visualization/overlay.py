"""OpenCV drawing helpers for detections and debug overlays."""

from __future__ import annotations

import cv2
import numpy as np

from swishsync_cv.config import VideoOutputConfig
from swishsync_cv.data import DetectionRecord, HoopLock, ShotCandidate, SparseBallDetection

BASKETBALL_COLOR = (0, 140, 255)
HOOP_CANDIDATE_COLOR = (255, 120, 80)
HOOP_LOCK_COLOR = (80, 220, 120)
COLLECTING_POINT_COLOR = (200, 200, 255)
COLLECTING_PATH_COLOR = (160, 160, 220)
FINAL_ARC_COLOR = (80, 220, 255)
TEXT_COLOR = (255, 255, 255)


def render_debug_panel(
    frame: np.ndarray,
    detections: list[DetectionRecord],
    hoop_lock: HoopLock | None,
    sparse_point: SparseBallDetection | None,
    config: VideoOutputConfig,
    frame_index: int,
    detection_ran: bool,
    hoop_phase: str = "acquisition",
    collecting_shot: ShotCandidate | None = None,
    lifecycle_state: str = "idle",
    candidate_point_count: int = 0,
    display_shot: ShotCandidate | None = None,
) -> np.ndarray:
    """Return left panel with original video and debug overlays."""

    panel = frame.copy()
    visible_detections = _filter_detections_for_display(detections, hoop_lock)
    _draw_detections(panel, visible_detections, draw_confidence=config.draw_confidence)
    if hoop_lock is not None:
        _draw_hoop_lock(panel, hoop_lock, hoop_phase)
    if sparse_point is not None and detection_ran:
        _draw_sparse_detection(panel, sparse_point)
    if collecting_shot is not None and collecting_shot.state == "collecting_shot":
        _draw_collection_preview(panel, collecting_shot)
    elif display_shot is not None and display_shot.parabola_fit is not None:
        _draw_finalized_arc(panel, display_shot)
    _draw_debug_hud(
        panel,
        frame_index=frame_index,
        detection_ran=detection_ran,
        hoop_lock=hoop_lock,
        hoop_phase=hoop_phase,
        draw_frame_index=config.draw_frame_index,
        lifecycle_state=lifecycle_state,
        candidate_point_count=candidate_point_count,
    )
    return panel


def _filter_detections_for_display(
    detections: list[DetectionRecord],
    hoop_lock: HoopLock | None,
) -> list[DetectionRecord]:
    if hoop_lock is not None and hoop_lock.is_locked:
        return [detection for detection in detections if detection.label != "hoop"]
    return detections


def _draw_detections(
    frame: np.ndarray,
    detections: list[DetectionRecord],
    draw_confidence: bool,
) -> None:
    for detection in detections:
        color = BASKETBALL_COLOR if detection.label == "basketball" else HOOP_CANDIDATE_COLOR
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


def _draw_hoop_lock(
    frame: np.ndarray,
    hoop_lock: HoopLock,
    hoop_phase: str,
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in hoop_lock.bbox_xyxy]
    color = HOOP_LOCK_COLOR if hoop_lock.is_locked else HOOP_CANDIDATE_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    center = (int(round(hoop_lock.center_x)), int(round(hoop_lock.center_y)))
    cv2.drawMarker(
        frame,
        center,
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=16,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    status = "HOOP LOCKED" if hoop_lock.is_locked else hoop_phase.upper()
    cv2.putText(
        frame,
        status,
        (x1, max(16, y1 - 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    confidence_pct = int(round(hoop_lock.confidence * 100))
    cv2.putText(
        frame,
        f"conf {confidence_pct}%",
        (x1, max(16, y1 - 44)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_collection_preview(frame: np.ndarray, collecting_shot: ShotCandidate) -> None:
    points = [
        (int(round(point.x)), int(round(point.y)))
        for point in collecting_shot.candidate_points
    ]
    for point in points:
        cv2.circle(frame, point, 4, COLLECTING_POINT_COLOR, -1)
    if len(points) >= 2:
        for start, end in zip(points, points[1:]):
            _draw_dotted_line(frame, start, end, COLLECTING_PATH_COLOR)


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


def _draw_finalized_arc(frame: np.ndarray, display_shot: ShotCandidate) -> None:
    fit = display_shot.parabola_fit
    if fit is None:
        return
    arc_points = fit.sample_arc(num_points=80)
    pixel_points = [(int(round(x)), int(round(y))) for x, y in arc_points]
    if len(pixel_points) >= 2:
        cv2.polylines(
            frame,
            [np.asarray(pixel_points, dtype=np.int32)],
            isClosed=False,
            color=FINAL_ARC_COLOR,
            thickness=3,
            lineType=cv2.LINE_AA,
        )


def _draw_sparse_detection(frame: np.ndarray, point: SparseBallDetection) -> None:
    center = (int(round(point.x)), int(round(point.y)))
    cv2.circle(frame, center, 5, COLLECTING_POINT_COLOR, 2)
    cv2.putText(
        frame,
        f"sparse {point.confidence:.2f}",
        (center[0] + 8, center[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLLECTING_POINT_COLOR,
        1,
        cv2.LINE_AA,
    )


def _draw_debug_hud(
    frame: np.ndarray,
    frame_index: int,
    detection_ran: bool,
    hoop_lock: HoopLock | None,
    hoop_phase: str,
    draw_frame_index: bool,
    lifecycle_state: str,
    candidate_point_count: int,
) -> None:
    lines = []
    if draw_frame_index:
        lines.append(f"frame={frame_index}")
    lines.append(f"detect={'yes' if detection_ran else 'skip'}")
    lines.append(f"shot state={lifecycle_state.upper()}")
    lines.append(f"candidate points={candidate_point_count}")
    lines.append(f"HOOP LOCKED: {'TRUE' if hoop_lock and hoop_lock.is_locked else 'FALSE'}")
    if hoop_lock is not None:
        lines.append(f"HOOP CONFIDENCE: {int(round(hoop_lock.confidence * 100))}%")
    else:
        lines.append(f"hoop phase: {hoop_phase}")

    y_offset = 24
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
        y_offset += 22
