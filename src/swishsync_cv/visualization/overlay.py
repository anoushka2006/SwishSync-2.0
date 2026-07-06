"""OpenCV drawing helpers for detections and debug overlays."""

from __future__ import annotations

import cv2
import numpy as np

from swishsync_cv.config import VideoOutputConfig
from swishsync_cv.data import DetectionRecord, HoopLock, ShotCandidate, SparseBallDetection
from swishsync_cv.visualization.shot_story_drawing import draw_pickup_preview, draw_shot_story

__all__ = ["render_debug_panel", "draw_posture_overlay", "HOOP_LOCK_COLOR"]

BALL_TRAIL_COLOR = (170, 170, 170)  # faint grey dribble/pre/post trail
BASKETBALL_COLOR = (0, 140, 255)
HOOP_CANDIDATE_COLOR = (255, 120, 80)
HOOP_LOCK_COLOR = (80, 220, 120)
COLLECTING_POINT_COLOR = (200, 200, 255)
COLLECTING_PATH_COLOR = (160, 160, 220)
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
    preview_pickup_points: list[SparseBallDetection] | None = None,
    ball_trail: list[tuple[int, float, float]] | None = None,
) -> np.ndarray:
    """Return left panel with original video and debug overlays."""

    panel = frame.copy()
    if ball_trail:
        _draw_ball_trail(panel, ball_trail)
    visible_detections = _filter_detections_for_display(detections, hoop_lock)
    _draw_detections(panel, visible_detections, draw_confidence=config.draw_confidence)
    if hoop_lock is not None:
        _draw_hoop_lock(panel, hoop_lock, hoop_phase)
    if sparse_point is not None and detection_ran:
        _draw_sparse_detection(panel, sparse_point)
    if preview_pickup_points and len(preview_pickup_points) >= 2:
        draw_pickup_preview(panel, preview_pickup_points)
    if collecting_shot is not None and collecting_shot.state == "collecting_shot":
        _draw_collection_preview(panel, collecting_shot, config)
    elif display_shot is not None and display_shot.insufficient_points_for_fit:
        _draw_insufficient_fit(panel, display_shot, config)
    elif display_shot is not None and display_shot.parabola_fit is not None:
        _draw_finalized_shot_overlay(panel, display_shot, config)
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
    rim = (int(round(hoop_lock.rim_center_x)), int(round(hoop_lock.rim_center_y)))
    cv2.circle(frame, rim, 6, color, 2)
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


def _draw_collection_preview(
    frame: np.ndarray,
    collecting_shot: ShotCandidate,
    config: VideoOutputConfig,
) -> None:
    draw_shot_story(
        frame,
        collecting_shot,
        simplified=True,
        story_config=config.shot_story,
    )
    cv2.putText(
        frame,
        "COLLECTING (no fit)",
        (16, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def _draw_finalized_shot_overlay(
    frame: np.ndarray,
    display_shot: ShotCandidate,
    config: VideoOutputConfig,
) -> None:
    draw_shot_story(
        frame,
        display_shot,
        simplified=True,
        story_config=config.shot_story,
    )


def _draw_insufficient_fit(
    frame: np.ndarray,
    display_shot: ShotCandidate,
    config: VideoOutputConfig,
) -> None:
    draw_shot_story(
        frame,
        display_shot,
        simplified=True,
        story_config=config.shot_story,
    )
    cv2.putText(
        frame,
        "INSUFFICIENT POINTS FOR FIT",
        (16, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 80, 255),
        2,
        cv2.LINE_AA,
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


def _draw_ball_trail(
    panel: np.ndarray,
    ball_trail: list[tuple[int, float, float]],
) -> None:
    """Draw the render-only ball-dot trail (dribbles + pre/post-shot), oldest faintest."""

    if not ball_trail:
        return
    ordered = sorted(ball_trail, key=lambda item: item[0])
    n = len(ordered)
    overlay = panel.copy()
    for index, (_frame_index, x, y) in enumerate(ordered):
        # newest dots brighter/bigger; skip the very newest (current ball drawn
        # separately as the live detection marker)
        if index == n - 1:
            continue
        recency = (index + 1) / n
        radius = 2 if recency < 0.6 else 3
        cv2.circle(overlay, (int(round(x)), int(round(y))), radius, BALL_TRAIL_COLOR, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, panel, 0.45, 0, panel)


_POSE_SKELETON = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
                  (23, 25), (25, 27), (24, 26), (26, 28), (11, 12), (23, 24)]


def draw_posture_overlay(panel, landmarks, metrics) -> None:
    """Draw shooter skeleton + elbow/knee/back angles on the left panel."""

    if landmarks:
        for a, b in _POSE_SKELETON:
            if a in landmarks and b in landmarks:
                pa, pb = landmarks[a], landmarks[b]
                cv2.line(panel, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                         (0, 255, 0), 2, cv2.LINE_AA)
        for x, y, *_ in landmarks.values():
            cv2.circle(panel, (int(x), int(y)), 4, (0, 200, 255), -1, cv2.LINE_AA)
    if metrics is None:
        return
    lines = [
        f"Elbow: {metrics.elbow_angle_deg:.0f} deg" if metrics.elbow_angle_deg else "Elbow: --",
        f"Knee:  {metrics.knee_bend_deg:.0f} deg" if metrics.knee_bend_deg else "Knee: --",
        f"Back:  {metrics.back_bend_deg:.0f} deg" if metrics.back_bend_deg else "Back: --",
    ]
    y0 = panel.shape[0] - 150
    for i, text in enumerate(lines):
        cv2.putText(panel, text, (20, y0 + i * 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (40, 220, 40), 2, cv2.LINE_AA)
