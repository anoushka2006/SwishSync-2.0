"""Phased shot lifecycle visualization."""

from __future__ import annotations

import cv2
import numpy as np

from swishsync_cv.config import ShotStoryConfig
from swishsync_cv.data import (
    ShotCandidate,
    ShotStoryMetadata,
    SparseBallDetection,
    effective_point_source,
)
from swishsync_cv.tracking.gap_recovery import continuity_track
from swishsync_cv.tracking.parabola import confidence_tier
from swishsync_cv.visualization.arc_drawing import draw_finalized_arc

PICKUP_COLOR = (140, 140, 160)
POST_SHOT_COLOR = (100, 160, 120)
GAP_PREDICTED_COLOR = (60, 180, 255)
RELEASE_COLOR = (255, 220, 120)
MEASURED_COLOR = (200, 200, 255)
HIGH_CONF_COLOR = (80, 220, 120)
MEDIUM_CONF_COLOR = (80, 200, 255)
LOW_CONF_COLOR = (100, 100, 255)


def draw_shot_story(
    frame: np.ndarray,
    shot: ShotCandidate,
    *,
    opacity: float = 1.0,
    arc_thickness: int = 3,
    show_legend: bool = False,
    show_flight_dots: bool = False,
    simplified: bool = False,
    story_config: ShotStoryConfig | None = None,
) -> None:
    """Draw pickup → release → arc → gap → post layers."""

    story_cfg = story_config or ShotStoryConfig()
    story = shot.story
    collecting = shot.state == "collecting_shot"

    if collecting:
        _draw_pickup_path(frame, shot.pickup_points, opacity=opacity)
        _draw_pre_release_continuity(frame, shot, opacity=opacity)
        if shot.candidate_points:
            _draw_release_marker(
                frame,
                shot.candidate_points[0].x,
                shot.candidate_points[0].y,
                opacity=opacity,
            )
        return

    if story is None:
        return

    if story.show_pickup:
        pickup = [
            point
            for point in shot.pickup_points
            if point.frame_index in story.pickup_frames
        ]
        _draw_pickup_path(frame, pickup, opacity=opacity)

    if story.show_post_shot and shot.post_shot_points:
        _draw_post_shot_path(frame, shot.post_shot_points, opacity=opacity)

    if shot.parabola_fit is not None and not shot.insufficient_points_for_fit:
        _draw_gap_bridges(frame, shot, story, opacity=opacity)
        _draw_arc_with_opacity(frame, shot, opacity=opacity, thickness=arc_thickness)
    else:
        _draw_measured_continuity_path(frame, shot, opacity=opacity)

    _draw_release_marker(frame, story.release_xy[0], story.release_xy[1], opacity=opacity)

    if show_flight_dots and not simplified:
        _draw_flight_diagnostic_dots(
            frame,
            shot,
            story,
            opacity=opacity,
            max_dots=story_cfg.max_flight_diagnostic_dots,
        )

    if show_legend and opacity >= 0.9:
        _draw_legend(frame)


def _draw_pickup_path(
    frame: np.ndarray,
    points: list[SparseBallDetection],
    *,
    opacity: float,
) -> None:
    if len(points) < 2:
        return
    ordered = sorted(points, key=lambda point: point.frame_index)
    color = _scale_color(PICKUP_COLOR, opacity)
    for point in ordered:
        center = (int(round(point.x)), int(round(point.y)))
        cv2.circle(frame, center, 3, color, -1)
    for start, end in zip(ordered, ordered[1:]):
        _draw_dotted_line(
            frame,
            (int(round(start.x)), int(round(start.y))),
            (int(round(end.x)), int(round(end.y))),
            color,
        )


def _draw_pre_release_continuity(
    frame: np.ndarray,
    shot: ShotCandidate,
    *,
    opacity: float,
) -> None:
    if not shot.candidate_points:
        return
    release_frame = shot.candidate_points[0].frame_index
    points = [
        point
        for point in continuity_track(shot)
        if point.frame_index >= release_frame
        or effective_point_source(point) == "gap_predicted"
    ]
    if not points:
        return

    for point in points:
        center = (int(round(point.x)), int(round(point.y)))
        if effective_point_source(point) == "gap_predicted":
            cv2.circle(frame, center, 4, _scale_color(GAP_PREDICTED_COLOR, opacity), 2)
        else:
            cv2.circle(frame, center, 4, _scale_color(MEASURED_COLOR, opacity), -1)

    if len(points) < 2:
        return

    for start, end in zip(points, points[1:]):
        p0 = (int(round(start.x)), int(round(start.y)))
        p1 = (int(round(end.x)), int(round(end.y)))
        if _uses_gap_prediction(start, end):
            _draw_dotted_line(frame, p0, p1, _scale_color(GAP_PREDICTED_COLOR, opacity))
        else:
            _draw_dotted_line(frame, p0, p1, _scale_color(MEASURED_COLOR, opacity))


def _draw_post_shot_path(
    frame: np.ndarray,
    points: list[SparseBallDetection],
    *,
    opacity: float,
) -> None:
    if len(points) < 2:
        return
    ordered = sorted(points, key=lambda point: point.frame_index)
    color = _scale_color(POST_SHOT_COLOR, opacity)
    for point in ordered:
        center = (int(round(point.x)), int(round(point.y)))
        cv2.circle(frame, center, 3, color, -1)
    for start, end in zip(ordered, ordered[1:]):
        _draw_dotted_line(
            frame,
            (int(round(start.x)), int(round(start.y))),
            (int(round(end.x)), int(round(end.y))),
            color,
        )


def _draw_measured_continuity_path(
    frame: np.ndarray,
    shot: ShotCandidate,
    *,
    opacity: float,
) -> None:
    points = continuity_track(shot)
    if not points:
        return

    measured_color = _scale_color(MEASURED_COLOR, opacity)
    gap_color = _scale_color(GAP_PREDICTED_COLOR, opacity)
    for point in points:
        center = (int(round(point.x)), int(round(point.y)))
        if effective_point_source(point) == "gap_predicted":
            cv2.circle(frame, center, 4, gap_color, 2)
        else:
            cv2.circle(frame, center, 4, measured_color, -1)

    if len(points) < 2:
        return

    for start, end in zip(points, points[1:]):
        p0 = (int(round(start.x)), int(round(start.y)))
        p1 = (int(round(end.x)), int(round(end.y)))
        if _uses_gap_prediction(start, end):
            _draw_dotted_line(frame, p0, p1, gap_color)
        else:
            _draw_dotted_line(frame, p0, p1, measured_color)


def _draw_gap_bridges(
    frame: np.ndarray,
    shot: ShotCandidate,
    story: ShotStoryMetadata,
    *,
    opacity: float,
) -> None:
    gap_frames = set(story.gap_predicted_frames)
    if not gap_frames:
        return

    points = [
        point
        for point in continuity_track(shot)
        if story.flight_start_frame <= point.frame_index <= story.flight_end_frame
    ]
    if len(points) < 2:
        return

    color = _scale_color(GAP_PREDICTED_COLOR, opacity)
    for point in points:
        if point.frame_index not in gap_frames:
            continue
        center = (int(round(point.x)), int(round(point.y)))
        cv2.circle(frame, center, 4, color, 2)

    for start, end in zip(points, points[1:]):
        if not _uses_gap_prediction(start, end):
            continue
        _draw_dotted_line(
            frame,
            (int(round(start.x)), int(round(start.y))),
            (int(round(end.x)), int(round(end.y))),
            color,
        )


def _draw_release_marker(
    frame: np.ndarray,
    x: float,
    y: float,
    *,
    opacity: float,
) -> None:
    center = (int(round(x)), int(round(y)))
    color = _scale_color(RELEASE_COLOR, opacity)
    cv2.circle(frame, center, 6, color, 2)
    cv2.circle(frame, center, 2, color, -1)


def _draw_arc_with_opacity(
    frame: np.ndarray,
    shot: ShotCandidate,
    *,
    opacity: float,
    thickness: int,
) -> None:
    del thickness
    if opacity >= 0.99:
        draw_finalized_arc(frame, shot)
        return

    overlay = frame.copy()
    draw_finalized_arc(overlay, shot)
    cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0, frame)


def _draw_flight_diagnostic_dots(
    frame: np.ndarray,
    shot: ShotCandidate,
    story: ShotStoryMetadata,
    *,
    opacity: float,
    max_dots: int,
) -> None:
    diagnostics = shot.fit_diagnostics
    if diagnostics is not None:
        shown = 0
        for point in diagnostics.points:
            if point.frame_index < 0:
                continue
            if point.frame_index < story.flight_start_frame:
                continue
            if point.frame_index > story.flight_end_frame:
                continue
            if shown >= max_dots:
                break
            color = _scale_color(_confidence_color(point.confidence), opacity)
            cv2.circle(
                frame,
                (int(round(point.x)), int(round(point.y))),
                4,
                color,
                -1,
            )
            shown += 1
        return

    shown = 0
    for point in shot.candidate_points:
        if point.frame_index < story.flight_start_frame:
            continue
        if point.frame_index > story.flight_end_frame:
            continue
        if shown >= max_dots:
            break
        color = _scale_color(_confidence_color(point.confidence), opacity)
        cv2.circle(frame, (int(round(point.x)), int(round(point.y))), 4, color, -1)
        shown += 1


def _draw_legend(frame: np.ndarray) -> None:
    legend_x = frame.shape[1] - 210
    y = frame.shape[0] - 72
    entries = [
        ("pickup", PICKUP_COLOR),
        ("release", RELEASE_COLOR),
        ("arc", (80, 220, 255)),
        ("gap", GAP_PREDICTED_COLOR),
        ("post", POST_SHOT_COLOR),
        ("extrap", (120, 180, 220)),
    ]
    for label, color in entries:
        cv2.putText(
            frame,
            label,
            (legend_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
        legend_x += 34


def _uses_gap_prediction(
    start: SparseBallDetection,
    end: SparseBallDetection,
) -> bool:
    return (
        effective_point_source(start) == "gap_predicted"
        or effective_point_source(end) == "gap_predicted"
    )


def _confidence_color(confidence: float) -> tuple[int, int, int]:
    tier = confidence_tier(confidence)
    if tier == "high":
        return HIGH_CONF_COLOR
    if tier == "medium":
        return MEDIUM_CONF_COLOR
    return LOW_CONF_COLOR


def _scale_color(color: tuple[int, int, int], opacity: float) -> tuple[int, int, int]:
    return tuple(int(channel * opacity) for channel in color)


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
