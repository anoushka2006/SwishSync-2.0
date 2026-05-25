"""Shared finalized-arc drawing with optional rim extension."""

from __future__ import annotations

import cv2
import numpy as np

from swishsync_cv.data import ShotCandidate

FINAL_ARC_COLOR = (80, 220, 255)
EXTENDED_ARC_COLOR = (120, 180, 220)


def draw_finalized_arc(frame: np.ndarray, display_shot: ShotCandidate) -> None:
    fit = display_shot.parabola_fit
    if fit is None or display_shot.insufficient_points_for_fit:
        return

    arc_render = display_shot.arc_render
    if arc_render is None:
        _draw_solid_polyline(frame, fit.sample_arc(num_points=80), FINAL_ARC_COLOR, 3)
        return

    fit_x_min, fit_x_max = arc_render.fit_x_range
    observed_points = fit.sample_arc_range(fit_x_min, fit_x_max, num_points=64)
    _draw_solid_polyline(frame, observed_points, FINAL_ARC_COLOR, 3)

    if not arc_render.visual_extension_used:
        return

    if arc_render.observed_segment_end is None or arc_render.extended_segment_end is None:
        return

    observed_x = arc_render.observed_segment_end[0]
    extended_x = arc_render.extended_segment_end[0]
    if abs(extended_x - observed_x) < 1.0:
        return

    extension_points = fit.sample_arc_range(observed_x, extended_x, num_points=24)
    _draw_dotted_polyline(frame, extension_points, EXTENDED_ARC_COLOR, 2)


def _draw_solid_polyline(
    frame: np.ndarray,
    arc_points: list[tuple[float, float]],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    pixel_points = [(int(round(x)), int(round(y))) for x, y in arc_points]
    if len(pixel_points) < 2:
        return
    cv2.polylines(
        frame,
        [np.asarray(pixel_points, dtype=np.int32)],
        isClosed=False,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def _draw_dotted_polyline(
    frame: np.ndarray,
    arc_points: list[tuple[float, float]],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    pixel_points = [(int(round(x)), int(round(y))) for x, y in arc_points]
    if len(pixel_points) < 2:
        return
    for start, end in zip(pixel_points, pixel_points[1:]):
        _draw_dotted_line(frame, start, end, color, thickness)


def _draw_dotted_line(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
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
        cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)
