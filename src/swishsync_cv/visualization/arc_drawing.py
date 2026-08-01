"""Shared finalized-arc drawing with optional rim extension."""

from __future__ import annotations

import cv2
import numpy as np

from swishsync_cv.data import ShotCandidate

FINAL_ARC_COLOR = (80, 220, 255)
EXTENDED_ARC_COLOR = (120, 180, 220)
MAKE_ARC_COLOR = (90, 210, 90)  # green (BGR)
MISS_ARC_COLOR = (80, 80, 230)  # red (BGR)
ARC_OUTLINE_COLOR = (25, 25, 25)  # dark edge for legibility on sky/dark bg

ARC_THICKNESS = 5  # thicker than the 4px-radius trajectory dots
ARC_OUTLINE_THICKNESS = ARC_THICKNESS + 3


def _arc_color(shot: ShotCandidate) -> tuple[int, int, int]:
    verdict = shot.outcome.verdict if shot.outcome is not None else None
    if verdict == "make":
        return MAKE_ARC_COLOR
    if verdict == "miss":
        return MISS_ARC_COLOR
    return FINAL_ARC_COLOR


def draw_finalized_arc(frame: np.ndarray, display_shot: ShotCandidate) -> None:
    fit = display_shot.parabola_fit
    if fit is None or display_shot.insufficient_points_for_fit:
        return

    arc_render = display_shot.arc_render
    if arc_render is None:
        arc_points = fit.sample_arc(num_points=96)
    else:
        # one continuous smooth arc across the whole render span (release → rim),
        # not a stubby observed segment plus a dotted extension
        x_min, x_max = arc_render.render_x_range
        arc_points = fit.sample_arc_range(x_min, x_max, num_points=96)

    # dark underlay then bright colored line = crisp edge on any background
    _draw_solid_polyline(frame, arc_points, ARC_OUTLINE_COLOR, ARC_OUTLINE_THICKNESS)
    _draw_solid_polyline(frame, arc_points, _arc_color(display_shot), ARC_THICKNESS)


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
