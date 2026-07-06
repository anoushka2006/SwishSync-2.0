"""Render a half-court shot chart with make/miss markers.

Court coords are feet in the convention of homography.py (origin at the hoop
floor point, +x toward half-court, +y toward the camera's right sideline).
Standard dims: 50ft wide, hoop 5.25ft from baseline, 16ft lane, ~23.75ft arc.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

COURT_COLOR = (30, 30, 34)
LINE_COLOR = (150, 150, 150)
MAKE_COLOR = (90, 210, 90)   # green (BGR)
MISS_COLOR = (80, 80, 230)   # red (BGR)

HALF_LENGTH_FT = 47.0
WIDTH_FT = 50.0
HOOP_FROM_BASELINE_FT = 5.25
LANE_WIDTH_FT = 16.0
LANE_LENGTH_FT = 19.0
THREE_RADIUS_FT = 23.75


@dataclass(frozen=True)
class ShotMark:
    court_x: float  # feet from hoop toward half-court
    court_y: float  # feet toward right sideline (signed)
    make: bool


def render_shot_chart(
    marks: list[ShotMark],
    scale_px_per_ft: int = 14,
    margin_ft: float = 3.0,
) -> np.ndarray:
    """Return a BGR half-court image with the shot marks plotted."""

    # image spans x in [-HOOP_FROM_BASELINE, HALF_LENGTH], y in [-25, 25]
    x_min = -HOOP_FROM_BASELINE_FT - margin_ft
    x_max = HALF_LENGTH_FT + margin_ft
    y_min = -WIDTH_FT / 2 - margin_ft
    y_max = WIDTH_FT / 2 + margin_ft
    w = int((x_max - x_min) * scale_px_per_ft)
    h = int((y_max - y_min) * scale_px_per_ft)
    img = np.full((h, w, 3), COURT_COLOR, dtype=np.uint8)

    def to_px(cx: float, cy: float) -> tuple[int, int]:
        px = int((cx - x_min) * scale_px_per_ft)
        py = int((cy - y_min) * scale_px_per_ft)
        return px, py

    # baseline, sidelines, half-court line
    bl = to_px(-HOOP_FROM_BASELINE_FT, -WIDTH_FT / 2)
    tr = to_px(HALF_LENGTH_FT, WIDTH_FT / 2)
    cv2.rectangle(img, bl, tr, LINE_COLOR, 2)
    # lane
    lane_a = to_px(-HOOP_FROM_BASELINE_FT, -LANE_WIDTH_FT / 2)
    lane_b = to_px(-HOOP_FROM_BASELINE_FT + LANE_LENGTH_FT, LANE_WIDTH_FT / 2)
    cv2.rectangle(img, lane_a, lane_b, LINE_COLOR, 2)
    # hoop
    cv2.circle(img, to_px(0.0, 0.0), max(2, int(0.75 * scale_px_per_ft)), LINE_COLOR, 2)
    # three-point arc (approx: circle centered at hoop)
    cx, cy = to_px(0.0, 0.0)
    cv2.ellipse(img, (cx, cy), (int(THREE_RADIUS_FT * scale_px_per_ft),) * 2,
                0, -90, 90, LINE_COLOR, 2)

    for mark in marks:
        px, py = to_px(mark.court_x, mark.court_y)
        if not (0 <= px < w and 0 <= py < h):
            continue
        color = MAKE_COLOR if mark.make else MISS_COLOR
        cv2.circle(img, (px, py), 7, color, -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), 7, (20, 20, 20), 1, cv2.LINE_AA)

    makes = sum(1 for m in marks if m.make)
    cv2.putText(img, f"{makes}/{len(marks)} made", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, LINE_COLOR, 1, cv2.LINE_AA)
    return img
