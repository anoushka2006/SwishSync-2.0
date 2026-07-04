"""Make/miss classification from rim-plane crossings.

Pure and render/observability-only: computed once in finalize_shot() after fit
and confidence, never read back by fit, confidence, story, or lifecycle code.

A shot is a make when the ball's final descent crosses the ring line inside the
rim's horizontal span. Two methods, in preference order:

1. ``measured`` — interpolate between the last measured pair straddling the
   ring line. Rare in practice: tracking usually loses the ball to net/backboard
   occlusion just above the rim.
2. ``fit`` — evaluate the fitted parabola's descending branch at the ring line,
   only when the last measured point approached within
   ``MAX_APPROACH_GAP_PX`` of it. No close approach → honest "unknown".

Ring line: for tight rim boxes (automatic hoop lock) the box spans rim+net, so
the ring plane is the box *top*. Hand-drawn boxes often include backboard/pole
(very tall), where the historical ``rim_center_y = bbox[3]`` bottom convention
is the better rim estimate.
"""

from __future__ import annotations

import math

from swishsync_cv.data import (
    HoopLock,
    ParabolaFit,
    ShotOutcome,
    SparseBallDetection,
    is_measured_detection,
)

# ponytail: naive thresholds, tune from eval if verdicts cluster wrong.
MAX_CROSSING_GAP_FRAMES = 20  # straddling pair further apart is too uncertain
MAX_APPROACH_GAP_PX = 160.0  # fit fallback only if ball got this close to ring


def _ring_y(hoop_lock: HoopLock) -> float:
    x1, y1, x2, y2 = hoop_lock.bbox_xyxy
    width = max(x2 - x1, 1.0)
    height = y2 - y1
    # squat box (tight rim+net) → ring is the top; tall box (backboard/pole
    # included, e.g. hand-drawn) → keep the legacy bottom convention
    return y1 if height <= width * 1.5 else y2


def classify_shot_outcome(
    points: list[SparseBallDetection],
    hoop_lock: HoopLock | None,
    parabola_fit: ParabolaFit | None = None,
) -> ShotOutcome:
    """Classify make/miss from measured points and the fitted arc."""

    if hoop_lock is None:
        return ShotOutcome(verdict="unknown")

    measured = sorted(
        (point for point in points if is_measured_detection(point)),
        key=lambda point: point.frame_index,
    )
    if len(measured) < 2:
        return ShotOutcome(verdict="unknown")

    ring_y = _ring_y(hoop_lock)
    rim_x1, _, rim_x2, _ = hoop_lock.bbox_xyxy

    crossing = _measured_crossing(measured, ring_y)
    method = "measured"
    if crossing is None:
        crossing = _fit_crossing(measured, parabola_fit, ring_y)
        method = "fit"
    if crossing is None:
        return ShotOutcome(verdict="unknown")

    crossing_frame, crossing_x = crossing
    half_width = max((rim_x2 - rim_x1) / 2.0, 1.0)
    rim_center_x = (rim_x1 + rim_x2) / 2.0
    margin_ratio = abs(crossing_x - rim_center_x) / half_width
    return ShotOutcome(
        verdict="make" if rim_x1 <= crossing_x <= rim_x2 else "miss",
        crossing_frame=crossing_frame,
        crossing_x=crossing_x,
        rim_x_span=(rim_x1, rim_x2),
        margin_ratio=margin_ratio,
        method=method,
    )


def _measured_crossing(
    measured: list[SparseBallDetection],
    ring_y: float,
) -> tuple[int, float] | None:
    """Last downward measured crossing of the ring line, if any."""

    crossing: tuple[int, float] | None = None
    for above, below in zip(measured, measured[1:]):
        if not (above.y < ring_y <= below.y):
            continue
        if below.frame_index - above.frame_index > MAX_CROSSING_GAP_FRAMES:
            continue
        t = (ring_y - above.y) / (below.y - above.y)
        crossing = (below.frame_index, above.x + t * (below.x - above.x))
    return crossing


def _fit_crossing(
    measured: list[SparseBallDetection],
    parabola_fit: ParabolaFit | None,
    ring_y: float,
) -> tuple[int, float] | None:
    """Descending-branch parabola crossing of the ring line, if the ball got close."""

    if parabola_fit is None:
        return None
    last = measured[-1]
    if last.y < ring_y - MAX_APPROACH_GAP_PX:
        return None  # tracking lost too far above the rim to extrapolate

    a, b, c = parabola_fit.coefficients
    if abs(a) < 1e-12:
        return None
    discriminant = b * b - 4.0 * a * (c - ring_y)
    if discriminant < 0:
        return None  # arc never reaches the ring line

    sqrt_d = math.sqrt(discriminant)
    roots = ((-b - sqrt_d) / (2.0 * a), (-b + sqrt_d) / (2.0 * a))
    direction = last.x - measured[0].x  # horizontal travel direction
    # descending branch: y increases along travel, i.e. sign(dy/dx) == sign(dx)
    candidates = [
        x for x in roots if (2.0 * a * x + b) * direction > 0
    ]
    if not candidates:
        return None
    # of the descending-branch roots, take the one nearest the last observation
    crossing_x = min(candidates, key=lambda x: abs(x - last.x))
    return (last.frame_index, crossing_x)
