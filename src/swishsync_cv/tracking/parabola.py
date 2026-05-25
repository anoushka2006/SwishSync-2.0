"""Parabolic arc reconstruction from validated shot points."""

from __future__ import annotations

import numpy as np

from swishsync_cv.data import ParabolaFit, SparseBallDetection


def fit_parabola(points: list[SparseBallDetection]) -> ParabolaFit | None:
    """Fit y = a*x^2 + b*x + c to validated points."""

    if len(points) < 3:
        return None

    xs = np.array([point.x for point in points], dtype=float)
    ys = np.array([point.y for point in points], dtype=float)
    if np.unique(xs).size < 3:
        return None

    coefficients = np.polyfit(xs, ys, deg=2)
    a, b, c = (float(value) for value in coefficients)
    predicted = a * xs * xs + b * xs + c

    ss_res = float(np.sum((ys - predicted) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    apex_x = -b / (2 * a) if abs(a) > 1e-8 else float(np.mean(xs))
    apex_y = a * apex_x * apex_x + b * apex_x + c

    return ParabolaFit(
        coefficients=(a, b, c),
        r_squared=max(0.0, min(1.0, r_squared)),
        apex_x=float(apex_x),
        apex_y=float(apex_y),
        x_min=float(np.min(xs)),
        x_max=float(np.max(xs)),
    )
