"""Confidence-weighted parabolic arc reconstruction."""

from __future__ import annotations

import numpy as np

from swishsync_cv.data import FitDiagnostics, ParabolaFit, PointDiagnostic, SparseBallDetection

LOW_CONFIDENCE_THRESHOLD = 0.30
MEDIUM_CONFIDENCE_THRESHOLD = 0.60
LOW_FITTING_WEIGHT = 0.15
MEDIUM_FITTING_WEIGHT = 0.50
HIGH_FITTING_WEIGHT = 1.00
OUTLIER_RESIDUAL_FLOOR_PX = 35.0


def fitting_weight(confidence: float) -> float:
    """Map detector confidence to a clamped polyfit weight."""

    clamped = max(0.0, min(1.0, confidence))
    if clamped < LOW_CONFIDENCE_THRESHOLD:
        return LOW_FITTING_WEIGHT
    if clamped < MEDIUM_CONFIDENCE_THRESHOLD:
        return MEDIUM_FITTING_WEIGHT
    return HIGH_FITTING_WEIGHT


def confidence_tier(confidence: float) -> str:
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return "low"
    if confidence < MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "high"


def fit_weighted_parabola(
    points: list[SparseBallDetection],
) -> tuple[ParabolaFit, FitDiagnostics] | None:
    """Fit one weighted parabola and return curve plus point diagnostics."""

    if len(points) < 3:
        return None

    xs = np.array([point.x for point in points], dtype=float)
    ys = np.array([point.y for point in points], dtype=float)
    if np.unique(xs).size < 3:
        return None

    weights = np.array([fitting_weight(point.confidence) for point in points], dtype=float)
    coefficients = np.polyfit(xs, ys, deg=2, w=weights)
    a, b, c = (float(value) for value in coefficients)
    predicted = a * xs * xs + b * xs + c
    residuals = np.abs(ys - predicted)

    weighted_mean_y = float(np.average(ys, weights=weights))
    ss_res = float(np.sum(weights * (ys - predicted) ** 2))
    ss_tot = float(np.sum(weights * (ys - weighted_mean_y) ** 2))
    weighted_r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    weighted_rmse = float(np.sqrt(ss_res / max(np.sum(weights), 1e-6)))

    unweighted_r_squared = _unweighted_r_squared(ys, predicted)
    apex_x = -b / (2 * a) if abs(a) > 1e-8 else float(np.mean(xs))
    apex_y = a * apex_x * apex_x + b * apex_x + c

    outlier_threshold = _outlier_threshold(residuals)
    diagnostics = _build_point_diagnostics(points, residuals, outlier_threshold)

    fit = ParabolaFit(
        coefficients=(a, b, c),
        r_squared=max(0.0, min(1.0, unweighted_r_squared)),
        apex_x=float(apex_x),
        apex_y=float(apex_y),
        x_min=float(np.min(xs)),
        x_max=float(np.max(xs)),
        weighted_r_squared=max(0.0, min(1.0, weighted_r_squared)),
        weighted_residual_rmse=weighted_rmse,
    )
    aggregate = FitDiagnostics(
        point_count=len(points),
        average_detection_confidence=float(np.mean([point.confidence for point in points])),
        weighted_residual_rmse=weighted_rmse,
        outlier_count=sum(1 for point in diagnostics if point.is_outlier),
        points=tuple(diagnostics),
    )
    return fit, aggregate


def fit_parabola(points: list[SparseBallDetection]) -> ParabolaFit | None:
    """Backward-compatible wrapper around weighted fitting."""

    result = fit_weighted_parabola(points)
    if result is None:
        return None
    fit, _ = result
    return fit


def _unweighted_r_squared(ys: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = float(np.sum((ys - predicted) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def _outlier_threshold(residuals: np.ndarray) -> float:
    mean_residual = float(np.mean(residuals))
    std_residual = float(np.std(residuals))
    return max(OUTLIER_RESIDUAL_FLOOR_PX, mean_residual + 1.5 * std_residual)


def _build_point_diagnostics(
    points: list[SparseBallDetection],
    residuals: np.ndarray,
    outlier_threshold: float,
) -> list[PointDiagnostic]:
    diagnostics: list[PointDiagnostic] = []
    for point, residual in zip(points, residuals):
        diagnostics.append(
            PointDiagnostic(
                frame_index=point.frame_index,
                x=point.x,
                y=point.y,
                confidence=point.confidence,
                fitting_weight=fitting_weight(point.confidence),
                residual_px=float(residual),
                is_outlier=float(residual) > outlier_threshold,
            )
        )
    return diagnostics
