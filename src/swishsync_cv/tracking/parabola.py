"""Confidence-weighted parabolic arc reconstruction."""

from __future__ import annotations

import numpy as np

from swishsync_cv.data import FitDiagnostics, HoopLock, ParabolaFit, PointDiagnostic, SparseBallDetection

LOW_CONFIDENCE_THRESHOLD = 0.30
MEDIUM_CONFIDENCE_THRESHOLD = 0.60
LOW_FITTING_WEIGHT = 0.15
MEDIUM_FITTING_WEIGHT = 0.50
HIGH_FITTING_WEIGHT = 1.00
RIM_ANCHOR_WEIGHT = 2.0
OUTLIER_RESIDUAL_FLOOR_PX = 35.0
RIM_ANCHOR_FRAME_INDEX = -1


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


def is_floor_bounce_point(
    point: SparseBallDetection,
    hoop_lock: HoopLock | None,
    floor_margin_px: float = 100.0,
) -> bool:
    """True when the ball is far below the rim (court/floor bounce zone)."""

    if hoop_lock is None or not hoop_lock.is_locked:
        return False
    return point.y > hoop_lock.rim_center_y + floor_margin_px


def select_flight_fit_points(
    points: list[SparseBallDetection],
    hoop_lock: HoopLock | None,
    floor_margin_px: float = 100.0,
) -> tuple[list[SparseBallDetection], list[SparseBallDetection]]:
    """Split release/flight/rim-approach points from floor bounce points."""

    fit_points: list[SparseBallDetection] = []
    excluded: list[SparseBallDetection] = []
    for point in points:
        if is_floor_bounce_point(point, hoop_lock, floor_margin_px):
            excluded.append(point)
        else:
            fit_points.append(point)
    return fit_points, excluded


def select_contiguous_flight_cluster(
    points: list[SparseBallDetection],
    max_gap_frames: int = 15,
) -> list[SparseBallDetection]:
    """Keep measured points from release until the first adjacent gap > max_gap_frames."""

    if not points:
        return []

    ordered = sorted(points, key=lambda point: point.frame_index)
    cluster = [ordered[0]]
    for point in ordered[1:]:
        if point.frame_index - cluster[-1].frame_index > max_gap_frames:
            break
        cluster.append(point)
    return cluster


def analyze_trajectory_completeness(
    points: list[SparseBallDetection],
    max_gap_frames: int = 15,
) -> tuple[bool, int | None]:
    """Return whether a later cluster exists and the relevant gap size for metadata/HUD."""

    cluster = select_contiguous_flight_cluster(points, max_gap_frames)
    ordered = sorted(points, key=lambda point: point.frame_index)

    max_gap_in_cluster = 0
    for start, end in zip(cluster, cluster[1:]):
        max_gap_in_cluster = max(max_gap_in_cluster, end.frame_index - start.frame_index)

    if len(cluster) < len(ordered):
        split_gap = ordered[len(cluster)].frame_index - cluster[-1].frame_index
        return True, split_gap

    return False, max_gap_in_cluster if max_gap_in_cluster > 0 else None


def fit_weighted_parabola(
    points: list[SparseBallDetection],
    rim_anchor: tuple[float, float] | None = None,
    rim_anchor_weight: float = RIM_ANCHOR_WEIGHT,
) -> tuple[ParabolaFit, FitDiagnostics] | None:
    """Fit one weighted parabola and return curve plus point diagnostics."""

    if len(points) < 3:
        return None

    xs = np.array([point.x for point in points], dtype=float)
    ys = np.array([point.y for point in points], dtype=float)
    weights = np.array([fitting_weight(point.confidence) for point in points], dtype=float)
    fit_points: list[SparseBallDetection] = list(points)

    if rim_anchor is not None:
        rim_x, rim_y = rim_anchor
        xs = np.append(xs, rim_x)
        ys = np.append(ys, rim_y)
        weights = np.append(weights, rim_anchor_weight)
        fit_points.append(
            SparseBallDetection(
                frame_index=RIM_ANCHOR_FRAME_INDEX,
                timestamp_ms=0.0,
                x=rim_x,
                y=rim_y,
                confidence=1.0,
                interpolated=False,
            )
        )

    if np.unique(xs).size < 3:
        return None

    coefficients = np.polyfit(xs, ys, deg=2, w=weights)
    return _build_fit_from_coefficients(
        coefficients=coefficients,
        fit_points=fit_points,
        ball_points=points,
        rim_anchor_weight=rim_anchor_weight,
    )


def fit_weighted_parabola_robust(
    points: list[SparseBallDetection],
    rim_anchor: tuple[float, float] | None = None,
    rim_anchor_weight: float = RIM_ANCHOR_WEIGHT,
    excluded_fit_points: list[SparseBallDetection] | None = None,
) -> tuple[ParabolaFit, FitDiagnostics] | None:
    """Fit, remove strong outliers, refit once, and return merged diagnostics."""

    if len(points) < 3:
        return None

    initial = fit_weighted_parabola(points, rim_anchor, rim_anchor_weight)
    if initial is None:
        return None

    _, initial_diag = initial
    initial_rmse = initial_diag.weighted_residual_rmse

    outlier_frames = {
        point.frame_index
        for point in initial_diag.points
        if point.is_outlier and point.frame_index >= 0
    }
    trimmed = [point for point in points if point.frame_index not in outlier_frames]

    if len(outlier_frames) == 0 or len(trimmed) < 3:
        fit, diagnostics = initial
        merged = _merge_excluded_diagnostics(
            diagnostics,
            excluded_fit_points or [],
            fit.coefficients,
            rim_anchor_weight,
            initial_rmse=initial_rmse,
        )
        return fit, merged

    refit = fit_weighted_parabola(trimmed, rim_anchor, rim_anchor_weight)
    if refit is None:
        fit, diagnostics = initial
        merged = _merge_excluded_diagnostics(
            diagnostics,
            excluded_fit_points or [],
            fit.coefficients,
            rim_anchor_weight,
            initial_rmse=initial_rmse,
        )
        return fit, merged

    fit, refit_diag = refit
    merged = _merge_excluded_diagnostics(
        refit_diag,
        excluded_fit_points or [],
        fit.coefficients,
        rim_anchor_weight,
        initial_rmse=initial_rmse,
        outlier_frames=outlier_frames,
    )
    return fit, merged


def _build_fit_from_coefficients(
    coefficients: np.ndarray,
    fit_points: list[SparseBallDetection],
    ball_points: list[SparseBallDetection],
    rim_anchor_weight: float,
) -> tuple[ParabolaFit, FitDiagnostics]:
    a, b, c = (float(value) for value in coefficients)
    xs = np.array([point.x for point in fit_points], dtype=float)
    ys = np.array([point.y for point in fit_points], dtype=float)
    weights = np.array(
        [
            rim_anchor_weight
            if point.frame_index == RIM_ANCHOR_FRAME_INDEX
            else fitting_weight(point.confidence)
            for point in fit_points
        ],
        dtype=float,
    )
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
    diagnostics = _build_point_diagnostics(fit_points, residuals, outlier_threshold, rim_anchor_weight)

    fit = ParabolaFit(
        coefficients=(a, b, c),
        r_squared=max(0.0, min(1.0, unweighted_r_squared)),
        apex_x=float(apex_x),
        apex_y=float(apex_y),
        x_min=float(np.min([point.x for point in ball_points])),
        x_max=float(np.max([point.x for point in ball_points])),
        weighted_r_squared=max(0.0, min(1.0, weighted_r_squared)),
        weighted_residual_rmse=weighted_rmse,
    )
    aggregate = FitDiagnostics(
        point_count=len(ball_points),
        average_detection_confidence=float(np.mean([point.confidence for point in ball_points])),
        weighted_residual_rmse=weighted_rmse,
        outlier_count=sum(1 for point in diagnostics if point.is_outlier),
        points=tuple(diagnostics),
        fit_point_count=len(ball_points),
    )
    return fit, aggregate


def _merge_excluded_diagnostics(
    fit_diagnostics: FitDiagnostics,
    excluded_fit_points: list[SparseBallDetection],
    coefficients: tuple[float, float, float],
    rim_anchor_weight: float,
    initial_rmse: float,
    outlier_frames: set[int] | None = None,
) -> FitDiagnostics:
    a, b, c = coefficients
    outlier_frames = outlier_frames or set()
    merged_points: list[PointDiagnostic] = []

    for point in fit_diagnostics.points:
        if point.frame_index < 0:
            merged_points.append(point)
            continue
        removed_as_outlier = point.frame_index in outlier_frames
        merged_points.append(
            PointDiagnostic(
                frame_index=point.frame_index,
                x=point.x,
                y=point.y,
                confidence=point.confidence,
                fitting_weight=point.fitting_weight,
                residual_px=point.residual_px,
                is_outlier=point.is_outlier or removed_as_outlier,
                used_in_fit=not removed_as_outlier,
                excluded_from_fit=removed_as_outlier,
            )
        )

    for point in excluded_fit_points:
        residual = abs(point.y - (a * point.x * point.x + b * point.x + c))
        merged_points.append(
            PointDiagnostic(
                frame_index=point.frame_index,
                x=point.x,
                y=point.y,
                confidence=point.confidence,
                fitting_weight=0.0,
                residual_px=float(residual),
                is_outlier=True,
                used_in_fit=False,
                excluded_from_fit=True,
            )
        )

    fit_count = sum(1 for point in merged_points if point.used_in_fit and point.frame_index >= 0)
    outlier_count = sum(
        1
        for point in merged_points
        if point.is_outlier and point.frame_index >= 0
    )
    ball_points = [point for point in merged_points if point.frame_index >= 0 and point.used_in_fit]
    avg_conf = (
        float(np.mean([point.confidence for point in ball_points]))
        if ball_points
        else fit_diagnostics.average_detection_confidence
    )

    return FitDiagnostics(
        point_count=fit_diagnostics.point_count + len(excluded_fit_points),
        average_detection_confidence=avg_conf,
        weighted_residual_rmse=fit_diagnostics.weighted_residual_rmse,
        outlier_count=outlier_count,
        points=tuple(merged_points),
        initial_weighted_residual_rmse=initial_rmse,
        fit_point_count=fit_count,
    )


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
    rim_anchor_weight: float = RIM_ANCHOR_WEIGHT,
) -> list[PointDiagnostic]:
    diagnostics: list[PointDiagnostic] = []
    for point, residual in zip(points, residuals):
        weight = (
            rim_anchor_weight
            if point.frame_index == RIM_ANCHOR_FRAME_INDEX
            else fitting_weight(point.confidence)
        )
        is_outlier = float(residual) > outlier_threshold and point.frame_index >= 0
        diagnostics.append(
            PointDiagnostic(
                frame_index=point.frame_index,
                x=point.x,
                y=point.y,
                confidence=point.confidence,
                fitting_weight=weight,
                residual_px=float(residual),
                is_outlier=is_outlier,
                used_in_fit=True,
                excluded_from_fit=False,
            )
        )
    return diagnostics
