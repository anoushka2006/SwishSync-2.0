"""Post-collection shot reconstruction: fit exactly one weighted parabola."""

from __future__ import annotations

import logging

from swishsync_cv.config import HoopLockConfig, ShotCandidateConfig
from swishsync_cv.data import HoopLock, ShotCandidate
from swishsync_cv.tracking.arc_render import compute_arc_render_metadata
from swishsync_cv.tracking.confidence import score_shot_confidence
from swishsync_cv.tracking.parabola import (
    fit_weighted_parabola_robust,
    select_flight_fit_points,
)

logger = logging.getLogger("swishsync_cv.shot")


def finalize_shot(
    candidate: ShotCandidate,
    config: ShotCandidateConfig,
    interpolated_ignored_count: int = 0,
    hoop_lock: HoopLock | None = None,
    hoop_lock_config: HoopLockConfig | None = None,
) -> ShotCandidate:
    """Freeze buffered points and fit one confidence-weighted parabola."""

    logger.info(
        "shot finalized frame=%s points=%s",
        candidate.end_frame,
        len(candidate.candidate_points),
    )

    candidate.parabola_fit = None
    candidate.fit_diagnostics = None
    candidate.confidence = None
    candidate.insufficient_points_for_fit = False

    measured_count = len(candidate.candidate_points)
    fit_points: list = []
    excluded_for_fit: list = []
    if measured_count >= config.min_measured_points_for_fit:
        fit_points, excluded_for_fit = select_flight_fit_points(
            candidate.candidate_points,
            hoop_lock,
            floor_margin_px=config.floor_below_rim_margin_px,
        )
        candidate.excluded_debug_points.extend(excluded_for_fit)

        if len(excluded_for_fit) > 0:
            logger.info(
                "flight fit excluded floor/bounce frames=%s",
                [point.frame_index for point in excluded_for_fit],
            )

        if len(fit_points) < config.min_measured_points_for_fit:
            fit_points = list(candidate.candidate_points)
            excluded_for_fit = []
            logger.info(
                "flight subset too small (%s); falling back to all candidate points",
                len(fit_points),
            )

        rim_anchor = None
        rim_anchor_weight = (
            hoop_lock_config.rim_anchor_weight if hoop_lock_config is not None else 2.0
        )
        regression_ratio = config.rim_anchor_rmse_regression_ratio
        if hoop_lock is not None and hoop_lock.is_locked:
            rim_anchor = (hoop_lock.rim_center_x, hoop_lock.rim_center_y)

        unanchored = fit_weighted_parabola_robust(
            fit_points,
            rim_anchor=None,
            excluded_fit_points=excluded_for_fit,
        )
        anchored = None
        if rim_anchor is not None:
            anchored = fit_weighted_parabola_robust(
                fit_points,
                rim_anchor=rim_anchor,
                rim_anchor_weight=rim_anchor_weight,
                excluded_fit_points=excluded_for_fit,
            )

        result = _select_best_fit(unanchored, anchored, regression_ratio)
        if result is not None:
            candidate.parabola_fit, candidate.fit_diagnostics, rim_anchor_used = result
            candidate.confidence = score_shot_confidence(candidate)
            candidate.arc_render = compute_arc_render_metadata(
                candidate.parabola_fit,
                fit_points,
                hoop_lock,
                rim_anchor_used,
                config,
            )
            _log_fit_summary(candidate, fit_points, excluded_for_fit)
            _log_outliers(candidate)
            if candidate.arc_render.visual_extension_used:
                logger.info(
                    "visual arc extension fit_x=%.0f..%.0f render_x=%.0f..%.0f",
                    candidate.arc_render.fit_x_range[0],
                    candidate.arc_render.fit_x_range[1],
                    candidate.arc_render.render_x_range[0],
                    candidate.arc_render.render_x_range[1],
                )
    elif measured_count > 0:
        candidate.insufficient_points_for_fit = True
        logger.info(
            "parabola fit skipped insufficient measured points=%s (need %s)",
            measured_count,
            config.min_measured_points_for_fit,
        )

    if candidate.parabola_fit is not None and candidate.fit_diagnostics is not None:
        logger.info(
            "parabola fit complete r2=%.3f weighted_r2=%.3f rmse=%.1f apex=(%.1f, %.1f)",
            candidate.parabola_fit.r_squared,
            candidate.parabola_fit.weighted_r_squared,
            candidate.fit_diagnostics.weighted_residual_rmse,
            candidate.parabola_fit.apex_x,
            candidate.parabola_fit.apex_y,
        )
        if candidate.confidence is not None:
            logger.info(
                "trajectory confidence=%.0f%% avg_detection=%.2f",
                candidate.confidence.trajectory_confidence * 100,
                candidate.fit_diagnostics.average_detection_confidence,
            )
        logger.info(
            "shot summary start=%s end=%s measured=%s fit=%s excluded=%s "
            "rmse_initial=%.1fpx rmse_final=%.1fpx apex=(%.1f, %.1f) trajectory_confidence=%.0f%%",
            candidate.start_frame,
            candidate.end_frame,
            len(candidate.candidate_points),
            candidate.fit_diagnostics.fit_point_count or len(fit_points),
            len(candidate.excluded_debug_points),
            candidate.fit_diagnostics.initial_weighted_residual_rmse or 0.0,
            candidate.fit_diagnostics.weighted_residual_rmse,
            candidate.parabola_fit.apex_x,
            candidate.parabola_fit.apex_y,
            candidate.confidence.trajectory_confidence * 100
            if candidate.confidence is not None
            else 0.0,
        )
    else:
        logger.info("parabola fit skipped (not enough usable points)")

    candidate.state = "shot_finalized"
    return candidate


def _select_best_fit(
    unanchored: tuple | None,
    anchored: tuple | None,
    regression_ratio: float,
) -> tuple | None:
    if unanchored is None and anchored is None:
        return None
    if unanchored is None:
        fit, diag = anchored
        return fit, diag, True
    if anchored is None:
        fit, diag = unanchored
        return fit, diag, False

    unanchored_fit, unanchored_diag = unanchored
    anchored_fit, anchored_diag = anchored
    if anchored_diag.weighted_residual_rmse <= unanchored_diag.weighted_residual_rmse * regression_ratio:
        logger.info(
            "using rim anchor rmse=%.1f (unanchored=%.1f)",
            anchored_diag.weighted_residual_rmse,
            unanchored_diag.weighted_residual_rmse,
        )
        return anchored_fit, anchored_diag, True

    logger.info(
        "skipping rim anchor rmse=%.1f worse than unanchored=%.1f",
        anchored_diag.weighted_residual_rmse,
        unanchored_diag.weighted_residual_rmse,
    )
    return unanchored_fit, unanchored_diag, False


def _log_fit_summary(
    candidate: ShotCandidate,
    fit_points: list,
    excluded_for_fit: list,
) -> None:
    if candidate.fit_diagnostics is None:
        return
    logger.info(
        "fit points frames=%s excluded frames=%s",
        [point.frame_index for point in fit_points],
        [point.frame_index for point in excluded_for_fit],
    )


def _log_outliers(candidate: ShotCandidate) -> None:
    if candidate.fit_diagnostics is None:
        return

    for point in candidate.fit_diagnostics.points:
        if point.frame_index < 0:
            continue
        if not point.is_outlier and not point.excluded_from_fit:
            continue
        logger.info(
            "outlier frame=%s x=%.1f y=%.1f confidence=%.2f residual=%.1fpx excluded=%s",
            point.frame_index,
            point.x,
            point.y,
            point.confidence,
            point.residual_px,
            point.excluded_from_fit,
        )
