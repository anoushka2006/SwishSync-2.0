"""Post-collection shot reconstruction: fit exactly one weighted parabola."""

from __future__ import annotations

import logging

from swishsync_cv.config import ShotCandidateConfig
from swishsync_cv.data import ShotCandidate
from swishsync_cv.tracking.confidence import score_shot_confidence
from swishsync_cv.tracking.parabola import fit_weighted_parabola

logger = logging.getLogger("swishsync_cv.shot")


def finalize_shot(
    candidate: ShotCandidate,
    config: ShotCandidateConfig,
    interpolated_ignored_count: int = 0,
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

    if len(candidate.candidate_points) >= config.min_validated_points_for_fit:
        result = fit_weighted_parabola(candidate.candidate_points)
        if result is not None:
            candidate.parabola_fit, candidate.fit_diagnostics = result
            candidate.confidence = score_shot_confidence(candidate)
            _log_outliers(candidate)

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
            "shot summary start=%s end=%s measured=%s interpolated_ignored=%s "
            "weighted_rmse=%.1fpx apex=(%.1f, %.1f) trajectory_confidence=%.0f%%",
            candidate.start_frame,
            candidate.end_frame,
            len(candidate.candidate_points),
            interpolated_ignored_count,
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


def _log_outliers(candidate: ShotCandidate) -> None:
    if candidate.fit_diagnostics is None:
        return

    for point in candidate.fit_diagnostics.points:
        if not point.is_outlier:
            continue
        logger.info(
            "outlier frame=%s x=%.1f y=%.1f confidence=%.2f residual=%.1fpx",
            point.frame_index,
            point.x,
            point.y,
            point.confidence,
            point.residual_px,
        )
