"""Post-collection shot reconstruction: fit exactly one parabola."""

from __future__ import annotations

import logging

from swishsync_cv.config import ShotCandidateConfig
from swishsync_cv.data import ShotCandidate
from swishsync_cv.tracking.parabola import fit_parabola

logger = logging.getLogger("swishsync_cv.shot")


def finalize_shot(
    candidate: ShotCandidate,
    config: ShotCandidateConfig,
) -> ShotCandidate:
    """Freeze buffered points and fit one parabola with polyfit."""

    logger.info(
        "shot finalized frame=%s points=%s",
        candidate.end_frame,
        len(candidate.candidate_points),
    )

    candidate.parabola_fit = None
    if len(candidate.candidate_points) >= config.min_validated_points_for_fit:
        candidate.parabola_fit = fit_parabola(candidate.candidate_points)

    if candidate.parabola_fit is not None:
        logger.info(
            "parabola fit complete r2=%.3f apex=(%.1f, %.1f)",
            candidate.parabola_fit.r_squared,
            candidate.parabola_fit.apex_x,
            candidate.parabola_fit.apex_y,
        )
    else:
        logger.info("parabola fit skipped (not enough usable points)")

    candidate.state = "shot_finalized"
    return candidate
