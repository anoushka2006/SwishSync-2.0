"""Pre-fit trusted flight point selection: measured-only, contiguous, non-floor-bounce."""

from __future__ import annotations

from swishsync_cv.config import TrustedFlightConfig
from swishsync_cv.data import (
    HoopLock,
    SparseBallDetection,
    TrustedFlightExclusion,
    TrustedFlightSelection,
    effective_point_source,
)
from swishsync_cv.tracking.parabola import (
    is_floor_bounce_point,
    select_contiguous_flight_cluster,
)


def select_trusted_flight_points(
    points: list[SparseBallDetection],
    hoop_lock: HoopLock | None,
    config: TrustedFlightConfig,
    floor_margin_px: float = 100.0,
) -> TrustedFlightSelection:
    """Return measured-only, contiguous, non-floor-bounce points from candidate_points.

    Filter stages applied in order — each individually observable in excluded:
      1. Measured-only: drop gap_predicted synthetic interpolations.
      2. Primary contiguous cluster: split at any gap > config.max_flight_gap_frames (default 10).
         Tighter than reacquisition_gap_frames (15) to isolate the primary arc.
      3. Floor bounce: drop points at y > rim_center_y + floor_margin_px.

    Statistical outlier removal stays inside fit_weighted_parabola_robust.
    Motion-validation filters (horizontal jump, vertical acceleration) are reserved
    for a documented future stage; add each there and prove CORE-safe before merging.

    Result is stored as ShotCandidate.trusted_flight_debug. NOT read by the
    fitter or confidence scorer. Phase 1: compute_shot_story sources its
    render-only flight window from this selection.
    """
    excluded: list[TrustedFlightExclusion] = []

    # Stage 1: measured-only — gap_predicted are synthetic, not real detections.
    measured: list[SparseBallDetection] = []
    for point in points:
        if effective_point_source(point) != "measured":
            excluded.append(TrustedFlightExclusion(point=point, reason="gap_predicted"))
        else:
            measured.append(point)

    # Stage 2: primary contiguous cluster (tighter gap than reacquisition).
    cluster = select_contiguous_flight_cluster(measured, max_gap_frames=config.max_flight_gap_frames)
    cluster_frames = {p.frame_index for p in cluster}
    for point in measured:
        if point.frame_index not in cluster_frames:
            excluded.append(TrustedFlightExclusion(point=point, reason="post_cluster"))

    # Stage 3: floor bounce filter — same geometry as fit_points selection.
    trusted: list[SparseBallDetection] = []
    for point in cluster:
        if is_floor_bounce_point(point, hoop_lock, floor_margin_px):
            excluded.append(TrustedFlightExclusion(point=point, reason="floor_bounce"))
        else:
            trusted.append(point)

    return TrustedFlightSelection(
        trusted=tuple(trusted),
        excluded=tuple(excluded),
        max_gap_frames=config.max_flight_gap_frames,
    )
