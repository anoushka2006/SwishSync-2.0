"""Short-gap parabolic continuity for active shot collection."""

from __future__ import annotations

from swishsync_cv.config import GapRecoveryConfig, ShotCandidateConfig
from swishsync_cv.data import HoopLock, ParabolaFit, ShotCandidate, SparseBallDetection
from swishsync_cv.tracking.motion_validation import validate_point
from swishsync_cv.tracking.parabola import fit_weighted_parabola, is_floor_bounce_point


def measured_tail(points: list[SparseBallDetection]) -> list[SparseBallDetection]:
    return [point for point in points if not point.interpolated]


def open_gap_frames(measured_points: list[SparseBallDetection], frame_index: int) -> int:
    if not measured_points:
        return 0
    return frame_index - measured_points[-1].frame_index


def can_predict_gap(
    measured_points: list[SparseBallDetection],
    frame_index: int,
    hoop_lock: HoopLock | None,
    shot_config: ShotCandidateConfig,
    *,
    post_rim_started: bool,
) -> bool:
    gap_cfg = shot_config.gap_recovery
    if post_rim_started:
        return False
    if len(measured_points) < gap_cfg.min_measured_for_gap_predict:
        return False

    gap = open_gap_frames(measured_points, frame_index)
    if gap < 1 or gap > gap_cfg.max_short_gap_fill_frames:
        return False

    predicted = predict_gap_point(
        measured_points=measured_points,
        frame_index=frame_index,
        timestamp_ms=_estimate_timestamp(measured_points, frame_index),
        hoop_lock=hoop_lock,
        shot_config=shot_config,
    )
    return predicted is not None


def predict_gap_point(
    measured_points: list[SparseBallDetection],
    frame_index: int,
    timestamp_ms: float,
    hoop_lock: HoopLock | None,
    shot_config: ShotCandidateConfig,
) -> SparseBallDetection | None:
    gap_cfg = shot_config.gap_recovery
    if len(measured_points) < gap_cfg.min_measured_for_gap_predict:
        return None

    gap = open_gap_frames(measured_points, frame_index)
    if gap < 1 or gap > gap_cfg.max_short_gap_fill_frames:
        return None

    tail = measured_points[-gap_cfg.gap_predict_tail_points :]
    previous = tail[-2]
    latest = tail[-1]
    dt = max(frame_index - latest.frame_index, 1)

    if gap <= 2:
        dt_prev = max(latest.frame_index - previous.frame_index, 1)
        vx = (latest.x - previous.x) / dt_prev
        vy = (latest.y - previous.y) / dt_prev
        x = latest.x + vx * dt
        y = latest.y + vy * dt
    else:
        local_fit = fit_weighted_parabola(tail)
        if local_fit is None:
            dt_prev = max(latest.frame_index - previous.frame_index, 1)
            vx = (latest.x - previous.x) / dt_prev
            vy = (latest.y - previous.y) / dt_prev
            x = latest.x + vx * dt
            y = latest.y + vy * dt
        else:
            fit, _diagnostics = local_fit
            dt_prev = max(latest.frame_index - previous.frame_index, 1)
            vx = (latest.x - previous.x) / dt_prev
            x = latest.x + vx * dt
            y = fit.evaluate_y(x)

    candidate = SparseBallDetection(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        x=float(x),
        y=float(y),
        confidence=gap_cfg.synthetic_gap_confidence,
        interpolated=False,
        source="gap_predicted",
    )

    if is_floor_bounce_point(
        candidate,
        hoop_lock,
        floor_margin_px=shot_config.floor_below_rim_margin_px,
    ):
        return None

    if not validate_point(
        candidate,
        measured_points,
        shot_config,
        parabola_fit=_local_parabola_fit(tail),
    ):
        return None

    return candidate


def backfill_gap_points(
    measured_points: list[SparseBallDetection],
    reacquisition: SparseBallDetection,
    hoop_lock: HoopLock | None,
    shot_config: ShotCandidateConfig,
) -> list[SparseBallDetection]:
    if len(measured_points) < shot_config.gap_recovery.min_measured_for_gap_predict:
        return []

    last_measured = measured_points[-1]
    gap = reacquisition.frame_index - last_measured.frame_index - 1
    if gap < 1 or gap > shot_config.gap_recovery.max_short_gap_fill_frames:
        return []

    local_fit = _local_parabola_fit(
        measured_points[-shot_config.gap_recovery.gap_predict_tail_points :]
    )
    if local_fit is not None:
        predicted_y = local_fit.evaluate_y(reacquisition.x)
        if abs(reacquisition.y - predicted_y) > shot_config.max_parabola_deviation_px:
            return []

    predicted_frames: list[SparseBallDetection] = []
    for frame_index in range(last_measured.frame_index + 1, reacquisition.frame_index):
        point = predict_gap_point(
            measured_points=measured_points,
            frame_index=frame_index,
            timestamp_ms=_estimate_timestamp(measured_points, frame_index),
            hoop_lock=hoop_lock,
            shot_config=shot_config,
        )
        if point is None:
            return []
        predicted_frames.append(point)
    return predicted_frames


def append_continuity_point(candidate: ShotCandidate, point: SparseBallDetection) -> None:
    existing = {entry.frame_index for entry in candidate.continuity_points}
    if point.frame_index in existing:
        return
    candidate.continuity_points.append(point)
    candidate.continuity_points.sort(key=lambda entry: entry.frame_index)
    if point.source == "gap_predicted":
        if point.frame_index not in candidate.gap_predicted_frames:
            candidate.gap_predicted_frames.append(point.frame_index)


def continuity_track(candidate: ShotCandidate) -> list[SparseBallDetection]:
    if candidate.continuity_points:
        return candidate.continuity_points
    return candidate.candidate_points


def _estimate_timestamp(measured_points: list[SparseBallDetection], frame_index: int) -> float:
    if not measured_points:
        return 0.0
    latest = measured_points[-1]
    if len(measured_points) >= 2:
        previous = measured_points[-2]
        dt_frames = max(latest.frame_index - previous.frame_index, 1)
        dt_ms = latest.timestamp_ms - previous.timestamp_ms
        ms_per_frame = dt_ms / dt_frames
    else:
        ms_per_frame = 33.0
    return latest.timestamp_ms + (frame_index - latest.frame_index) * ms_per_frame


def _local_parabola_fit(tail: list[SparseBallDetection]) -> ParabolaFit | None:
    if len(tail) < 3:
        return None
    result = fit_weighted_parabola(tail)
    if result is None:
        return None
    fit, _diagnostics = result
    return fit
