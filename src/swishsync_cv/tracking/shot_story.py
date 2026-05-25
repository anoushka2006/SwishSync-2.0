"""Render-only shot lifecycle phase boundaries."""

from __future__ import annotations

from swishsync_cv.config import ShotCandidateConfig, ShotStoryConfig
from swishsync_cv.data import HoopLock, ShotCandidate, ShotStoryMetadata, SparseBallDetection, effective_point_source
from swishsync_cv.tracking.gap_recovery import continuity_track
from swishsync_cv.tracking.parabola import is_floor_bounce_point


def compute_shot_story(
    candidate: ShotCandidate,
    config: ShotCandidateConfig,
    hoop_lock: HoopLock | None,
    fit_points: list[SparseBallDetection],
    story_config: ShotStoryConfig | None = None,
) -> ShotStoryMetadata:
    """Derive render phases without changing fit inputs."""

    story_cfg = story_config or ShotStoryConfig()
    measured = list(candidate.candidate_points)
    if not measured:
        return _empty_story()

    release = measured[0]
    release_frame = release.frame_index
    flight_start = release_frame
    flight_end = (
        fit_points[-1].frame_index
        if fit_points
        else measured[-1].frame_index
    )

    pickup_frames = tuple(
        sorted({point.frame_index for point in candidate.pickup_points})
    )
    show_pickup = _should_show_pickup(
        candidate.pickup_points,
        hoop_lock,
        config,
        story_cfg,
    )

    gap_frames = tuple(
        sorted(
            point.frame_index
            for point in continuity_track(candidate)
            if effective_point_source(point) == "gap_predicted"
            and flight_start <= point.frame_index <= flight_end
        )
    )

    post_shot_start = _post_shot_start_frame(
        candidate,
        hoop_lock,
        flight_end,
    )
    show_post_shot = _should_show_post_shot(
        candidate.post_shot_points,
        post_shot_start,
        flight_end,
        config,
    )

    return ShotStoryMetadata(
        release_frame=release_frame,
        release_xy=(release.x, release.y),
        flight_start_frame=flight_start,
        flight_end_frame=flight_end,
        post_shot_start_frame=post_shot_start,
        pickup_frames=pickup_frames if show_pickup else tuple(),
        gap_predicted_frames=gap_frames,
        show_post_shot=show_post_shot,
        show_pickup=show_pickup,
    )


def _empty_story() -> ShotStoryMetadata:
    return ShotStoryMetadata(
        release_frame=0,
        release_xy=(0.0, 0.0),
        flight_start_frame=0,
        flight_end_frame=0,
        post_shot_start_frame=None,
        pickup_frames=tuple(),
        gap_predicted_frames=tuple(),
        show_post_shot=False,
        show_pickup=False,
    )


def _should_show_pickup(
    pickup_points: list[SparseBallDetection],
    hoop_lock: HoopLock | None,
    config: ShotCandidateConfig,
    story_config: ShotStoryConfig,
) -> bool:
    if len(pickup_points) < story_config.pickup_min_points:
        return False

    frames = [point.frame_index for point in pickup_points]
    if max(frames) - min(frames) + 1 < story_config.pickup_min_frame_span:
        return False

    if hoop_lock is not None and all(
        is_floor_bounce_point(point, hoop_lock, config.floor_below_rim_margin_px)
        for point in pickup_points
    ):
        return False

    xs = [point.x for point in pickup_points]
    if max(xs) - min(xs) > story_config.pickup_max_horizontal_span_px:
        return False

    if hoop_lock is not None and hoop_lock.is_locked:
        rim_y = hoop_lock.rim_center_y
        ys = [point.y for point in pickup_points]
        if all(abs(point.y - rim_y) < 80.0 for point in pickup_points):
            if max(ys) - min(ys) < 30.0:
                return False

    return True


def _post_shot_start_frame(
    candidate: ShotCandidate,
    hoop_lock: HoopLock | None,
    flight_end_frame: int,
) -> int | None:
    candidates: list[int] = []

    for point in candidate.post_shot_points:
        if point.frame_index > flight_end_frame:
            candidates.append(point.frame_index)

    for point in candidate.excluded_debug_points:
        if point.frame_index > flight_end_frame:
            candidates.append(point.frame_index)

    if hoop_lock is not None and hoop_lock.is_locked:
        for point in continuity_track(candidate):
            if point.frame_index <= flight_end_frame:
                continue
            if effective_point_source(point) != "measured":
                continue
            if point.y >= hoop_lock.rim_center_y - 60.0:
                candidates.append(point.frame_index)

    if not candidates:
        return None
    return min(candidates)


def _should_show_post_shot(
    post_shot_points: list[SparseBallDetection],
    post_shot_start_frame: int | None,
    flight_end_frame: int,
    config: ShotCandidateConfig,
) -> bool:
    if post_shot_start_frame is None or len(post_shot_points) < 2:
        return False

    if post_shot_start_frame - flight_end_frame > config.reacquisition_gap_frames:
        return False

    frames = sorted(point.frame_index for point in post_shot_points)
    for previous, current in zip(frames, frames[1:]):
        if current - previous > config.reacquisition_gap_frames:
            return False
    return True
