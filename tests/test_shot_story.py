import json

import numpy as np

from swishsync_cv.config import ShotCandidateConfig, ShotStoryConfig, VideoOutputConfig
from swishsync_cv.data import HoopLock, ShotCandidate, ShotStoryMetadata, SparseBallDetection
from swishsync_cv.tracking.shot_candidate import ShotCandidateManager
from swishsync_cv.tracking.shot_finalization import finalize_shot
from swishsync_cv.tracking.parabola import is_floor_bounce_point
from swishsync_cv.tracking.shot_story import _should_show_pickup, compute_shot_story
from swishsync_cv.utils.serialization import shot_candidate_to_dict, write_finalized_shots_json
from swishsync_cv.visualization.shot_story_drawing import (
    MEASURED_COLOR,
    PICKUP_CONNECTOR_COLOR,
    draw_shot_story,
    pickup_connector_eligible,
)
from swishsync_cv.visualization.trajectory_panel import render_trajectory_panel


def _hoop() -> HoopLock:
    return HoopLock(
        center_x=500.0,
        center_y=365.0,
        bbox_xyxy=(430.0, 320.0, 570.0, 410.0),
        confidence=1.0,
        locked_at_frame=0,
        is_locked=True,
    )


def _ascending_points(start_frame: int = 62) -> list[SparseBallDetection]:
    return [
        SparseBallDetection(start_frame + 0, 0.0, 1200.0, 400.0, 0.8),
        SparseBallDetection(start_frame + 1, 33.0, 1190.0, 360.0, 0.8),
        SparseBallDetection(start_frame + 2, 66.0, 1180.0, 320.0, 0.8),
        SparseBallDetection(start_frame + 3, 99.0, 1170.0, 285.0, 0.8),
        SparseBallDetection(start_frame + 6, 231.0, 1155.0, 240.0, 0.7),
    ]


def _finalize_with_story(
    candidate: ShotCandidate,
    fit_points: list[SparseBallDetection] | None = None,
) -> ShotCandidate:
    return finalize_shot(
        candidate,
        ShotCandidateConfig(),
        hoop_lock=_hoop(),
        story_config=ShotStoryConfig(),
    )


def test_compute_shot_story_clip_d_boundaries():
    measured = [
        SparseBallDetection(62, 0.0, 1200.0, 400.0, 0.8),
        SparseBallDetection(63, 33.0, 1190.0, 360.0, 0.8),
        SparseBallDetection(64, 66.0, 1180.0, 320.0, 0.8),
        SparseBallDetection(67, 165.0, 1160.0, 270.0, 0.8),
        SparseBallDetection(68, 198.0, 1155.0, 240.0, 0.7),
    ]
    fit_points = measured[:4]
    continuity = list(measured[:3])
    continuity.extend(
        [
            SparseBallDetection(65, 99.0, 1175.0, 295.0, 0.35, source="gap_predicted"),
            SparseBallDetection(66, 132.0, 1170.0, 280.0, 0.35, source="gap_predicted"),
        ]
    )
    continuity.extend(measured[3:])

    candidate = ShotCandidate(start_frame=62, end_frame=68, state="shot_finalized")
    candidate.candidate_points = measured
    candidate.continuity_points = continuity
    candidate.gap_predicted_frames = [65, 66]

    story = compute_shot_story(
        candidate,
        ShotCandidateConfig(),
        _hoop(),
        fit_points,
        ShotStoryConfig(),
    )

    assert story.release_frame == 62
    assert story.flight_start_frame == 62
    assert story.flight_end_frame == 67
    assert story.gap_predicted_frames == (65, 66)
    assert story.show_pickup is False


def test_pickup_requires_min_span_and_points():
    pickup = [
        SparseBallDetection(55, 0.0, 300.0, 280.0, 0.8),
        SparseBallDetection(58, 99.0, 305.0, 275.0, 0.8),
        SparseBallDetection(61, 198.0, 310.0, 270.0, 0.8),
    ]
    measured = _ascending_points()[:5]
    candidate = ShotCandidate(start_frame=62, end_frame=66, state="shot_finalized")
    candidate.pickup_points = pickup
    candidate.candidate_points = measured

    story = compute_shot_story(
        candidate,
        ShotCandidateConfig(),
        _hoop(),
        measured[:4],
        ShotStoryConfig(pickup_min_points=2, pickup_min_frame_span=3),
    )

    assert story.show_pickup is True
    assert story.pickup_frames == (55, 58, 61)


def test_pickup_rejected_when_horizontal_span_too_large():
    pickup = [
        SparseBallDetection(50, 0.0, 100.0, 700.0, 0.8),
        SparseBallDetection(54, 132.0, 350.0, 690.0, 0.8),
    ]
    measured = _ascending_points()[:5]
    candidate = ShotCandidate(start_frame=62, end_frame=66, state="shot_finalized")
    candidate.pickup_points = pickup
    candidate.candidate_points = measured

    story = compute_shot_story(
        candidate,
        ShotCandidateConfig(),
        _hoop(),
        measured[:4],
        ShotStoryConfig(pickup_max_horizontal_span_px=200.0),
    )

    assert story.show_pickup is False
    assert story.pickup_frames == tuple()


def test_post_shot_hidden_across_large_gap():
    post_points = [
        SparseBallDetection(80, 0.0, 500.0, 360.0, 0.7),
        SparseBallDetection(150, 0.0, 510.0, 370.0, 0.7),
    ]
    measured = _ascending_points()[:5]
    candidate = ShotCandidate(start_frame=62, end_frame=68, state="shot_finalized")
    candidate.candidate_points = measured
    candidate.post_shot_points = post_points

    story = compute_shot_story(
        candidate,
        ShotCandidateConfig(reacquisition_gap_frames=32),
        _hoop(),
        measured[:4],
        ShotStoryConfig(),
    )

    assert story.show_post_shot is False


def test_post_shot_hidden_after_flight_gap():
    post_points = [
        SparseBallDetection(81, 0.0, 500.0, 360.0, 0.7),
        SparseBallDetection(83, 0.0, 505.0, 365.0, 0.7),
    ]
    measured = _ascending_points()[:5]
    candidate = ShotCandidate(start_frame=62, end_frame=68, state="shot_finalized")
    candidate.candidate_points = measured
    candidate.post_shot_points = post_points

    story = compute_shot_story(
        candidate,
        ShotCandidateConfig(reacquisition_gap_frames=15),
        _hoop(),
        measured[:4],
        ShotStoryConfig(),
    )

    assert story.show_post_shot is False


def test_post_shot_shown_for_tight_sequence():
    post_points = [
        SparseBallDetection(70, 0.0, 500.0, 360.0, 0.7),
        SparseBallDetection(72, 0.0, 505.0, 365.0, 0.7),
        SparseBallDetection(74, 0.0, 510.0, 370.0, 0.7),
    ]
    measured = _ascending_points()[:5]
    candidate = ShotCandidate(start_frame=62, end_frame=68, state="shot_finalized")
    candidate.candidate_points = measured
    candidate.post_shot_points = post_points

    story = compute_shot_story(
        candidate,
        ShotCandidateConfig(reacquisition_gap_frames=32),
        _hoop(),
        measured[:4],
        ShotStoryConfig(),
    )

    assert story.show_post_shot is True
    assert story.post_shot_start_frame == 70


def test_manager_snapshots_pickup_before_release():
    manager = ShotCandidateManager(ShotCandidateConfig(), frame_height=1920)
    manager._last_hoop_lock = _hoop()
    manager._pre_shot_buffer = [
        SparseBallDetection(58, 0.0, 300.0, 420.0, 0.8),
        SparseBallDetection(59, 33.0, 305.0, 415.0, 0.8),
        SparseBallDetection(60, 66.0, 310.0, 410.0, 0.8),
        SparseBallDetection(61, 99.0, 315.0, 405.0, 0.8),
        SparseBallDetection(62, 132.0, 1200.0, 400.0, 0.8),
        SparseBallDetection(63, 165.0, 1190.0, 360.0, 0.8),
        SparseBallDetection(64, 198.0, 1180.0, 320.0, 0.8),
    ]
    manager._try_start_collection()

    assert manager.active is not None
    assert [point.frame_index for point in manager.active.pickup_points] == [58, 59, 60, 61]
    assert manager.active.start_frame == 62


def test_preview_pickup_points_shown_during_idle_release_gap():
    manager = ShotCandidateManager(ShotCandidateConfig(), frame_height=1920)
    manager._last_hoop_lock = _hoop()
    manager._pre_shot_buffer = [
        SparseBallDetection(59, 0.0, 1033.0, 450.0, 0.8),
        SparseBallDetection(60, 33.0, 1016.0, 425.0, 0.8),
        SparseBallDetection(61, 66.0, 1000.0, 403.0, 0.8),
        SparseBallDetection(62, 99.0, 982.0, 381.0, 0.8),
    ]

    preview = manager.preview_pickup_points()

    assert [point.frame_index for point in preview] == [59, 60, 61]
    assert manager.active is None


def test_finalize_attaches_story_and_post_shot_points():
    candidate = ShotCandidate(start_frame=62, end_frame=68, state="collecting_shot")
    candidate.candidate_points = _ascending_points()
    candidate.continuity_points = list(candidate.candidate_points)
    candidate.post_shot_points = [
        SparseBallDetection(70, 0.0, 500.0, 360.0, 0.7),
        SparseBallDetection(72, 0.0, 505.0, 365.0, 0.7),
    ]

    finalized = _finalize_with_story(candidate)

    assert finalized.story is not None
    assert isinstance(finalized.story, ShotStoryMetadata)
    assert finalized.story.release_frame == 62
    assert len(finalized.post_shot_points) == 2


def test_clip_c_like_pickup_is_shown():
    hoop = HoopLock(
        center_x=432.5,
        center_y=319.5,
        bbox_xyxy=(371.5, 233.0, 493.5, 406.0),
        confidence=1.0,
        locked_at_frame=0,
        is_locked=True,
    )
    pickup = [
        SparseBallDetection(59, 0.0, 1033.12, 449.91, 0.8),
        SparseBallDetection(60, 0.0, 1016.35, 425.65, 0.8),
        SparseBallDetection(61, 0.0, 1000.26, 403.29, 0.8),
    ]
    measured = [SparseBallDetection(62, 0.0, 982.03, 380.81, 0.8)]
    candidate = ShotCandidate(start_frame=62, end_frame=77, state="shot_finalized")
    candidate.pickup_points = pickup
    candidate.candidate_points = measured

    cfg = ShotCandidateConfig()
    story_cfg = ShotStoryConfig()
    assert not all(
        is_floor_bounce_point(point, hoop, cfg.floor_below_rim_margin_px)
        for point in pickup
    )
    assert _should_show_pickup(pickup, hoop, cfg, story_cfg) is True

    story = compute_shot_story(
        candidate,
        cfg,
        hoop,
        measured,
        story_cfg,
    )

    assert story.show_pickup is True
    assert story.pickup_frames == (59, 60, 61)


def test_serialization_exports_story_and_render_buffers(tmp_path):
    candidate = ShotCandidate(start_frame=62, end_frame=68, state="shot_finalized")
    candidate.candidate_points = _ascending_points()
    candidate.pickup_points = [
        SparseBallDetection(58, 0.0, 300.0, 700.0, 0.8),
        SparseBallDetection(59, 33.0, 305.0, 695.0, 0.8),
    ]
    candidate.post_shot_points = [
        SparseBallDetection(70, 0.0, 500.0, 360.0, 0.7),
        SparseBallDetection(72, 0.0, 505.0, 365.0, 0.7),
    ]
    candidate.story = compute_shot_story(
        candidate,
        ShotCandidateConfig(),
        _hoop(),
        candidate.candidate_points[:4],
        ShotStoryConfig(),
    )

    payload = shot_candidate_to_dict(candidate)
    assert payload["story"] is not None
    assert payload["story"]["release_frame"] == 62
    assert len(payload["pickup_points"]) == 2
    assert len(payload["post_shot_points"]) == 2

    output_path = tmp_path / "shots.json"
    write_finalized_shots_json(output_path, [candidate])
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported[0]["story"]["flight_start_frame"] == 62


def test_pickup_connector_eligible_when_pickup_and_release_are_close():
    config = ShotStoryConfig(
        max_pickup_to_release_frame_gap=6,
        max_pickup_to_release_distance_px=90.0,
        min_pickup_connector_distance_px=5.0,
    )
    last_pickup = SparseBallDetection(61, 0.0, 300.0, 280.0, 0.8)
    first_flight = SparseBallDetection(62, 33.0, 320.0, 260.0, 0.8)

    assert pickup_connector_eligible(last_pickup, first_flight, config) is True


def test_pickup_connector_rejected_when_gap_or_distance_too_large():
    config = ShotStoryConfig(
        max_pickup_to_release_frame_gap=3,
        max_pickup_to_release_distance_px=40.0,
        min_pickup_connector_distance_px=5.0,
    )
    last_pickup = SparseBallDetection(50, 0.0, 100.0, 700.0, 0.8)
    far_flight = SparseBallDetection(62, 396.0, 350.0, 690.0, 0.8)
    late_flight = SparseBallDetection(55, 165.0, 310.0, 270.0, 0.8)

    assert pickup_connector_eligible(last_pickup, far_flight, config) is False
    assert pickup_connector_eligible(last_pickup, late_flight, config) is False


def test_pickup_connector_draws_for_close_pickup_and_flight():
    pickup = [
        SparseBallDetection(59, 0.0, 1140.0, 450.0, 0.8),
        SparseBallDetection(60, 33.0, 1155.0, 430.0, 0.8),
        SparseBallDetection(61, 66.0, 1170.0, 410.0, 0.8),
    ]
    measured = [
        SparseBallDetection(62, 99.0, 1200.0, 400.0, 0.8),
        SparseBallDetection(63, 132.0, 1190.0, 360.0, 0.8),
        SparseBallDetection(64, 165.0, 1180.0, 320.0, 0.8),
        SparseBallDetection(65, 198.0, 1170.0, 285.0, 0.8),
        SparseBallDetection(66, 231.0, 1155.0, 240.0, 0.7),
    ]
    candidate = ShotCandidate(start_frame=62, end_frame=66, state="shot_finalized")
    candidate.pickup_points = pickup
    candidate.candidate_points = measured
    candidate.continuity_points = list(measured)
    candidate = finalize_shot(
        candidate,
        ShotCandidateConfig(),
        hoop_lock=_hoop(),
        story_config=ShotStoryConfig(
            pickup_min_points=2,
            pickup_min_frame_span=2,
            max_pickup_to_release_frame_gap=6,
            max_pickup_to_release_distance_px=90.0,
        ),
    )
    assert candidate.story is not None
    assert candidate.story.show_pickup is True

    before = shot_candidate_to_dict(candidate)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_shot_story(frame, candidate, story_config=ShotStoryConfig())
    after = shot_candidate_to_dict(candidate)
    assert before == after

    connector = np.array(
        tuple(int(channel * 0.55) for channel in PICKUP_CONNECTOR_COLOR),
        dtype=np.int16,
    )
    last_pickup = pickup[-1]
    first_flight = measured[0]
    mid_x = int(round((last_pickup.x + first_flight.x) / 2))
    mid_y = int(round((last_pickup.y + first_flight.y) / 2))
    patch = frame[mid_y - 2 : mid_y + 3, mid_x - 2 : mid_x + 3].reshape(-1, 3)
    assert np.any(np.all(np.abs(patch.astype(np.int16) - connector) <= 12, axis=1))


def test_pickup_connector_not_drawn_when_pickup_and_flight_disconnected():
    pickup = [
        SparseBallDetection(40, 0.0, 100.0, 700.0, 0.8),
        SparseBallDetection(44, 132.0, 350.0, 690.0, 0.8),
    ]
    measured = _ascending_points(start_frame=62)
    candidate = ShotCandidate(start_frame=62, end_frame=66, state="shot_finalized")
    candidate.pickup_points = pickup
    candidate.candidate_points = measured
    candidate.continuity_points = list(measured)
    candidate = finalize_shot(
        candidate,
        ShotCandidateConfig(),
        hoop_lock=_hoop(),
        story_config=ShotStoryConfig(),
    )
    assert candidate.story is not None
    assert candidate.story.show_pickup is False

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_shot_story(frame, candidate)
    connector = np.array(
        tuple(int(channel * 0.55) for channel in PICKUP_CONNECTOR_COLOR),
        dtype=np.int16,
    )
    last_pickup = pickup[-1]
    first_flight = measured[0]
    mid_x = int(round((last_pickup.x + first_flight.x) / 2))
    mid_y = int(round((last_pickup.y + first_flight.y) / 2))
    patch = frame[mid_y - 2 : mid_y + 3, mid_x - 2 : mid_x + 3].reshape(-1, 3)
    assert not np.any(np.all(np.abs(patch.astype(np.int16) - connector) <= 12, axis=1))


def test_finalized_shot_draws_measured_continuity_with_parabola():
    candidate = ShotCandidate(start_frame=62, end_frame=68, state="shot_finalized")
    candidate.candidate_points = _ascending_points()
    candidate.continuity_points = list(candidate.candidate_points)
    candidate = _finalize_with_story(candidate)
    assert candidate.parabola_fit is not None

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_shot_story(frame, candidate)

    target = np.array(MEASURED_COLOR, dtype=np.int16)
    hits = 0
    for point in candidate.continuity_points:
        x, y = int(round(point.x)), int(round(point.y))
        patch = frame[y - 2 : y + 3, x - 2 : x + 3].reshape(-1, 3).astype(np.int16)
        if np.any(np.all(np.abs(patch - target) <= 25, axis=1)):
            hits += 1

    assert hits >= 3


def test_finalized_measured_continuity_clips_to_flight_window():
    """Post-flight bounce in continuity must not draw after finalize (clip H)."""
    measured = [
        SparseBallDetection(58, 0.0, 439.0, 361.0, 0.71),
        SparseBallDetection(59, 33.0, 481.0, 317.0, 0.84),
        SparseBallDetection(60, 66.0, 521.0, 277.0, 0.60),
        SparseBallDetection(61, 99.0, 559.0, 241.0, 0.56),
        SparseBallDetection(62, 132.0, 594.0, 210.0, 0.36),
        SparseBallDetection(63, 165.0, 628.0, 181.0, 0.27),
        SparseBallDetection(66, 231.0, 722.0, 117.0, 0.21),
    ]
    bounce = SparseBallDetection(127, 4342.0, 1428.0, 685.0, 0.32)
    candidate = ShotCandidate(start_frame=58, end_frame=127, state="shot_finalized")
    candidate.candidate_points = measured + [bounce]
    candidate.continuity_points = list(measured) + [bounce]
    candidate = _finalize_with_story(candidate)

    assert candidate.story is not None
    assert candidate.story.flight_end_frame == 66

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    draw_shot_story(frame, candidate)

    target = np.array(MEASURED_COLOR, dtype=np.int16)

    def _has_measured_dot(x: int, y: int) -> bool:
        patch = frame[y - 2 : y + 3, x - 2 : x + 3].reshape(-1, 3).astype(np.int16)
        return bool(np.any(np.all(np.abs(patch - target) <= 25, axis=1)))

    apex = measured[-1]
    assert _has_measured_dot(int(round(apex.x)), int(round(apex.y)))
    assert not _has_measured_dot(int(round(bounce.x)), int(round(bounce.y)))


def test_draw_shot_story_and_panel_do_not_crash():
    candidate = ShotCandidate(start_frame=62, end_frame=68, state="shot_finalized")
    candidate.candidate_points = _ascending_points()
    candidate.continuity_points = list(candidate.candidate_points)
    candidate = _finalize_with_story(candidate)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_shot_story(frame, candidate, show_flight_dots=True, show_legend=True)

    panel = render_trajectory_panel(
        frame_size=(160, 120),
        collecting_shot=None,
        display_shot=candidate,
        hoop_lock=_hoop(),
        lifecycle_state="idle",
        candidate_point_count=0,
        finalized_shots=[candidate],
        video_config=VideoOutputConfig(show_shot_history=True),
    )

    assert np.count_nonzero(panel) > 0


def test_trajectory_panel_renders_faded_prior_shot():
    prior = ShotCandidate(start_frame=10, end_frame=14, state="shot_finalized")
    prior.candidate_points = [
        SparseBallDetection(10, 0.0, 20.0, 80.0, 0.9),
        SparseBallDetection(11, 33.0, 30.0, 60.0, 0.9),
        SparseBallDetection(12, 66.0, 40.0, 45.0, 0.9),
        SparseBallDetection(13, 99.0, 50.0, 40.0, 0.9),
        SparseBallDetection(14, 132.0, 60.0, 42.0, 0.9),
    ]
    prior = _finalize_with_story(prior)

    current = ShotCandidate(start_frame=62, end_frame=66, state="shot_finalized")
    current.candidate_points = _ascending_points()[:5]
    current = _finalize_with_story(current)

    panel = render_trajectory_panel(
        frame_size=(160, 120),
        collecting_shot=None,
        display_shot=current,
        hoop_lock=None,
        lifecycle_state="idle",
        candidate_point_count=0,
        finalized_shots=[prior, current],
        video_config=VideoOutputConfig(show_shot_history=True),
    )

    assert panel.shape == (120, 160, 3)
    assert np.count_nonzero(panel) > 0
