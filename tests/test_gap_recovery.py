from swishsync_cv.config import GapRecoveryConfig, ShotCandidateConfig
from swishsync_cv.data import HoopLock, ShotCandidate, SparseBallDetection, effective_point_source
from swishsync_cv.tracking.gap_recovery import (
    append_continuity_point,
    backfill_gap_points,
    can_predict_gap,
    continuity_track,
    predict_gap_point,
)
from swishsync_cv.tracking.parabola import select_flight_fit_points
from swishsync_cv.tracking.shot_candidate import ShotCandidateManager
from swishsync_cv.tracking.shot_finalization import finalize_shot


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
    ]


def test_backfill_inserts_predicted_frames_without_touching_candidate_points():
    config = ShotCandidateConfig(gap_recovery=GapRecoveryConfig(max_short_gap_fill_frames=6))
    measured = _ascending_points()
    reacquisition = SparseBallDetection(68, 231.0, 1155.0, 240.0, 0.7)

    predicted = backfill_gap_points(measured, reacquisition, _hoop(), config)

    assert [point.frame_index for point in predicted] == [66, 67]
    assert all(point.source == "gap_predicted" for point in predicted)
    assert len(measured) == 4


def test_backfill_on_measured_reacquisition_via_manager():
    manager = ShotCandidateManager(ShotCandidateConfig(), frame_height=1920)
    manager.active = ShotCandidate(start_frame=62, state="collecting_shot")
    manager.active.candidate_points = _ascending_points()
    manager.active.continuity_points = list(_ascending_points())

    reacquisition = SparseBallDetection(68, 231.0, 1155.0, 240.0, 0.7)
    manager.update(68, reacquisition, _hoop())

    assert [point.frame_index for point in manager.active.candidate_points] == [
        62,
        63,
        64,
        65,
        68,
    ]
    continuity_frames = [point.frame_index for point in continuity_track(manager.active)]
    assert 66 in continuity_frames
    assert 67 in continuity_frames
    assert len(manager.active.candidate_points) == 5


def test_long_gap_does_not_predict():
    config = ShotCandidateConfig(gap_recovery=GapRecoveryConfig(max_short_gap_fill_frames=6))
    measured = _ascending_points()

    assert (
        can_predict_gap(
            measured,
            measured[-1].frame_index + 7,
            _hoop(),
            config,
            post_rim_started=False,
        )
        is False
    )


def test_gap_predicted_excluded_from_fit_selection():
    measured = _ascending_points()
    fit_points, excluded = select_flight_fit_points(measured, _hoop())
    assert fit_points == measured
    assert excluded == []


def test_finalize_fit_uses_measured_candidate_points_only():
    candidate = ShotCandidate(start_frame=62, state="collecting_shot")
    candidate.candidate_points = _ascending_points() + [
        SparseBallDetection(68, 231.0, 1155.0, 240.0, 0.7),
    ]
    for frame_index in (65, 66, 67):
        append_continuity_point(
            candidate,
            SparseBallDetection(
                frame_index,
                float(frame_index * 33),
                1165.0 - frame_index,
                260.0,
                0.35,
                source="gap_predicted",
            ),
        )
    for point in candidate.candidate_points:
        append_continuity_point(candidate, point)

    finalized = finalize_shot(candidate, ShotCandidateConfig(), hoop_lock=_hoop())

    assert finalized.parabola_fit is not None
    assert finalized.confidence is not None
    assert finalized.confidence.gap_predicted_count == 3
    assert finalized.confidence.continuity_coverage < 1.0
    assert len(finalized.candidate_points) == 5


def test_reacquisition_deviation_blocks_backfill():
    config = ShotCandidateConfig(
        gap_recovery=GapRecoveryConfig(max_short_gap_fill_frames=6),
        max_parabola_deviation_px=20.0,
    )
    measured = _ascending_points()
    bad_reacquisition = SparseBallDetection(67, 198.0, 900.0, 700.0, 0.7)

    predicted = backfill_gap_points(measured, bad_reacquisition, _hoop(), config)

    assert predicted == []


def test_post_rim_started_blocks_gap_prediction():
    config = ShotCandidateConfig()
    measured = _ascending_points()

    assert (
        can_predict_gap(
            measured,
            measured[-1].frame_index + 2,
            _hoop(),
            config,
            post_rim_started=True,
        )
        is False
    )


def test_effective_point_source_backward_compat():
    sparse_linear = SparseBallDetection(1, 0.0, 10.0, 10.0, 0.5, interpolated=True)
    measured = SparseBallDetection(2, 0.0, 10.0, 10.0, 0.5)
    predicted = SparseBallDetection(
        3,
        0.0,
        10.0,
        10.0,
        0.35,
        source="gap_predicted",
    )

    assert effective_point_source(sparse_linear) == "sparse_linear"
    assert effective_point_source(measured) == "measured"
    assert effective_point_source(predicted) == "gap_predicted"


def test_predict_gap_point_returns_none_with_too_few_measured():
    config = ShotCandidateConfig()
    measured = _ascending_points()[:2]

    assert (
        predict_gap_point(
            measured,
            measured[-1].frame_index + 1,
            100.0,
            _hoop(),
            config,
        )
        is None
    )
