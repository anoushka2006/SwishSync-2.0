from swishsync_cv.config import ShotCandidateConfig, TrustedFlightConfig
from swishsync_cv.data import HoopLock, ShotCandidate, SparseBallDetection
from swishsync_cv.tracking.shot_finalization import finalize_shot
from swishsync_cv.tracking.trusted_flight import select_trusted_flight_points
from swishsync_cv.utils.serialization import shot_candidate_to_dict


def _hoop() -> HoopLock:
    # rim_center_y = bbox_xyxy[3] = 410.0 → floor bounce threshold = 510.0
    return HoopLock(
        center_x=500.0,
        center_y=365.0,
        bbox_xyxy=(430.0, 320.0, 570.0, 410.0),
        confidence=1.0,
        locked_at_frame=0,
        is_locked=True,
    )


def _cfg(max_gap: int = 10) -> TrustedFlightConfig:
    return TrustedFlightConfig(max_flight_gap_frames=max_gap)


def test_empty_candidate_points_returns_empty():
    result = select_trusted_flight_points([], None, _cfg())
    assert result.trusted == ()
    assert result.excluded == ()
    assert result.max_gap_frames == 10


def test_all_measured_contiguous_no_floor_bounces_all_trusted():
    points = [
        SparseBallDetection(i, float(i * 33), 400.0 + i * 10, 200.0, 0.8)
        for i in range(5)
    ]
    result = select_trusted_flight_points(points, None, _cfg())
    assert len(result.trusted) == 5
    assert result.excluded == ()


def test_gap_predicted_excluded_with_correct_reason():
    points = [
        SparseBallDetection(0, 0.0, 100.0, 200.0, 0.8),
        SparseBallDetection(1, 33.0, 110.0, 190.0, 0.35, source="gap_predicted"),
        SparseBallDetection(2, 66.0, 120.0, 180.0, 0.8),
    ]
    result = select_trusted_flight_points(points, None, _cfg())
    assert len(result.trusted) == 2
    assert len(result.excluded) == 1
    assert result.excluded[0].reason == "gap_predicted"
    assert result.excluded[0].point.frame_index == 1


def test_post_cluster_excluded_at_gap_greater_than_max():
    # Gap of 15 frames between frame 5 and frame 20 exceeds max_flight_gap_frames=10.
    points = [
        SparseBallDetection(0, 0.0, 100.0, 200.0, 0.8),
        SparseBallDetection(5, 165.0, 150.0, 180.0, 0.8),
        SparseBallDetection(20, 660.0, 200.0, 160.0, 0.8),
        SparseBallDetection(25, 825.0, 250.0, 155.0, 0.8),
    ]
    result = select_trusted_flight_points(points, None, _cfg())
    assert len(result.trusted) == 2
    assert len(result.excluded) == 2
    assert all(e.reason == "post_cluster" for e in result.excluded)
    assert {e.point.frame_index for e in result.excluded} == {20, 25}


def test_floor_bounce_excluded_with_correct_reason():
    hoop = _hoop()  # floor bounce threshold: y > 510.0
    points = [
        SparseBallDetection(0, 0.0, 100.0, 300.0, 0.8),   # y=300 → trusted
        SparseBallDetection(1, 33.0, 150.0, 520.0, 0.8),  # y=520 > 510 → floor_bounce
    ]
    result = select_trusted_flight_points(points, hoop, _cfg())
    assert len(result.trusted) == 1
    assert len(result.excluded) == 1
    assert result.excluded[0].reason == "floor_bounce"
    assert result.excluded[0].point.frame_index == 1


def test_tighter_gap_than_reacquisition_gap_frames():
    # max_flight_gap_frames=10 splits at gap=12; reacquisition_gap_frames=15 would not.
    points = [
        SparseBallDetection(0, 0.0, 100.0, 200.0, 0.8),
        SparseBallDetection(12, 396.0, 150.0, 180.0, 0.8),  # gap=12 > 10
    ]
    result = select_trusted_flight_points(points, None, _cfg(max_gap=10))
    assert len(result.trusted) == 1
    assert len(result.excluded) == 1
    assert result.excluded[0].reason == "post_cluster"
    assert result.excluded[0].point.frame_index == 12


def test_mixed_exclusion_reasons():
    hoop = _hoop()  # floor bounce threshold: y > 510.0
    points = [
        SparseBallDetection(0, 0.0, 100.0, 200.0, 0.8),                           # trusted
        SparseBallDetection(1, 33.0, 110.0, 190.0, 0.35, source="gap_predicted"),  # gap_predicted
        SparseBallDetection(2, 66.0, 120.0, 180.0, 0.8),                          # trusted
        SparseBallDetection(3, 99.0, 130.0, 520.0, 0.8),                          # floor_bounce
        SparseBallDetection(25, 825.0, 200.0, 160.0, 0.8),                        # post_cluster (gap=22>10)
    ]
    result = select_trusted_flight_points(points, hoop, _cfg())
    assert len(result.trusted) == 2
    assert len(result.excluded) == 3
    assert {e.reason for e in result.excluded} == {"gap_predicted", "floor_bounce", "post_cluster"}


def test_finalize_shot_sets_trusted_flight_debug():
    candidate = ShotCandidate(start_frame=0, end_frame=4, state="collecting_shot")
    candidate.candidate_points = [
        SparseBallDetection(0, 0.0, 786.0, 266.0, 0.66),
        SparseBallDetection(1, 33.0, 761.0, 255.0, 0.55),
        SparseBallDetection(2, 66.0, 700.0, 280.0, 0.70),
        SparseBallDetection(3, 99.0, 650.0, 320.0, 0.65),
        SparseBallDetection(4, 132.0, 600.0, 360.0, 0.60),
    ]
    candidate.continuity_points = list(candidate.candidate_points)
    finalized = finalize_shot(candidate, ShotCandidateConfig())
    assert finalized.trusted_flight_debug is not None
    assert isinstance(finalized.trusted_flight_debug.trusted, tuple)
    assert isinstance(finalized.trusted_flight_debug.excluded, tuple)


def test_serialization_includes_trusted_flight_block():
    candidate = ShotCandidate(start_frame=0, end_frame=4, state="collecting_shot")
    candidate.candidate_points = [
        SparseBallDetection(i, float(i * 33), 100.0 + i * 10.0, 300.0 - i * 5.0, 0.8)
        for i in range(5)
    ]
    candidate.continuity_points = list(candidate.candidate_points)
    finalized = finalize_shot(candidate, ShotCandidateConfig())
    d = shot_candidate_to_dict(finalized)
    assert "trusted_flight" in d
    tf = d["trusted_flight"]
    assert tf is not None
    assert "trusted" in tf
    assert "excluded" in tf
    assert "trusted_count" in tf
    assert "excluded_count" in tf
    assert tf["trusted_count"] == len(tf["trusted"])
    assert tf["excluded_count"] == len(tf["excluded"])
