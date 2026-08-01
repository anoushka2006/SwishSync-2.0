from swishsync_cv.config import HoopLockConfig, ShotCandidateConfig
from swishsync_cv.data import HoopLock, ShotCandidate, SparseBallDetection
from swishsync_cv.tracking.hoop_geometry import parse_hoop_bbox_arg, rim_center_from_bbox
from swishsync_cv.tracking.hoop_lock import HoopLockTracker
from swishsync_cv.tracking.parabola import RIM_ANCHOR_FRAME_INDEX, fit_weighted_parabola
from swishsync_cv.tracking.shot_finalization import finalize_shot


def test_parse_hoop_bbox_arg():
    assert parse_hoop_bbox_arg("100,200,50,30") == (100.0, 200.0, 50.0, 30.0)


def test_manual_hoop_lock_sets_locked_state():
    tracker = HoopLockTracker(HoopLockConfig())
    lock = tracker.lock_manual_bbox(0, 100.0, 50.0, 80.0, 40.0)

    assert tracker.is_locked is True
    assert tracker.is_manual_lock is True
    assert lock.is_locked is True
    assert lock.rim_center_x == 140.0
    assert lock.rim_center_y == 90.0


def test_rim_anchor_pulls_parabola_toward_hoop():
    points = [
        SparseBallDetection(0, 0.0, 100.0, 300.0, 0.9),
        SparseBallDetection(1, 33.3, 150.0, 220.0, 0.9),
        SparseBallDetection(2, 66.6, 200.0, 180.0, 0.9),
        SparseBallDetection(3, 99.9, 250.0, 170.0, 0.9),
        SparseBallDetection(4, 133.2, 300.0, 190.0, 0.9),
    ]
    rim = rim_center_from_bbox((130.0, 70.0, 170.0, 100.0))

    unanchored = fit_weighted_parabola(points)
    anchored = fit_weighted_parabola(points, rim_anchor=rim, rim_anchor_weight=8.0)

    assert unanchored is not None
    assert anchored is not None
    unanchored_fit, unanchored_diag = unanchored
    anchored_fit, anchored_diag = anchored

    rim_y_unanchored = unanchored_fit.evaluate_y(rim[0])
    rim_y_anchored = anchored_fit.evaluate_y(rim[0])
    assert abs(rim_y_anchored - rim[1]) < abs(rim_y_unanchored - rim[1])
    assert any(point.frame_index == RIM_ANCHOR_FRAME_INDEX for point in anchored_diag.points)


def _anchor_hoop() -> HoopLock:
    return HoopLock(
        center_x=80.0,
        center_y=20.0,
        bbox_xyxy=(60.0, 0.0, 100.0, 40.0),
        confidence=1.0,
        locked_at_frame=0,
        is_locked=True,
    )


def test_finalize_insufficient_below_four_measured_points():
    # 3 points is below min_measured_points_for_fit (4) → no fit
    candidate = ShotCandidate(start_frame=0, state="collecting_shot", end_frame=2)
    candidate.candidate_points = [
        SparseBallDetection(0, 0.0, 10.0, 80.0, 0.9),
        SparseBallDetection(1, 33.3, 30.0, 60.0, 0.9),
        SparseBallDetection(2, 66.6, 50.0, 45.0, 0.9),
    ]
    finalized = finalize_shot(
        candidate, ShotCandidateConfig(),
        hoop_lock=_anchor_hoop(), hoop_lock_config=HoopLockConfig(),
    )
    assert finalized.insufficient_points_for_fit is True
    assert finalized.parabola_fit is None


def test_finalize_fits_with_four_measured_points():
    # 4 clean flight points now render an arc (short shots like clips I/K)
    candidate = ShotCandidate(start_frame=0, state="collecting_shot", end_frame=3)
    candidate.candidate_points = [
        SparseBallDetection(0, 0.0, 10.0, 80.0, 0.9),
        SparseBallDetection(1, 33.3, 30.0, 60.0, 0.9),
        SparseBallDetection(2, 66.6, 50.0, 45.0, 0.9),
        SparseBallDetection(3, 99.9, 70.0, 40.0, 0.9),
    ]
    finalized = finalize_shot(
        candidate, ShotCandidateConfig(),
        hoop_lock=_anchor_hoop(), hoop_lock_config=HoopLockConfig(),
    )
    assert finalized.insufficient_points_for_fit is False
    assert finalized.parabola_fit is not None


def test_finalize_fits_with_five_points_and_locked_hoop():
    candidate = ShotCandidate(start_frame=0, state="collecting_shot", end_frame=4)
    candidate.candidate_points = [
        SparseBallDetection(0, 0.0, 10.0, 80.0, 0.9),
        SparseBallDetection(1, 33.3, 30.0, 60.0, 0.9),
        SparseBallDetection(2, 66.6, 50.0, 45.0, 0.9),
        SparseBallDetection(3, 99.9, 70.0, 40.0, 0.9),
        SparseBallDetection(4, 133.2, 90.0, 42.0, 0.9),
    ]
    hoop = HoopLock(
        center_x=120.0,
        center_y=20.0,
        bbox_xyxy=(100.0, 0.0, 140.0, 40.0),
        confidence=1.0,
        locked_at_frame=0,
        is_locked=True,
    )

    finalized = finalize_shot(
        candidate,
        ShotCandidateConfig(),
        hoop_lock=hoop,
        hoop_lock_config=HoopLockConfig(),
    )

    assert finalized.parabola_fit is not None
    assert finalized.insufficient_points_for_fit is False
