from swishsync_cv.config import HoopLockConfig, ShotCandidateConfig
from swishsync_cv.data import HoopLock, ShotCandidate, SparseBallDetection
from swishsync_cv.tracking.parabola import (
    fit_weighted_parabola_robust,
    is_floor_bounce_point,
    select_flight_fit_points,
)
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


def test_long_gap_finalizes_before_bounce_cluster():
    manager = ShotCandidateManager(ShotCandidateConfig(), frame_height=1920)
    manager.active = ShotCandidate(start_frame=6, state="collecting_shot")
    manager.active.candidate_points = [
        SparseBallDetection(6, 200.0, 786.0, 266.0, 0.66),
        SparseBallDetection(7, 233.0, 761.0, 255.0, 0.55),
    ]
    bounce = SparseBallDetection(31, 1033.0, 472.0, 510.0, 0.30)

    finalized = manager.update(31, bounce, _hoop())

    assert finalized is not None
    assert finalized.end_frame == 7
    assert len(finalized.candidate_points) == 2
    assert manager.lifecycle_state == "idle"


def test_floor_bounce_point_excluded_after_post_rim():
    manager = ShotCandidateManager(
        ShotCandidateConfig(floor_below_rim_margin_px=100.0),
        frame_height=1920,
    )
    hoop = _hoop()
    manager.active = ShotCandidate(start_frame=90, state="collecting_shot", post_rim_started=True)
    manager.active.candidate_points = [
        SparseBallDetection(94, 0.0, 700.0, 145.0, 0.9),
        SparseBallDetection(111, 0.0, 490.0, 437.0, 0.4),
    ]
    floor = SparseBallDetection(120, 0.0, 542.0, 761.0, 0.45)

    manager.update(120, floor, hoop)

    assert len(manager.active.candidate_points) == 2
    assert len(manager.active.excluded_debug_points) == 1
    assert manager.active.excluded_debug_points[0].frame_index == 120


def test_select_flight_fit_points_excludes_floor():
    hoop = _hoop()
    points = [
        SparseBallDetection(6, 0.0, 786.0, 266.0, 0.66),
        SparseBallDetection(7, 0.0, 761.0, 255.0, 0.55),
        SparseBallDetection(37, 0.0, 528.0, 827.0, 0.82),
    ]
    fit_points, excluded = select_flight_fit_points(points, hoop, floor_margin_px=100.0)

    assert [point.frame_index for point in fit_points] == [6, 7]
    assert [point.frame_index for point in excluded] == [37]


def test_robust_fit_refit_lowers_rmse():
    points = [
        SparseBallDetection(0, 0.0, 100.0, 300.0, 0.9),
        SparseBallDetection(1, 33.3, 150.0, 220.0, 0.9),
        SparseBallDetection(2, 66.6, 200.0, 180.0, 0.9),
        SparseBallDetection(3, 99.9, 250.0, 170.0, 0.9),
        SparseBallDetection(4, 133.2, 900.0, 900.0, 0.9),
    ]
    result = fit_weighted_parabola_robust(points)
    assert result is not None
    _, diagnostics = result
    assert diagnostics.initial_weighted_residual_rmse is not None
    assert diagnostics.weighted_residual_rmse <= diagnostics.initial_weighted_residual_rmse


def test_conditional_rim_anchor_skips_when_worse():
    candidate = ShotCandidate(start_frame=0, state="collecting_shot", end_frame=4)
    candidate.candidate_points = [
        SparseBallDetection(0, 0.0, 786.0, 266.0, 0.66),
        SparseBallDetection(1, 33.0, 761.0, 255.0, 0.55),
        SparseBallDetection(2, 66.0, 700.0, 280.0, 0.70),
        SparseBallDetection(3, 99.0, 650.0, 320.0, 0.65),
        SparseBallDetection(4, 132.0, 600.0, 360.0, 0.60),
    ]
    finalized = finalize_shot(
        candidate,
        ShotCandidateConfig(),
        hoop_lock=_hoop(),
        hoop_lock_config=HoopLockConfig(rim_anchor_weight=2.0),
    )
    assert finalized.parabola_fit is not None
    assert finalized.fit_diagnostics is not None


def test_is_floor_bounce_point():
    hoop = _hoop()
    assert is_floor_bounce_point(SparseBallDetection(0, 0.0, 500.0, 520.0, 0.5), hoop, 100.0)
    assert not is_floor_bounce_point(SparseBallDetection(0, 0.0, 500.0, 430.0, 0.5), hoop, 100.0)
