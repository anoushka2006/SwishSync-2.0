from swishsync_cv.config import ShotCandidateConfig
from swishsync_cv.data import HoopLock, SparseBallDetection
from swishsync_cv.tracking.shot_candidate import ShotCandidateManager


def _hoop() -> HoopLock:
    return HoopLock(
        center_x=1294.5,
        center_y=491.0,
        bbox_xyxy=(1231.0, 312.0, 1358.0, 491.0),
        confidence=1.0,
        locked_at_frame=0,
        is_locked=True,
    )


def test_start_ignores_floor_band_seeds_when_hoop_locked():
    from swishsync_cv.data import ShotCandidate

    manager = ShotCandidateManager(ShotCandidateConfig(), frame_height=1080)
    hoop = _hoop()
    floor_points = [
        SparseBallDetection(16, 0.0, 340.0, 794.0, 0.30),
        SparseBallDetection(21, 0.0, 363.0, 649.0, 0.31),
        SparseBallDetection(62, 0.0, 465.0, 392.0, 0.66),
    ]
    for point in floor_points:
        manager.update(point.frame_index, point, hoop)

    assert manager.lifecycle_state == "idle"

    flight_points = [
        SparseBallDetection(63, 0.0, 511.0, 349.0, 0.57),
        SparseBallDetection(64, 0.0, 558.0, 308.0, 0.73),
    ]
    for point in flight_points:
        manager.update(point.frame_index, point, hoop)

    assert manager.lifecycle_state == "collecting_shot"
    assert manager.active is not None
    frames = [p.frame_index for p in manager.active.candidate_points]
    assert 16 not in frames
    assert 21 not in frames
    assert frames[0] == 62


def test_floor_excluded_after_interior_apex_without_post_rim():
    from swishsync_cv.data import ShotCandidate

    manager = ShotCandidateManager(ShotCandidateConfig(), frame_height=1080)
    hoop = _hoop()
    manager.active = ShotCandidate(start_frame=0, state="collecting_shot")
    manager.active.candidate_points = [
        SparseBallDetection(0, 0.0, 500.0, 300.0, 0.9),
        SparseBallDetection(1, 0.0, 520.0, 250.0, 0.9),
        SparseBallDetection(2, 0.0, 540.0, 200.0, 0.9),
        SparseBallDetection(3, 0.0, 560.0, 180.0, 0.9),
    ]
    manager.active.interior_apex_seen = True
    floor_point = SparseBallDetection(4, 0.0, 1300.0, 700.0, 0.8)
    assert manager._should_skip_floor_bounce_point(floor_point, hoop) is True
