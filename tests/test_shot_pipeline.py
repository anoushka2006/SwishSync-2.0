from swishsync_cv.config import HoopLockConfig, ShotCandidateConfig, SparseDetectionConfig
from swishsync_cv.data import DetectionRecord, SparseBallDetection
from swishsync_cv.tracking.hoop_lock import HoopLockTracker
from swishsync_cv.tracking.parabola import fit_parabola
from swishsync_cv.tracking.shot_candidate import FinalizeCooldownState, ShotCandidateManager
from swishsync_cv.data import HoopLock, ShotCandidate
from swishsync_cv.tracking.sparse_detection import SparseBallDetectionBuffer


def _ball_detection(frame_index: int, x: float, y: float) -> DetectionRecord:
    return DetectionRecord(
        frame_index=frame_index,
        timestamp_ms=float(frame_index * 33.3),
        label="basketball",
        class_name="sports ball",
        confidence=0.9,
        bbox_xyxy=(x - 5, y - 5, x + 5, y + 5),
    )


def test_sparse_buffer_runs_detection_on_stride():
    buffer = SparseBallDetectionBuffer(SparseDetectionConfig(detection_stride=3))
    assert buffer.should_detect(0) is True
    assert buffer.should_detect(1) is False
    assert buffer.should_detect(3) is True


def test_sparse_buffer_extracts_highest_confidence_ball():
    buffer = SparseBallDetectionBuffer(SparseDetectionConfig(min_confidence=0.25))
    detections = [
        _ball_detection(0, 10, 20),
        DetectionRecord(0, 0.0, "basketball", "sports ball", 0.95, (30, 30, 40, 40)),
    ]
    point = buffer.extract_ball_detection(0, 0.0, detections)
    assert point is not None
    assert point.x == 35.0
    assert point.y == 35.0


def test_parabola_fit_returns_coefficients_for_arc():
    points = [
        SparseBallDetection(0, 0.0, 10.0, 50.0, 0.9),
        SparseBallDetection(1, 33.3, 20.0, 40.0, 0.9),
        SparseBallDetection(2, 66.6, 30.0, 35.0, 0.9),
        SparseBallDetection(3, 99.9, 40.0, 40.0, 0.9),
    ]
    fit = fit_parabola(points)
    assert fit is not None
    assert fit.r_squared > 0.5
    assert len(fit.sample_arc()) >= 2


def test_shot_manager_collects_without_fitting():
    manager = ShotCandidateManager(ShotCandidateConfig(), frame_height=100)
    points = [
        SparseBallDetection(0, 0.0, 20.0, 80.0, 0.9),
        SparseBallDetection(1, 33.3, 22.0, 70.0, 0.9),
        SparseBallDetection(2, 66.6, 24.0, 52.0, 0.9),
        SparseBallDetection(3, 99.9, 26.0, 45.0, 0.9),
    ]
    for point in points:
        manager.update(point.frame_index, point)
    assert manager.lifecycle_state == "collecting_shot"
    assert manager.candidate_point_count >= 3
    assert manager.active is not None
    assert manager.active.parabola_fit is None


def test_sparse_buffer_dense_mode_detects_every_frame():
    buffer = SparseBallDetectionBuffer(SparseDetectionConfig(detection_stride=3))
    assert buffer.should_detect(1, dense=True) is True
    assert buffer.should_detect(2, dense=True) is True


def test_reacquisition_suppresses_horizontal_jump_end():
    from swishsync_cv.data import HoopLock
    from swishsync_cv.tracking.shot_candidate import ShotCandidate

    manager = ShotCandidateManager(ShotCandidateConfig(), frame_height=1000)
    manager.active = ShotCandidate(start_frame=0, state="collecting_shot")
    manager.active.candidate_points = [
        SparseBallDetection(0, 0.0, 10.0, 80.0, 0.9),
        SparseBallDetection(1, 33.3, 20.0, 60.0, 0.9),
        SparseBallDetection(2, 66.6, 30.0, 45.0, 0.9),
        SparseBallDetection(3, 99.9, 40.0, 40.0, 0.9),
    ]
    reacquired = SparseBallDetection(30, 999.0, 500.0, 120.0, 0.8)
    assert manager._should_end_collection(reacquired, None) is False


def test_post_rim_requires_rim_proximity_and_measured_point():
    manager = ShotCandidateManager(ShotCandidateConfig(), frame_height=1000)
    manager.active = ShotCandidate(start_frame=0, state="collecting_shot")
    manager.active.candidate_points = [
        SparseBallDetection(0, 0.0, 10.0, 80.0, 0.9),
        SparseBallDetection(1, 33.3, 20.0, 60.0, 0.9),
        SparseBallDetection(2, 66.6, 30.0, 45.0, 0.9),
        SparseBallDetection(3, 99.9, 40.0, 40.0, 0.9),
    ]
    hoop = HoopLock(
        center_x=100.0,
        center_y=50.0,
        bbox_xyxy=(80.0, 30.0, 120.0, 70.0),
        confidence=1.0,
        locked_at_frame=0,
        is_locked=True,
    )
    too_high = SparseBallDetection(10, 333.0, 100.0, 50.0, 0.9)
    assert manager._should_end_collection(too_high, hoop) is False
    interpolated = SparseBallDetection(11, 366.0, 100.0, 80.0, 0.8, interpolated=True)
    assert manager._should_end_collection(interpolated, hoop) is False


def test_finalize_cooldown_blocks_falling_reacquisition():
    manager = ShotCandidateManager(
        ShotCandidateConfig(post_finalize_cooldown_frames=25),
        frame_height=1000,
    )
    manager._cooldown = FinalizeCooldownState(
        frame=111,
        reason="post_rim",
        last_motion="descending",
    )

    falling_points = [
        SparseBallDetection(132, 4400.0, 500.0, 300.0, 0.9),
        SparseBallDetection(134, 4466.0, 490.0, 320.0, 0.9),
        SparseBallDetection(136, 4533.0, 480.0, 350.0, 0.9),
    ]
    for point in falling_points:
        manager.update(point.frame_index, point)

    assert manager.lifecycle_state == "idle"
    assert len(manager.post_shot_debug_points) >= 2


def test_clear_new_release_can_start_after_cooldown():
    manager = ShotCandidateManager(
        ShotCandidateConfig(post_finalize_cooldown_frames=10),
        frame_height=1000,
    )
    manager._cooldown = FinalizeCooldownState(
        frame=50,
        reason="post_rim",
        last_motion="descending",
    )

    release_points = [
        SparseBallDetection(100, 3330.0, 200.0, 400.0, 0.9),
        SparseBallDetection(101, 3363.0, 205.0, 380.0, 0.9),
        SparseBallDetection(102, 3396.0, 210.0, 340.0, 0.9),
    ]
    for point in release_points:
        manager.update(point.frame_index, point)

    assert manager.lifecycle_state == "collecting_shot"


def test_cooldown_blocks_rim_zone_reacquisition():
    manager = ShotCandidateManager(
        ShotCandidateConfig(post_finalize_cooldown_frames=25),
        frame_height=1920,
    )
    manager._last_hoop_lock = HoopLock(
        center_x=500.0,
        center_y=365.0,
        bbox_xyxy=(430.0, 320.0, 570.0, 410.0),
        confidence=1.0,
        locked_at_frame=0,
        is_locked=True,
    )
    manager._cooldown = FinalizeCooldownState(
        frame=111,
        reason="post_rim",
        last_motion="descending",
    )

    reacquired = [
        SparseBallDetection(132, 4400.0, 550.0, 580.0, 0.9),
        SparseBallDetection(133, 4433.0, 556.0, 559.0, 0.9),
        SparseBallDetection(134, 4466.0, 558.0, 544.0, 0.9),
    ]
    for point in reacquired:
        manager.update(point.frame_index, point, manager._last_hoop_lock)

    assert manager.lifecycle_state == "idle"
