from swishsync_cv.config import ShotCandidateConfig, SparseDetectionConfig
from swishsync_cv.data import HoopLock, SparseBallDetection
from swishsync_cv.tracking.shot_candidate import ShotCandidateManager
from swishsync_cv.tracking.sparse_detection import SparseBallDetectionBuffer


def test_should_detect_idle_runs_every_frame():
    buffer = SparseBallDetectionBuffer(SparseDetectionConfig(detection_stride=3))
    assert buffer.should_detect(0) is True
    assert buffer.should_detect(1) is False
    assert buffer.should_detect(1, idle=True) is True
    assert buffer.should_detect(2, idle=True) is True
    assert buffer.should_detect(2, dense=True) is True


def test_hoop_relative_gate_allows_side_angle_release():
    manager = ShotCandidateManager(
        ShotCandidateConfig(start_below_rim_margin_px=350.0),
        frame_height=1080,
    )
    manager._last_hoop_lock = HoopLock(
        center_x=440.0,
        center_y=284.5,
        bbox_xyxy=(380.0, 221.0, 500.0, 348.0),
        confidence=1.0,
        locked_at_frame=0,
        is_locked=True,
    )
    point = SparseBallDetection(10, 333.0, 500.0, 650.0, 0.9)
    assert manager._ball_in_valid_start_zone(point) is True


def test_hoop_relative_gate_blocks_floor_band():
    manager = ShotCandidateManager(
        ShotCandidateConfig(start_below_rim_margin_px=350.0),
        frame_height=1080,
    )
    manager._last_hoop_lock = HoopLock(
        center_x=440.0,
        center_y=284.5,
        bbox_xyxy=(380.0, 221.0, 500.0, 348.0),
        confidence=1.0,
        locked_at_frame=0,
        is_locked=True,
    )
    point = SparseBallDetection(10, 333.0, 500.0, 900.0, 0.9)
    assert manager._ball_in_valid_start_zone(point) is False


def test_unlocked_hoop_uses_upper_body_fallback():
    manager = ShotCandidateManager(
        ShotCandidateConfig(upper_body_y_ratio=0.55),
        frame_height=1080,
    )
    blocked = SparseBallDetection(10, 333.0, 500.0, 600.0, 0.9)
    assert manager._ball_in_valid_start_zone(blocked) is False

    small_frame = ShotCandidateManager(
        ShotCandidateConfig(upper_body_y_ratio=0.55),
        frame_height=1000,
    )
    allowed = SparseBallDetection(10, 333.0, 500.0, 500.0, 0.9)
    assert small_frame._ball_in_valid_start_zone(allowed) is True
