from swishsync_cv.config import HoopLockConfig, ShotCandidateConfig, SparseDetectionConfig
from swishsync_cv.data import DetectionRecord, SparseBallDetection
from swishsync_cv.tracking.hoop_lock import HoopLockTracker
from swishsync_cv.tracking.parabola import fit_parabola
from swishsync_cv.tracking.shot_candidate import ShotCandidateManager
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
