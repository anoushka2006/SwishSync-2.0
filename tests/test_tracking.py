from swishsync_cv.data import DetectionRecord
from swishsync_cv.tracking import BallTrajectoryTracker


def test_tracker_selects_highest_confidence_basketball_detection():
    tracker = BallTrajectoryTracker(min_confidence=0.25)
    detections = [
        DetectionRecord(0, 0.0, "basketball", "sports ball", 0.4, (0, 0, 10, 10)),
        DetectionRecord(0, 0.0, "basketball", "sports ball", 0.9, (20, 10, 40, 30)),
        DetectionRecord(0, 0.0, "hoop", "rim", 0.99, (50, 50, 80, 80)),
    ]

    point = tracker.update(frame_index=0, timestamp_ms=0.0, detections=detections)

    assert point is not None
    assert point.x == 30.0
    assert point.y == 20.0
    assert point.confidence == 0.9
    assert tracker.points == [point]


def test_tracker_returns_none_when_ball_is_missing():
    tracker = BallTrajectoryTracker(min_confidence=0.25)
    detections = [
        DetectionRecord(1, 33.3, "hoop", "rim", 0.99, (50, 50, 80, 80)),
    ]

    point = tracker.update(frame_index=1, timestamp_ms=33.3, detections=detections)

    assert point is None
    assert tracker.points == []
