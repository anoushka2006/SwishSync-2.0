import cv2
import numpy as np

from swishsync_cv.config import HoopLockConfig
from swishsync_cv.data import DetectionRecord
from swishsync_cv.detection.hoop_detector import HybridHoopDetector
from swishsync_cv.tracking.hoop_lock import HoopLockTracker


def _hoop_detection(frame_index: int) -> DetectionRecord:
    return DetectionRecord(
        frame_index=frame_index,
        timestamp_ms=float(frame_index * 33.3),
        label="hoop",
        class_name="rim",
        confidence=0.85,
        bbox_xyxy=(50.0, 10.0, 80.0, 35.0),
    )


def _blank_frame(width: int = 96, height: int = 64) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _orange_hoop_frame(width: int = 160, height: int = 120) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (60, 20), (110, 40), (0, 140, 255), -1)
    return frame


def test_hybrid_detector_finds_orange_region_without_yolo():
    detector = HybridHoopDetector(HoopLockConfig())
    candidates = detector.detect(_orange_hoop_frame(), [])
    assert candidates
    assert candidates[0].source in {"color", "hybrid"}
    assert candidates[0].color_score > 0.2


def test_hybrid_detector_boosts_yolo_with_color_overlap():
    detector = HybridHoopDetector(HoopLockConfig())
    frame = _orange_hoop_frame()
    yolo = [
        DetectionRecord(
            0,
            0.0,
            "hoop",
            "rim",
            0.75,
            (58.0, 18.0, 112.0, 42.0),
        )
    ]
    candidates = detector.detect(frame, yolo)
    assert candidates
    assert candidates[0].source == "hybrid"
    assert candidates[0].composite_score >= 0.45


def test_hoop_lock_acquires_after_stable_observations():
    tracker = HoopLockTracker(
        HoopLockConfig(
            acquisition_frames=30,
            min_acquisition_observations=3,
            lock_confidence=0.40,
        )
    )
    frame = _orange_hoop_frame()
    hoop = _hoop_detection(0)

    lock = None
    for frame_index in range(4):
        lock = tracker.update(frame_index, frame, [hoop])

    assert lock is not None
    assert tracker.is_locked is True
    assert lock.is_locked is True
    assert lock.confidence > 0.0


def test_hoop_lock_stays_stable_when_locked():
    tracker = HoopLockTracker(HoopLockConfig(min_acquisition_observations=2, lock_confidence=0.35))
    frame = _orange_hoop_frame()
    hoop = _hoop_detection(0)

    for frame_index in range(3):
        tracker.update(frame_index, frame, [hoop])

    locked_center = tracker.lock.center_x if tracker.lock else 0.0
    jittery = DetectionRecord(10, 0.0, "hoop", "rim", 0.85, (120.0, 80.0, 150.0, 110.0))
    tracker.update(10, frame, [jittery])

    assert tracker.lock is not None
    assert abs(tracker.lock.center_x - locked_center) < 20.0


def test_hoop_lock_revalidation_keeps_anchor_without_matching_detection():
    tracker = HoopLockTracker(HoopLockConfig(min_acquisition_observations=2, lock_confidence=0.35))
    frame = _orange_hoop_frame()
    hoop = _hoop_detection(0)

    for frame_index in range(3):
        tracker.update(frame_index, frame, [hoop])

    previous = tracker.lock
    tracker.update(20, _blank_frame(), [])

    assert tracker.lock is not None
    assert tracker.lock.center_x == previous.center_x
    assert tracker.lock.center_y == previous.center_y
