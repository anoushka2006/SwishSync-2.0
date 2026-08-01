import numpy as np

from swishsync_cv.config import DetectionConfig, HoopLockConfig, ShotCandidateConfig, ShotStoryConfig, SparseDetectionConfig
from swishsync_cv.data import DetectionRecord, HoopLock, ShotCandidate, SparseBallDetection
from swishsync_cv.detection.ball_roi_search import (
    compute_roi_bounds,
    effective_roi_min_confidence,
    offset_detection_to_full_frame,
    predict_tracking_center,
    try_roi_ball_detection,
    validate_roi_candidate,
)
from swishsync_cv.tracking.shot_candidate import ShotCandidateManager
from swishsync_cv.tracking.shot_finalization import finalize_shot
from swishsync_cv.utils.serialization import shot_candidate_to_dict


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
        SparseBallDetection(start_frame + 6, 231.0, 1155.0, 240.0, 0.7),
    ]


def test_predict_tracking_center_extrapolates_from_last_two_points():
    points = [
        SparseBallDetection(10, 0.0, 100.0, 200.0, 0.8),
        SparseBallDetection(11, 33.0, 110.0, 190.0, 0.8),
    ]

    x, y = predict_tracking_center(points, frame_index=13)

    assert x == 130.0
    assert y == 170.0


def test_compute_roi_bounds_clamps_to_frame():
    bounds = compute_roi_bounds(10.0, 20.0, padding_px=96, frame_width=200, frame_height=120)

    assert bounds == (0, 0, 106, 116)


def test_offset_detection_to_full_frame():
    detection = DetectionRecord(
        frame_index=5,
        timestamp_ms=100.0,
        label="basketball",
        class_name="sports ball",
        confidence=0.5,
        bbox_xyxy=(10.0, 20.0, 30.0, 40.0),
    )

    mapped = offset_detection_to_full_frame(detection, roi_x1=100, roi_y1=50)

    assert mapped.bbox_xyxy == (110.0, 70.0, 130.0, 90.0)
    assert mapped.center == (120.0, 80.0)


def test_validate_roi_candidate_rejects_large_horizontal_jump():
    recent = [SparseBallDetection(10, 0.0, 100.0, 200.0, 0.8)]
    candidate = SparseBallDetection(11, 33.0, 300.0, 200.0, 0.8)

    assert (
        validate_roi_candidate(
            candidate,
            recent,
            ShotCandidateConfig(max_horizontal_jump_px=120.0),
            predicted_x=130.0,
            predicted_y=170.0,
            max_prediction_distance_px=90.0,
        )
        is False
    )


def test_effective_roi_min_confidence_scales_with_track_length():
    config = SparseDetectionConfig(min_confidence=0.25, roi_min_confidence=0.15)

    assert effective_roi_min_confidence(config, 2) == 0.15
    assert effective_roi_min_confidence(config, 4) == 0.20
    assert effective_roi_min_confidence(config, 8) == 0.25


def test_try_roi_ball_detection_uses_crop_detector_and_motion_gates():
    recent = [
        SparseBallDetection(10, 0.0, 100.0, 200.0, 0.8),
        SparseBallDetection(11, 33.0, 110.0, 190.0, 0.8),
    ]
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    captured: dict[str, tuple[int, int, int, int]] = {}

    def detect_crop(frame, crop_xyxy, frame_index, timestamp_ms, min_confidence):
        captured["bounds"] = crop_xyxy
        x1, y1, x2, y2 = crop_xyxy
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return [
            DetectionRecord(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                label="basketball",
                class_name="sports ball",
                confidence=0.2,
                bbox_xyxy=(
                    center_x - 5,
                    center_y - 5,
                    center_x + 5,
                    center_y + 5,
                ),
            )
        ]

    point = try_roi_ball_detection(
        frame=frame,
        frame_index=13,
        timestamp_ms=429.0,
        recent_points=recent,
        validation_points=recent,
        detect_crop=detect_crop,
        sparse_config=SparseDetectionConfig(
            roi_min_confidence=0.15,
            roi_padding_px=96,
        ),
        detection_config=DetectionConfig(),
        shot_config=ShotCandidateConfig(max_horizontal_jump_px=120.0),
    )

    assert point is not None
    assert captured["bounds"][0] >= 0
    assert point.frame_index == 13
    assert point.confidence == 0.2
    assert not point.interpolated


def test_try_roi_ball_detection_rejects_disconnected_candidate():
    recent = [
        SparseBallDetection(10, 0.0, 100.0, 200.0, 0.8),
        SparseBallDetection(11, 33.0, 110.0, 190.0, 0.8),
    ]
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    def detect_crop(frame, crop_xyxy, frame_index, timestamp_ms, min_confidence):
        return [
            DetectionRecord(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                label="basketball",
                class_name="sports ball",
                confidence=0.9,
                bbox_xyxy=(10.0, 10.0, 20.0, 20.0),
            )
        ]

    point = try_roi_ball_detection(
        frame=frame,
        frame_index=13,
        timestamp_ms=429.0,
        recent_points=recent,
        validation_points=recent,
        detect_crop=detect_crop,
        sparse_config=SparseDetectionConfig(),
        detection_config=DetectionConfig(),
        shot_config=ShotCandidateConfig(max_horizontal_jump_px=20.0),
    )

    assert point is None


def test_finalize_outputs_unchanged_by_roi_module_presence():
    from swishsync_cv.data import ShotCandidate

    candidate = ShotCandidate(start_frame=62, end_frame=68, state="shot_finalized")
    candidate.candidate_points = _ascending_points()
    candidate.continuity_points = list(candidate.candidate_points)
    candidate = finalize_shot(
        candidate,
        ShotCandidateConfig(),
        hoop_lock=_hoop(),
        story_config=ShotStoryConfig(),
    )
    before = shot_candidate_to_dict(candidate)

    assert before["parabola_fit"] is not None
    assert before["fit_diagnostics"]["weighted_residual_rmse"] is not None


def test_tracking_context_points_requires_upward_pre_shot_motion():
    manager = ShotCandidateManager(
        config=ShotCandidateConfig(),
        frame_height=720,
        hoop_lock_config=HoopLockConfig(),
    )
    flat_points = [
        SparseBallDetection(10, 0.0, 100.0, 500.0, 0.8),
        SparseBallDetection(11, 33.0, 105.0, 505.0, 0.8),
        SparseBallDetection(12, 66.0, 110.0, 510.0, 0.8),
    ]
    for point in flat_points:
        manager.update(frame_index=point.frame_index, point=point, hoop_lock=_hoop())

    assert manager.tracking_context_points() == []

    rising_points = [
        SparseBallDetection(20, 0.0, 100.0, 500.0, 0.8),
        SparseBallDetection(21, 33.0, 110.0, 480.0, 0.8),
        SparseBallDetection(22, 66.0, 120.0, 460.0, 0.8),
    ]
    manager = ShotCandidateManager(
        config=ShotCandidateConfig(),
        frame_height=720,
    )
    for point in rising_points:
        manager.update(frame_index=point.frame_index, point=point, hoop_lock=_hoop())

    assert len(manager.tracking_context_points()) == 3
