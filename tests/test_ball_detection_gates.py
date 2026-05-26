from pathlib import Path

from swishsync_cv.config import ShotCandidateConfig
from swishsync_cv.data import DetectionRecord, HoopLock, SparseBallDetection
from swishsync_cv.detection.ball_detection_gates import filter_basketball_detections
from swishsync_cv.tracking.parabola import is_floor_bounce_point


def _hoop_f() -> HoopLock:
    return HoopLock(
        center_x=1286.5,
        center_y=280.5,
        bbox_xyxy=(1214.0, 4.0, 1359.0, 537.0),
        confidence=1.0,
        locked_at_frame=0,
        is_locked=True,
    )


def _hoop_j() -> HoopLock:
    return HoopLock(
        center_x=1168.5,
        center_y=166.5,
        bbox_xyxy=(1088.0, 0.0, 1249.0, 333.0),
        confidence=1.0,
        locked_at_frame=0,
        is_locked=True,
    )


def _basketball(
    frame_index: int,
    center_x: float,
    center_y: float,
    confidence: float = 0.5,
) -> DetectionRecord:
    half = 18.0
    return DetectionRecord(
        frame_index=frame_index,
        timestamp_ms=float(frame_index * 33),
        label="basketball",
        class_name="sports ball",
        confidence=confidence,
        bbox_xyxy=(
            center_x - half,
            center_y - half,
            center_x + half,
            center_y + half,
        ),
    )


def _hoop_detection(frame_index: int = 0) -> DetectionRecord:
    return DetectionRecord(
        frame_index=frame_index,
        timestamp_ms=0.0,
        label="hoop",
        class_name="hoop",
        confidence=0.9,
        bbox_xyxy=(1214.0, 4.0, 1359.0, 537.0),
    )


def test_filter_disabled_passes_through():
    floor = _basketball(6, 986.0, 979.0)
    detections = [floor]

    result = filter_basketball_detections(
        detections,
        _hoop_f(),
        floor_margin_px=100.0,
        enabled=False,
    )

    assert result == detections


def test_filter_passes_through_when_hoop_unlocked():
    floor = _basketball(6, 986.0, 979.0)
    unlocked = HoopLock(
        center_x=1286.5,
        center_y=280.5,
        bbox_xyxy=(1214.0, 4.0, 1359.0, 537.0),
        confidence=0.5,
        locked_at_frame=0,
        is_locked=False,
    )

    result = filter_basketball_detections(
        [floor],
        unlocked,
        floor_margin_px=100.0,
    )

    assert result == [floor]


def test_filter_rejects_floor_band_basketball_when_locked():
    floor = _basketball(6, 986.0, 979.0)
    upper = _basketball(66, 722.0, 117.0)

    result = filter_basketball_detections(
        [floor, upper, _hoop_detection()],
        _hoop_f(),
        floor_margin_px=100.0,
    )

    assert floor not in result
    assert upper in result
    assert _hoop_detection() in result


def test_filter_boundary_matches_is_floor_bounce_point():
    hoop = _hoop_f()
    margin = 100.0
    on_boundary = hoop.rim_center_y + margin
    below = on_boundary - 1.0
    above = on_boundary + 1.0

    kept = filter_basketball_detections(
        [_basketball(1, 500.0, below)],
        hoop,
        floor_margin_px=margin,
    )
    dropped = filter_basketball_detections(
        [_basketball(1, 500.0, above)],
        hoop,
        floor_margin_px=margin,
    )

    assert len(kept) == 1
    assert dropped == []


def test_f_floor_cluster_coordinates_rejected():
    hoop = _hoop_f()
    margin = ShotCandidateConfig().floor_below_rim_margin_px
    rows = [
        _basketball(5, 992.9, 946.5, 0.27),
        _basketball(6, 985.7, 979.3, 0.76),
        _basketball(11, 956.6, 973.5, 0.85),
        _basketball(12, 951.2, 949.5, 0.55),
    ]

    result = filter_basketball_detections(rows, hoop, floor_margin_px=margin)
    assert result == []


def test_j_gather_cluster_coordinates_rejected():
    hoop = _hoop_j()
    margin = ShotCandidateConfig().floor_below_rim_margin_px
    rows = [
        _basketball(0, 882.9, 539.5, 0.25),
        _basketball(1, 882.8, 541.1, 0.32),
        _basketball(7, 881.4, 578.3, 0.39),
        _basketball(9, 882.3, 591.9, 0.37),
        _basketball(10, 881.9, 596.8, 0.36),
        _basketball(11, 879.6, 601.3, 0.35),
    ]

    result = filter_basketball_detections(rows, hoop, floor_margin_px=margin)
    assert result == []


def test_j_burst_upper_court_points_preserved():
    hoop = _hoop_j()
    margin = ShotCandidateConfig().floor_below_rim_margin_px
    burst = [
        _basketball(40, 880.0, 443.8, 0.35),
        _basketball(41, 885.0, 415.0, 0.72),
        _basketball(42, 892.4, 387.4, 0.58),
        _basketball(43, 898.4, 361.3, 0.51),
        _basketball(44, 906.0, 336.1, 0.27),
    ]

    result = filter_basketball_detections(burst, hoop, floor_margin_px=margin)
    kept_frames = {detection.frame_index for detection in result}
    assert kept_frames == {41, 42, 43, 44}
    assert 40 not in kept_frames


def test_core_upper_court_points_pass_through():
    hoop = HoopLock(
        center_x=500.0,
        center_y=365.0,
        bbox_xyxy=(430.0, 320.0, 570.0, 410.0),
        confidence=1.0,
        locked_at_frame=0,
        is_locked=True,
    )
    margin = ShotCandidateConfig().floor_below_rim_margin_px
    core_like = [
        _basketball(62, 1200.0, 400.0, 0.8),
        _basketball(63, 1190.0, 360.0, 0.8),
        _basketball(64, 1180.0, 320.0, 0.8),
        _basketball(65, 1170.0, 285.0, 0.7),
    ]

    result = filter_basketball_detections(core_like, hoop, floor_margin_px=margin)

    assert result == core_like
    for detection in core_like:
        point = SparseBallDetection(
            detection.frame_index,
            detection.timestamp_ms,
            detection.center[0],
            detection.center[1],
            detection.confidence,
        )
        assert not is_floor_bounce_point(point, hoop, floor_margin_px=margin)
