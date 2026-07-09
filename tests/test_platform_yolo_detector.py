"""Platform YoloDetector: class mapping + backend device wiring.

Uses a stubbed legacy YoloObjectDetector so the test needs no torch/weights.
"""

from __future__ import annotations

import pytest

from swishsync.core.registry import DETECTORS, TRACKERS
from swishsync.core.schemas import ObjectClass
from swishsync.vision.backends import make_backend
from swishsync.vision.detection.yolo import YoloDetector
from swishsync_cv.data import DetectionRecord


class _StubLegacy:
    """Stands in for YoloObjectDetector; records the config it was built with."""

    def __init__(self, config):
        self.config = config
        self._records: list[DetectionRecord] = []

    def set_records(self, records):
        self._records = records

    def detect(self, frame, frame_index, timestamp_ms):
        return self._records


def _record(label, class_name):
    return DetectionRecord(
        frame_index=1,
        timestamp_ms=33.0,
        label=label,
        class_name=class_name,
        confidence=0.9,
        bbox_xyxy=(10.0, 20.0, 30.0, 40.0),
    )


@pytest.fixture
def detector(monkeypatch):
    monkeypatch.setattr(
        "swishsync.vision.detection.yolo.YoloObjectDetector", _StubLegacy
    )
    return YoloDetector(make_backend("cpu"), weights="yolov8n.pt", confidence=0.3)


def test_registered():
    assert DETECTORS["yolo"] is YoloDetector


def test_config_built_from_backend_and_args(detector):
    cfg = detector.legacy.config
    assert cfg.model_path == "yolov8n.pt"
    assert cfg.confidence_threshold == 0.3
    assert cfg.device == "cpu"


def test_maps_labels_to_object_class(detector):
    detector.legacy.set_records([_record("basketball", "sports ball"), _record("hoop", "hoop")])
    out = detector.detect(frame=None, frame_index=1, t_ms=33.0)
    assert [d.cls for d in out] == [ObjectClass.BALL, ObjectClass.HOOP]
    assert out[0].bbox_xyxy == (10.0, 20.0, 30.0, 40.0)
    assert out[0].conf == 0.9
    assert out[0].frame == 1 and out[0].t_ms == 33.0


def test_drops_unmapped_labels(detector):
    bogus = _record("basketball", "x")
    object.__setattr__(bogus, "label", "person")  # frozen dataclass
    detector.legacy.set_records([bogus])
    assert detector.detect(frame=None, frame_index=1, t_ms=33.0) == []


def test_engine_registered():
    # importing the tracker package registers it
    import swishsync.vision.tracking  # noqa: F401

    assert "legacy_shot_engine" in TRACKERS
