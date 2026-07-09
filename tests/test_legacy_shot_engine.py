"""LegacyShotEngine wiring: a stubbed ball detector drives the wholesale wrap
through detect -> sparse -> ShotCandidateManager -> finalize -> IR Event, with no
torch/weights/video. Byte-identical parity with run_pipeline is proven by the
clip gate (scripts/run_platform_clip.py), not here.
"""

from __future__ import annotations

import numpy as np

from swishsync.core.schemas import EventType, ObjectClass
from swishsync.vision.tracking.legacy_shot_engine import LegacyShotEngine
from swishsync_cv.data import DetectionRecord


class _StubBall:
    """Emits one basketball detection per frame along a scripted path."""

    def __init__(self, path):
        self.config = None
        self._path = path  # {frame_index: (x, y)}

    def detect(self, frame, frame_index, timestamp_ms):
        if frame_index not in self._path:
            return []
        x, y = self._path[frame_index]
        return [
            DetectionRecord(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                label="basketball",
                class_name="sports ball",
                confidence=0.9,
                bbox_xyxy=(x - 5, y - 5, x + 5, y + 5),
            )
        ]

    def detect_crop(self, *args, **kwargs):
        return []


class _StubDetector:
    def __init__(self, legacy):
        self.legacy = legacy


def _throw_path():
    # image-space throw: y falls (upward) to an interior apex, then rises.
    path = {}
    for i in range(20):
        x = 400 + i * 8
        y = 300 - 12 * i + 0.6 * i * i
        path[i] = (float(x), float(y))
    return path


def test_engine_finalizes_shot_and_emits_event():
    engine = LegacyShotEngine(_StubDetector(_StubBall(_throw_path())))
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    for i in range(20):
        engine.update(None, frame_index=i, t_ms=i * 33.0, frame=frame)
    engine.close()

    shots = engine.finalized_shots
    assert len(shots) == 1
    assert shots[0].parabola_fit is not None
    assert len(shots[0].candidate_points) >= 4

    events = engine.finalized_events()
    assert len(events) == 1
    ev = events[0]
    assert ev.type == EventType.SHOT
    assert "parabola_fit" in ev.payload
    assert ev.evidence["weighted_residual_rmse"] is not None
    assert ev.evidence["finalize_reason"] == "end_of_video"

    tracks = engine.tracks()
    assert len(tracks) == 1
    assert tracks[0].cls == ObjectClass.BALL
    assert len(tracks[0].states) == len(shots[0].candidate_points)


def test_update_requires_frame():
    engine = LegacyShotEngine(_StubDetector(_StubBall({})))
    try:
        engine.update(None, frame_index=0, t_ms=0.0, frame=None)
    except ValueError:
        return
    raise AssertionError("expected ValueError when frame is None")
