"""Vision layer interfaces. Implementations live in vision/detection, vision/
tracking, vision/calibration and register themselves.

Contract: vision models emit ObjectClass, never basketball roles. A detector
returns `player`, not `shooter`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from swishsync.core.schemas import CourtModel, Detection, Track
from swishsync.vision.backends import InferenceBackend


class Detector(ABC):
    """frame pixels -> detections. Backend chooses CPU/GPU/remote execution."""

    def __init__(self, backend: InferenceBackend) -> None:
        self.backend = backend

    @abstractmethod
    def detect(self, frame, frame_index: int, t_ms: float) -> list[Detection]:
        ...


class Tracker(ABC):
    """per-frame detections -> tracks. Assigns track_ids, nothing semantic.

    `frame` (raw pixels) and `t_ms` are optional context: pure trackers ignore
    them; the Phase-A wholesale legacy engine needs pixels (hoop color refine,
    rim rescan). The parameter dissolves when stages split in MP-C."""

    @abstractmethod
    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        t_ms: float = 0.0,
        frame=None,
    ) -> None:
        ...

    @abstractmethod
    def tracks(self) -> list[Track]:
        """Return all tracks accumulated so far."""


class CourtCalibrator(ABC):
    """frame(s) -> image->court homography. The analytics linchpin."""

    @abstractmethod
    def calibrate(self, frame, frame_index: int) -> CourtModel:
        ...
