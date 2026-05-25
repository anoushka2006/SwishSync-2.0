"""Stable hoop acquisition, locking, and revalidation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from swishsync_cv.config import HoopLockConfig
from swishsync_cv.data import DetectionRecord, HoopLock
from swishsync_cv.detection.hoop_detector import HoopCandidate, HybridHoopDetector

HoopPhase = Literal["acquisition", "locked", "revalidation"]


@dataclass
class _HoopObservation:
    frame_index: int
    center_x: float
    center_y: float
    bbox_xyxy: tuple[float, float, float, float]
    detector_confidence: float
    color_score: float
    composite_score: float


class HoopLockTracker:
    """Acquire, lock, and revalidate a stable hoop anchor across frames."""

    def __init__(self, config: HoopLockConfig) -> None:
        self.config = config
        self._detector = HybridHoopDetector(config)
        self._lock: HoopLock | None = None
        self._phase: HoopPhase = "acquisition"
        self._observations: list[_HoopObservation] = []
        self._recent_scores: list[float] = []
        self._missed_detection_frames = 0
        self._last_detection_frame = -1

    @property
    def is_locked(self) -> bool:
        return self._lock is not None and self._lock.is_locked

    @property
    def lock(self) -> HoopLock | None:
        return self._lock

    @property
    def phase(self) -> HoopPhase:
        return self._phase

    def should_process_frame(self, frame_index: int, detection_ran: bool) -> bool:
        if frame_index < self.config.acquisition_frames:
            return True
        if not self.is_locked:
            return detection_ran or frame_index % 2 == 0
        return detection_ran or self._needs_revalidation()

    def update(
        self,
        frame_index: int,
        frame: np.ndarray,
        yolo_hoop_detections: list[DetectionRecord] | None = None,
    ) -> HoopLock | None:
        detections = yolo_hoop_detections or []
        candidates = self._detector.detect(frame, detections)
        best = candidates[0] if candidates else None

        if best is not None:
            self._last_detection_frame = frame_index
            self._missed_detection_frames = 0
            self._recent_scores.append(best.composite_score)
            self._recent_scores = self._recent_scores[-12:]
        else:
            self._missed_detection_frames += 1

        if self._phase == "acquisition":
            return self._update_acquisition(frame_index, best)
        if self._phase == "revalidation":
            return self._update_revalidation(frame_index, best)
        return self._update_locked(frame_index, best)

    def _update_acquisition(
        self,
        frame_index: int,
        candidate: HoopCandidate | None,
    ) -> HoopLock | None:
        if candidate is not None:
            self._observations.append(_observation_from_candidate(frame_index, candidate))

        if self._should_lock_from_observations(frame_index):
            self._commit_lock(frame_index)
            return self._lock

        if frame_index >= self.config.acquisition_frames - 1:
            if self._observations:
                self._commit_lock(frame_index)
            return self._lock
        return self._preview_lock(frame_index)

    def _update_locked(
        self,
        frame_index: int,
        candidate: HoopCandidate | None,
    ) -> HoopLock | None:
        if self._lock is None:
            self._phase = "acquisition"
            return self._update_acquisition(frame_index, candidate)

        if self._needs_revalidation():
            self._phase = "revalidation"
            return self._update_revalidation(frame_index, candidate)

        if candidate is not None and self._candidate_matches_lock(candidate):
            self._smooth_toward(frame_index, candidate)
        return self._lock

    def _update_revalidation(
        self,
        frame_index: int,
        candidate: HoopCandidate | None,
    ) -> HoopLock | None:
        if self._lock is None:
            self._phase = "acquisition"
            return self._update_acquisition(frame_index, candidate)

        if candidate is None:
            if self._missed_detection_frames >= self.config.revalidation_miss_frames:
                self._downgrade_confidence()
            return self._lock

        if not self._candidate_matches_lock(candidate):
            if candidate.composite_score > self._lock.confidence * 1.15:
                self._observations = [_observation_from_candidate(frame_index, candidate)]
                self._commit_lock(frame_index, replace=True)
            return self._lock

        self._smooth_toward(frame_index, candidate)
        self._phase = "locked"
        return self._lock

    def _should_lock_from_observations(self, frame_index: int) -> bool:
        cluster = self._stable_cluster()
        if len(cluster) < self.config.min_acquisition_observations:
            return False
        average_confidence = sum(item.composite_score for item in cluster) / len(cluster)
        enough_frames = frame_index >= self.config.min_acquisition_observations
        return average_confidence >= self.config.lock_confidence and enough_frames

    def _commit_lock(self, frame_index: int, replace: bool = False) -> None:
        cluster = self._stable_cluster()
        if not cluster:
            return

        center_x = float(np.mean([item.center_x for item in cluster]))
        center_y = float(np.mean([item.center_y for item in cluster]))
        x1 = float(np.mean([item.bbox_xyxy[0] for item in cluster]))
        y1 = float(np.mean([item.bbox_xyxy[1] for item in cluster]))
        x2 = float(np.mean([item.bbox_xyxy[2] for item in cluster]))
        y2 = float(np.mean([item.bbox_xyxy[3] for item in cluster]))
        detector_confidence = float(np.mean([item.detector_confidence for item in cluster]))
        color_score = float(np.mean([item.color_score for item in cluster]))
        confidence = _hoop_confidence(
            detector_confidence=detector_confidence,
            color_score=color_score,
            observations=cluster,
        )

        if self._lock is not None and not replace:
            alpha = self.config.smoothing_alpha
            center_x = _ema(self._lock.center_x, center_x, alpha)
            center_y = _ema(self._lock.center_y, center_y, alpha)
            x1 = _ema(self._lock.bbox_xyxy[0], x1, alpha)
            y1 = _ema(self._lock.bbox_xyxy[1], y1, alpha)
            x2 = _ema(self._lock.bbox_xyxy[2], x2, alpha)
            y2 = _ema(self._lock.bbox_xyxy[3], y2, alpha)

        self._lock = HoopLock(
            center_x=center_x,
            center_y=center_y,
            bbox_xyxy=(x1, y1, x2, y2),
            confidence=confidence,
            locked_at_frame=frame_index,
            is_locked=True,
            detector_confidence=detector_confidence,
            color_score=color_score,
        )
        self._phase = "locked"
        self._observations.clear()

    def _preview_lock(self, frame_index: int) -> HoopLock | None:
        cluster = self._stable_cluster()
        if not cluster:
            return None
        center_x = float(np.mean([item.center_x for item in cluster]))
        center_y = float(np.mean([item.center_y for item in cluster]))
        x1 = float(np.mean([item.bbox_xyxy[0] for item in cluster]))
        y1 = float(np.mean([item.bbox_xyxy[1] for item in cluster]))
        x2 = float(np.mean([item.bbox_xyxy[2] for item in cluster]))
        y2 = float(np.mean([item.bbox_xyxy[3] for item in cluster]))
        detector_confidence = float(np.mean([item.detector_confidence for item in cluster]))
        color_score = float(np.mean([item.color_score for item in cluster]))
        return HoopLock(
            center_x=center_x,
            center_y=center_y,
            bbox_xyxy=(x1, y1, x2, y2),
            confidence=_hoop_confidence(
                detector_confidence=detector_confidence,
                color_score=color_score,
                observations=cluster,
            ),
            locked_at_frame=frame_index,
            is_locked=False,
            detector_confidence=detector_confidence,
            color_score=color_score,
        )

    def _smooth_toward(self, frame_index: int, candidate: HoopCandidate) -> None:
        if self._lock is None:
            return

        alpha = self.config.smoothing_alpha
        center_x = _ema(self._lock.center_x, candidate.center_x, alpha)
        center_y = _ema(self._lock.center_y, candidate.center_y, alpha)
        x1, y1, x2, y2 = self._lock.bbox_xyxy
        bx1, by1, bx2, by2 = candidate.bbox_xyxy
        bbox = (
            _ema(x1, bx1, alpha),
            _ema(y1, by1, alpha),
            _ema(x2, bx2, alpha),
            _ema(y2, by2, alpha),
        )
        detector_confidence = _ema(
            self._lock.detector_confidence,
            candidate.detector_confidence,
            alpha,
        )
        color_score = _ema(self._lock.color_score, candidate.color_score, alpha)
        confidence = _hoop_confidence(
            detector_confidence=detector_confidence,
            color_score=color_score,
            observations=[_observation_from_candidate(frame_index, candidate)],
            temporal_stability=1.0 - min(self._missed_detection_frames / 10.0, 1.0),
        )
        self._lock = HoopLock(
            center_x=center_x,
            center_y=center_y,
            bbox_xyxy=bbox,
            confidence=confidence,
            locked_at_frame=self._lock.locked_at_frame,
            is_locked=True,
            detector_confidence=detector_confidence,
            color_score=color_score,
        )

    def _stable_cluster(self) -> list[_HoopObservation]:
        if not self._observations:
            return []
        anchor = self._observations[-1]
        cluster = [
            observation
            for observation in self._observations
            if _center_distance(observation, anchor)
            <= self.config.max_candidate_center_drift_px
        ]
        return cluster[-max(self.config.min_acquisition_observations * 2, 6) :]

    def _candidate_matches_lock(self, candidate: HoopCandidate) -> bool:
        if self._lock is None:
            return False
        distance = math_hypot(
            candidate.center_x - self._lock.center_x,
            candidate.center_y - self._lock.center_y,
        )
        max_drift = max(self.config.max_candidate_center_drift_px * 2.0, 60.0)
        return distance <= max_drift

    def _needs_revalidation(self) -> bool:
        if self._lock is None:
            return False
        if self._missed_detection_frames >= self.config.revalidation_miss_frames:
            return True
        if not self._recent_scores:
            return False
        recent_average = sum(self._recent_scores[-4:]) / len(self._recent_scores[-4:])
        return recent_average < self._lock.confidence * self.config.unlock_confidence_ratio

    def _downgrade_confidence(self) -> None:
        if self._lock is None:
            return
        self._lock = HoopLock(
            center_x=self._lock.center_x,
            center_y=self._lock.center_y,
            bbox_xyxy=self._lock.bbox_xyxy,
            confidence=max(0.0, self._lock.confidence * 0.92),
            locked_at_frame=self._lock.locked_at_frame,
            is_locked=True,
            detector_confidence=self._lock.detector_confidence,
            color_score=self._lock.color_score,
        )


def _observation_from_candidate(
    frame_index: int,
    candidate: HoopCandidate,
) -> _HoopObservation:
    return _HoopObservation(
        frame_index=frame_index,
        center_x=candidate.center_x,
        center_y=candidate.center_y,
        bbox_xyxy=candidate.bbox_xyxy,
        detector_confidence=candidate.detector_confidence,
        color_score=candidate.color_score,
        composite_score=candidate.composite_score,
    )


def _hoop_confidence(
    detector_confidence: float,
    color_score: float,
    observations: list[_HoopObservation],
    temporal_stability: float | None = None,
) -> float:
    if temporal_stability is None:
        if len(observations) <= 1:
            temporal_stability = 0.55
        else:
            centers = np.array([(item.center_x, item.center_y) for item in observations])
            drift = float(np.mean(np.std(centers, axis=0)))
            temporal_stability = max(0.0, min(1.0, 1.0 - drift / 25.0))

    spatial_stability = temporal_stability
    score = (
        0.30 * detector_confidence
        + 0.25 * temporal_stability
        + 0.25 * spatial_stability
        + 0.20 * color_score
    )
    return max(0.0, min(1.0, score))


def _center_distance(first: _HoopObservation, second: _HoopObservation) -> float:
    return math_hypot(first.center_x - second.center_x, first.center_y - second.center_y)


def _ema(previous: float, current: float, alpha: float) -> float:
    return previous * (1.0 - alpha) + current * alpha


def math_hypot(x: float, y: float) -> float:
    return float((x * x + y * y) ** 0.5)
