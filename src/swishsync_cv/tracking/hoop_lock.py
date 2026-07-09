"""Stable hoop acquisition, locking, and revalidation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from swishsync_cv.config import HoopLockConfig
from swishsync_cv.data import DetectionRecord, HoopLock
from swishsync_cv.detection.hoop_detector import (
    HoopCandidate,
    HybridHoopDetector,
    refine_rim_bbox,
)
from swishsync_cv.tracking.hoop_geometry import bbox_xywh_to_xyxy, hoop_center_from_bbox

HoopPhase = Literal["acquisition", "locked", "revalidation"]

# ponytail: freeze hardcodes, one config toggle (freeze_when_locked) is enough
_FREEZE_STABLE_FRAMES = 6  # consecutive settled locked frames before freezing
_FREEZE_STABILITY_PX = 2.5  # max center drift between frames to count as settled
_FREEZE_MAX_LOCKED_FRAMES = 12  # hard cap: freeze to median box even if jittery,
# so a wobbly detector box can't keep smoothing into shot time


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
        self._manual_lock = False
        self._frozen = False
        self._stable_locked_count = 0
        self._last_locked_center: tuple[float, float] | None = None
        self._locked_box_samples: list[tuple[float, float, float, float]] = []

    @property
    def is_manual_lock(self) -> bool:
        return self._manual_lock

    def lock_manual_bbox(
        self,
        frame_index: int,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> HoopLock:
        """Lock hoop position from a user-provided bbox in x,y,w,h form."""

        bbox = bbox_xywh_to_xyxy(x, y, width, height)
        center_x, center_y = hoop_center_from_bbox(bbox)
        self._lock = HoopLock(
            center_x=center_x,
            center_y=center_y,
            bbox_xyxy=bbox,
            confidence=1.0,
            locked_at_frame=frame_index,
            is_locked=True,
            detector_confidence=1.0,
            color_score=1.0,
        )
        self._phase = "locked"
        self._manual_lock = True
        self._observations.clear()
        return self._lock

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
        if self._manual_lock:
            return self._lock

        # static camera: once a confident lock exists, freeze it so rim contact
        # (ball bouncing through the hoop) can never drag the anchor around.
        if self._frozen and self._lock is not None:
            self._refine_rim_bbox_once(frame)
            return self._lock

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
            result = self._update_acquisition(frame_index, best)
        elif self._phase == "revalidation":
            result = self._update_revalidation(frame_index, best)
        else:
            result = self._update_locked(frame_index, best)

        if self._refine_rim_bbox_once(frame):
            result = self._lock

        self._maybe_freeze()
        return result

    def _maybe_freeze(self) -> None:
        # freeze after the lock settles, snapping to the MEDIAN of the locked
        # boxes so a wobbly detector box can't keep smoothing into shot time.
        # Two triggers: a quiet run (6 near-identical frames) or a hard frame
        # cap (12) — whichever comes first. Median rejects jitter outliers.
        if not self.config.freeze_when_locked or self._lock is None:
            return
        # accumulate whenever a lock EXISTS, through locked<->revalidation
        # flicker (flaky far-court detection): resetting on every flicker means
        # jittery clips never freeze until the churn stops, well into shot time.
        if not self._lock.is_locked:
            self._stable_locked_count = 0
            self._locked_box_samples.clear()
            return
        center = (self._lock.center_x, self._lock.center_y)
        if self._last_locked_center is not None and (
            math_hypot(
                center[0] - self._last_locked_center[0],
                center[1] - self._last_locked_center[1],
            )
            <= _FREEZE_STABILITY_PX
        ):
            self._stable_locked_count += 1
        else:
            self._stable_locked_count = 1
        self._last_locked_center = center
        self._locked_box_samples.append(self._lock.bbox_xyxy)
        if (
            self._stable_locked_count >= _FREEZE_STABLE_FRAMES
            or len(self._locked_box_samples) >= _FREEZE_MAX_LOCKED_FRAMES
        ):
            self._freeze_to_median()

    def _freeze_to_median(self) -> None:
        import statistics

        samples = self._locked_box_samples
        median_box = tuple(
            statistics.median(sample[i] for sample in samples) for i in range(4)
        )
        center_x = (median_box[0] + median_box[2]) / 2.0
        center_y = (median_box[1] + median_box[3]) / 2.0
        self._lock = replace(
            self._lock,
            bbox_xyxy=median_box,
            center_x=center_x,
            center_y=center_y,
        )
        self._frozen = True

    def _refine_rim_bbox_once(self, frame: np.ndarray) -> bool:
        if self._lock is None or self._lock.rim_bbox_xyxy is not None:
            return False
        rim_bbox = refine_rim_bbox(frame, self._lock.bbox_xyxy, self.config)
        if rim_bbox is None:
            return False
        # sanity check (user spec 2026-07-10): the ring must sit INSIDE the
        # hoop box and toward its TOP. A "rim" found low in the box is the net
        # or floor clutter (clip AA locked on the net at dusk) — reject it and
        # keep retrying on later frames; None falls back to legacy geometry.
        if not rim_within_hoop_top(rim_bbox, self._lock.bbox_xyxy):
            return False
        self._lock = replace(self._lock, rim_bbox_xyxy=rim_bbox)
        return True

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


def rim_within_hoop_top(
    rim_bbox: tuple[float, float, float, float],
    hoop_bbox: tuple[float, float, float, float],
    horizontal_tolerance_ratio: float = 0.05,
    top_band_ratio: float = 0.6,
) -> bool:
    """True when the ring band sits inside the hoop box, in its upper region.

    The ring is physically the top of the hoop structure; anything detected in
    the bottom 40% of the hoop box is net/clutter, not the ring.
    """

    rx1, ry1, rx2, ry2 = rim_bbox
    hx1, hy1, hx2, hy2 = hoop_bbox
    tol = (hx2 - hx1) * horizontal_tolerance_ratio
    inside = (
        rx1 >= hx1 - tol
        and rx2 <= hx2 + tol
        and ry1 >= hy1 - tol
        and ry2 <= hy2 + tol
    )
    toward_top = ry2 <= hy1 + (hy2 - hy1) * top_band_ratio
    return inside and toward_top
