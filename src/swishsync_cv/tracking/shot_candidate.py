"""Minimal shot lifecycle: collect points, fit once, reset."""

from __future__ import annotations

import logging

from swishsync_cv.config import ShotCandidateConfig
from swishsync_cv.data import HoopLock, ShotCandidate, ShotLifecycleState, SparseBallDetection
from swishsync_cv.tracking.shot_finalization import finalize_shot

logger = logging.getLogger("swishsync_cv.shot")


class ShotCandidateManager:
    """Collect sparse ball points, fit one parabola when the shot ends, then reset."""

    def __init__(
        self,
        config: ShotCandidateConfig,
        frame_height: int,
    ) -> None:
        self.config = config
        self.frame_height = frame_height
        self.active: ShotCandidate | None = None
        self.display_shot: ShotCandidate | None = None
        self.finalized_shots: list[ShotCandidate] = []
        self._pre_shot_buffer: list[SparseBallDetection] = []
        self._frames_since_point = 0

    @property
    def lifecycle_state(self) -> ShotLifecycleState:
        if self.active is None:
            return "idle"
        return "collecting_shot"

    @property
    def candidate_point_count(self) -> int:
        if self.active is None:
            return 0
        return len(self.active.candidate_points)

    def update(
        self,
        frame_index: int,
        point: SparseBallDetection | None,
        hoop_lock: HoopLock | None = None,
    ) -> ShotCandidate | None:
        if point is None:
            if self.active is not None:
                self._frames_since_point += 1
                if self._frames_since_point >= self.config.max_idle_frames:
                    return self._finalize_active()
            return None

        if self.active is None:
            self._pre_shot_buffer.append(point)
            self._pre_shot_buffer = self._pre_shot_buffer[-12:]
            self._try_start_collection()
            return None

        self.active.candidate_points.append(point)
        self._frames_since_point = 0
        logger.info(
            "point added frame=%s total=%s",
            frame_index,
            len(self.active.candidate_points),
        )

        if self._should_end_collection(point, hoop_lock):
            return self._finalize_active()
        return None

    def finalize(self) -> ShotCandidate | None:
        if self.active is not None:
            return self._finalize_active()
        return None

    def _try_start_collection(self) -> None:
        if not self._should_start_collection(self._pre_shot_buffer):
            return

        seed_points = self._pre_shot_buffer[-self.config.min_points_to_start :]
        self.active = ShotCandidate(
            start_frame=seed_points[0].frame_index,
            state="collecting_shot",
        )
        self.active.candidate_points.extend(seed_points)
        self._frames_since_point = 0
        self._pre_shot_buffer.clear()
        logger.info(
            "shot started frame=%s seed_points=%s",
            self.active.start_frame,
            len(self.active.candidate_points),
        )

    def _should_start_collection(self, recent_points: list[SparseBallDetection]) -> bool:
        if len(recent_points) < self.config.min_points_to_start:
            return False

        last_three = recent_points[-3:]
        vertical_velocities = []
        for previous, current in zip(last_three, last_three[1:]):
            dt = max(current.frame_index - previous.frame_index, 1)
            vertical_velocities.append((current.y - previous.y) / dt)

        if not all(velocity < -self.config.upward_velocity_threshold for velocity in vertical_velocities):
            return False

        latest = recent_points[-1]
        upper_body_threshold = self.frame_height * self.config.upper_body_y_ratio
        if latest.y >= upper_body_threshold:
            return False

        xs = [point.x for point in recent_points]
        ys = [point.y for point in recent_points]
        y_spread = max(ys) - min(ys)
        x_spread = max(xs) - min(xs)
        return y_spread > 8 and x_spread < y_spread * 2.5

    def _should_end_collection(
        self,
        point: SparseBallDetection,
        hoop_lock: HoopLock | None,
    ) -> bool:
        assert self.active is not None
        points = self.active.candidate_points
        if len(points) < self.config.min_validated_points_for_fit:
            return False

        apex_index = min(range(len(points)), key=lambda index: points[index].y)
        if apex_index <= 0 or apex_index >= len(points) - 1:
            return False

        previous = points[-1]
        dt = max(point.frame_index - previous.frame_index, 1)
        if (point.y - previous.y) / dt <= 0.5:
            return False

        if hoop_lock is not None and point.y >= hoop_lock.center_y:
            if self.active.post_rim_frames_remaining == 0:
                self.active.post_rim_frames_remaining = self.config.post_rim_extension_frames
            self.active.post_rim_frames_remaining -= 1
            return self.active.post_rim_frames_remaining <= 0

        recent = points[-4:]
        if len(recent) >= 4:
            horizontal_jumps = [
                abs(recent[index].x - recent[index - 1].x) for index in range(1, len(recent))
            ]
            if max(horizontal_jumps) > 60:
                return True

        return False

    def _finalize_active(self) -> ShotCandidate | None:
        if self.active is None:
            return None

        if len(self.active.candidate_points) < self.config.min_points_to_start:
            logger.info("candidate reset (too few points)")
            self.active = None
            self._frames_since_point = 0
            return None

        self.active.end_frame = self.active.candidate_points[-1].frame_index
        finalized = finalize_shot(self.active, self.config)
        self.active = None
        self._frames_since_point = 0
        logger.info("candidate reset")

        if finalized.parabola_fit is None:
            return None

        self.display_shot = finalized
        self.finalized_shots.append(finalized)
        return finalized
