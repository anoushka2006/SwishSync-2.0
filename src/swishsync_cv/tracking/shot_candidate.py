"""Minimal shot lifecycle: collect points, fit once, reset."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from swishsync_cv.config import HoopLockConfig, ShotCandidateConfig, ShotStoryConfig
from swishsync_cv.data import (
    HoopLock,
    ShotCandidate,
    ShotFinalizeReason,
    ShotLifecycleState,
    ShotMotionDirection,
    SparseBallDetection,
)
from swishsync_cv.tracking.gap_recovery import append_continuity_point, backfill_gap_points
from swishsync_cv.tracking.parabola import is_floor_bounce_point
from swishsync_cv.tracking.shot_finalization import finalize_shot

logger = logging.getLogger("swishsync_cv.shot")


@dataclass(frozen=True)
class FinalizeCooldownState:
    """Blocks spurious new shots immediately after a shot ends."""

    frame: int
    reason: ShotFinalizeReason
    last_motion: ShotMotionDirection


class ShotCandidateManager:
    """Collect sparse ball points, fit one parabola when the shot ends, then reset."""

    def __init__(
        self,
        config: ShotCandidateConfig,
        frame_height: int,
        hoop_lock_config: HoopLockConfig | None = None,
        story_config: ShotStoryConfig | None = None,
    ) -> None:
        self.config = config
        self.hoop_lock_config = hoop_lock_config or HoopLockConfig()
        self.story_config = story_config or ShotStoryConfig()
        self.frame_height = frame_height
        self.active: ShotCandidate | None = None
        self.display_shot: ShotCandidate | None = None
        self.finalized_shots: list[ShotCandidate] = []
        self._pre_shot_buffer: list[SparseBallDetection] = []
        self._frames_since_point = 0
        self._interpolated_ignored_count = 0
        self._last_hoop_lock: HoopLock | None = None
        self._pending_finalize_reason: ShotFinalizeReason | None = None
        self._cooldown: FinalizeCooldownState | None = None
        self.post_shot_debug_points: list[SparseBallDetection] = []

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

    def preview_pickup_points(self) -> list[SparseBallDetection]:
        """Render-only gather preview while idle, before collection starts."""

        if self.active is not None:
            return []

        measured = self._non_floor_measured(
            self._pre_shot_buffer,
            self._last_hoop_lock,
        )
        if len(measured) < 2:
            return []

        latest = measured[-1]
        pickup = [
            point
            for point in measured
            if point.frame_index < latest.frame_index
        ]
        return pickup if len(pickup) >= 2 else []

    def update(
        self,
        frame_index: int,
        point: SparseBallDetection | None,
        hoop_lock: HoopLock | None = None,
    ) -> ShotCandidate | None:
        self._last_hoop_lock = hoop_lock
        if point is None:
            if self.active is not None:
                self._frames_since_point += 1
                if self._frames_since_point >= self.config.max_idle_frames:
                    self._pending_finalize_reason = "idle"
                    return self._finalize_active()
            return None

        if self.active is None:
            self._pre_shot_buffer.append(point)
            self._pre_shot_buffer = self._pre_shot_buffer[-12:]
            if (
                not point.interpolated
                and self._is_in_finalize_cooldown(point.frame_index)
                and self._is_falling_point(point)
            ):
                self.post_shot_debug_points.append(point)
                logger.info(
                    "post-shot continuation suppressed frame=%s",
                    point.frame_index,
                )
            self._try_start_collection()
            return None

        self._frames_since_point = 0
        if point.interpolated:
            self._interpolated_ignored_count += 1
        else:
            if self._should_finalize_on_long_gap(point):
                self._pending_finalize_reason = "unknown"
                finalized = self._finalize_active()
                if self._is_in_finalize_cooldown(point.frame_index):
                    self.post_shot_debug_points.append(point)
                    logger.info(
                        "post-shot continuation suppressed frame=%s",
                        point.frame_index,
                    )
                return finalized

            if self._should_finalize_after_rim_sequence(point, hoop_lock):
                self._pending_finalize_reason = "post_rim"
                finalized = self._finalize_active()
                if self._is_in_finalize_cooldown(point.frame_index):
                    self.post_shot_debug_points.append(point)
                    logger.info(
                        "post-shot continuation suppressed frame=%s",
                        point.frame_index,
                    )
                return finalized

            if self._should_finalize_before_reacquisition(point, hoop_lock):
                finalized = self._finalize_active()
                if self._is_in_finalize_cooldown(point.frame_index):
                    self.post_shot_debug_points.append(point)
                    logger.info(
                        "post-shot continuation suppressed frame=%s",
                        point.frame_index,
                    )
                return finalized

            if (
                hoop_lock is not None
                and hoop_lock.is_locked
                and _near_rim_approach(point, hoop_lock)
            ):
                self.active.post_rim_started = True

            skip_floor = self._should_skip_floor_bounce_point(point, hoop_lock)
            if skip_floor:
                self.active.excluded_debug_points.append(point)
                self.post_shot_debug_points.append(point)
                logger.info(
                    "floor bounce excluded frame=%s y=%.1f",
                    point.frame_index,
                    point.y,
                )
            else:
                for predicted in backfill_gap_points(
                    self.active.candidate_points,
                    point,
                    hoop_lock,
                    self.config,
                ):
                    append_continuity_point(self.active, predicted)
                self.active.candidate_points.append(point)
                append_continuity_point(self.active, point)
                self._update_interior_apex_seen()
                if self.active.post_rim_started and not point.interpolated:
                    self.active.post_rim_measured_count += 1
                    if (
                        self.active.post_rim_measured_count
                        >= self.config.post_rim_measured_cap
                    ):
                        self._pending_finalize_reason = "post_rim"
                        return self._finalize_active()
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
            self._pending_finalize_reason = "end_of_video"
            return self._finalize_active()
        return None

    def _try_start_collection(self) -> None:
        if not self._should_start_collection(self._pre_shot_buffer):
            return

        seed_points = self._pre_shot_buffer[-self.config.min_points_to_start :]
        measured_seeds = self._non_floor_measured(
            seed_points,
            self._last_hoop_lock,
        )
        if len(measured_seeds) < self.config.min_points_to_start:
            return
        self._interpolated_ignored_count = sum(
            1 for point in seed_points if point.interpolated
        )
        release_frame = measured_seeds[0].frame_index
        pickup_candidates = self._non_floor_measured(
            self._pre_shot_buffer,
            self._last_hoop_lock,
        )
        self.active = ShotCandidate(
            start_frame=release_frame,
            state="collecting_shot",
        )
        self.active.pickup_points = [
            point
            for point in pickup_candidates
            if point.frame_index < release_frame
        ]
        self.active.candidate_points.extend(measured_seeds)
        self.active.continuity_points.extend(measured_seeds)
        self._frames_since_point = 0
        self._pre_shot_buffer.clear()
        logger.info(
            "shot started frame=%s seed_points=%s",
            self.active.start_frame,
            len(self.active.candidate_points),
        )

    def _non_floor_measured(
        self,
        points: list[SparseBallDetection],
        hoop_lock: HoopLock | None,
    ) -> list[SparseBallDetection]:
        measured = [point for point in points if not point.interpolated]
        if hoop_lock is None or not hoop_lock.is_locked:
            return measured
        return [
            point
            for point in measured
            if not is_floor_bounce_point(
                point,
                hoop_lock,
                floor_margin_px=self.config.floor_below_rim_margin_px,
            )
        ]

    def _should_start_collection(self, recent_points: list[SparseBallDetection]) -> bool:
        measured = self._non_floor_measured(recent_points, self._last_hoop_lock)
        if len(measured) < self.config.min_points_to_start:
            return False

        latest = measured[-1]
        if self._is_in_finalize_cooldown(latest.frame_index):
            if not self._has_clear_new_release(recent_points):
                return False

        if not self._has_clear_new_release(recent_points):
            return False

        if self._looks_like_rim_reacquisition(
            measured[-self.config.min_points_to_start :]
        ):
            return False

        if not self._ball_in_valid_start_zone(latest):
            return False

        seed_points = measured[-self.config.min_points_to_start :]
        xs = [point.x for point in seed_points]
        ys = [point.y for point in seed_points]
        y_spread = max(ys) - min(ys)
        x_spread = max(xs) - min(xs)
        return y_spread > 8 and x_spread < y_spread * 2.5

    def _ball_in_valid_start_zone(self, point: SparseBallDetection) -> bool:
        if self._last_hoop_lock is not None and self._last_hoop_lock.is_locked:
            rim_y = self._last_hoop_lock.rim_center_y
            return point.y < rim_y + self.config.start_below_rim_margin_px
        upper_body_threshold = self.frame_height * self.config.upper_body_y_ratio
        return point.y < upper_body_threshold

    def _has_clear_new_release(self, recent_points: list[SparseBallDetection]) -> bool:
        measured = self._non_floor_measured(recent_points, self._last_hoop_lock)
        if len(measured) < self.config.min_points_to_start:
            return False

        last_three = measured[-self.config.min_points_to_start :]
        vertical_velocities = []
        for previous, current in zip(last_three, last_three[1:]):
            dt = max(current.frame_index - previous.frame_index, 1)
            vertical_velocities.append((current.y - previous.y) / dt)

        return all(
            velocity < -self.config.upward_velocity_threshold
            for velocity in vertical_velocities
        )

    def _is_in_finalize_cooldown(self, frame_index: int) -> bool:
        if self._cooldown is None:
            return False
        elapsed = frame_index - self._cooldown.frame
        return 0 <= elapsed <= self.config.post_finalize_cooldown_frames

    def _looks_like_rim_reacquisition(self, seed_points: list[SparseBallDetection]) -> bool:
        if self._last_hoop_lock is None or not self._last_hoop_lock.is_locked:
            return False
        if not seed_points:
            return False

        rim_y = self._last_hoop_lock.rim_center_y
        return all(point.y > rim_y - 80.0 for point in seed_points)

    def _should_finalize_on_long_gap(self, point: SparseBallDetection) -> bool:
        assert self.active is not None
        points = self.active.candidate_points
        if len(points) < 2:
            return False

        last_measured = points[-1]
        gap_frames = point.frame_index - last_measured.frame_index
        if gap_frames <= self.config.reacquisition_gap_frames:
            return False

        apex_index = min(range(len(points)), key=lambda index: points[index].y)
        if len(points) >= self.config.min_validated_points_for_fit:
            return apex_index > 0 and apex_index < len(points) - 1

        return apex_index == len(points) - 1

    def _should_skip_floor_bounce_point(
        self,
        point: SparseBallDetection,
        hoop_lock: HoopLock | None,
    ) -> bool:
        assert self.active is not None
        if not (
            self.active.post_rim_started or self.active.interior_apex_seen
        ):
            return False
        return is_floor_bounce_point(
            point,
            hoop_lock,
            floor_margin_px=self.config.floor_below_rim_margin_px,
        )

    def _update_interior_apex_seen(self) -> None:
        assert self.active is not None
        points = self.active.candidate_points
        if len(points) < self.config.min_validated_points_for_fit:
            return
        apex_index = min(range(len(points)), key=lambda index: points[index].y)
        if 0 < apex_index < len(points) - 1:
            self.active.interior_apex_seen = True

    def _should_finalize_after_rim_sequence(
        self,
        point: SparseBallDetection,
        hoop_lock: HoopLock | None,
    ) -> bool:
        assert self.active is not None
        if point.interpolated or hoop_lock is None or not hoop_lock.is_locked:
            return False
        if not self.active.post_rim_started:
            return False

        rim_points = [
            candidate
            for candidate in self.active.candidate_points
            if _near_rim_approach(candidate, hoop_lock)
        ]
        if len(rim_points) < 2:
            return False

        last_rim = rim_points[-1]
        gap_frames = point.frame_index - last_rim.frame_index
        return gap_frames > 3

    def _should_finalize_before_reacquisition(
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

        last_measured = points[-1]
        gap_frames = point.frame_index - last_measured.frame_index
        if gap_frames <= self.config.reacquisition_gap_frames:
            return False

        if hoop_lock is None or not hoop_lock.is_locked:
            return False

        recent_rim_contact = any(
            _near_rim_approach(candidate, hoop_lock) for candidate in points[-3:]
        )
        if not recent_rim_contact:
            return False

        if not self._looks_like_rim_reacquisition([point]):
            return False

        self._pending_finalize_reason = "post_rim"
        return True

    def _is_falling_point(self, point: SparseBallDetection) -> bool:
        measured = [
            candidate
            for candidate in self._pre_shot_buffer
            if not candidate.interpolated and candidate.frame_index < point.frame_index
        ]
        if not measured:
            return False

        previous = measured[-1]
        dt = max(point.frame_index - previous.frame_index, 1)
        vertical_velocity = (point.y - previous.y) / dt
        return vertical_velocity > self.config.upward_velocity_threshold

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

        if points[-1].frame_index == point.frame_index:
            if len(points) < 2:
                return False
            previous = points[-2]
        else:
            previous = points[-1]

        dt = max(point.frame_index - previous.frame_index, 1)
        if (point.y - previous.y) / dt <= 0.5:
            return False

        if (
            hoop_lock is not None
            and hoop_lock.is_locked
            and not point.interpolated
            and _near_rim_approach(point, hoop_lock)
        ):
            self.active.post_rim_started = True
            if self.active.post_rim_frames_remaining == 0:
                self.active.post_rim_frames_remaining = self.config.post_rim_extension_frames
            self.active.post_rim_frames_remaining -= 1
            if self.active.post_rim_frames_remaining <= 0:
                self._pending_finalize_reason = "post_rim"
                return True
            return False

        recent = points[-4:]
        if len(recent) >= 4:
            for index in range(1, len(recent)):
                segment_gap = (
                    recent[index].frame_index - recent[index - 1].frame_index
                )
                if segment_gap > self.config.reacquisition_gap_frames:
                    continue
                horizontal_jump = abs(recent[index].x - recent[index - 1].x)
                if horizontal_jump > self.config.horizontal_jump_end_px:
                    self._pending_finalize_reason = "horizontal_jump"
                    return True

        return False

    def _collect_post_shot_points(self) -> list[SparseBallDetection]:
        assert self.active is not None
        merged: list[SparseBallDetection] = []
        seen: set[int] = set()
        for point in self.post_shot_debug_points + self.active.excluded_debug_points:
            if point.frame_index in seen:
                continue
            merged.append(point)
            seen.add(point.frame_index)
        merged.sort(key=lambda item: item.frame_index)
        return merged[: self.config.post_rim_measured_cap]

    def _finalize_active(self) -> ShotCandidate | None:
        if self.active is None:
            return None

        if len(self.active.candidate_points) < 2:
            logger.info("candidate reset (too few points)")
            self.active = None
            self._frames_since_point = 0
            return None

        self.active.end_frame = self.active.candidate_points[-1].frame_index
        reason = self._pending_finalize_reason or "unknown"
        self._pending_finalize_reason = None
        last_motion = _infer_last_motion(self.active.candidate_points)
        ignored = self._interpolated_ignored_count
        self.active.post_shot_points = self._collect_post_shot_points()
        finalized = finalize_shot(
            self.active,
            self.config,
            interpolated_ignored_count=ignored,
            hoop_lock=self._last_hoop_lock,
            hoop_lock_config=self.hoop_lock_config,
            story_config=self.story_config,
        )
        self.active = None
        self._frames_since_point = 0
        self._interpolated_ignored_count = 0
        self.post_shot_debug_points.clear()
        self._cooldown = FinalizeCooldownState(
            frame=finalized.end_frame or finalized.start_frame,
            reason=reason,
            last_motion=last_motion,
        )
        logger.info(
            "finalize cooldown started frame=%s reason=%s last_motion=%s cooldown=%s",
            self._cooldown.frame,
            self._cooldown.reason,
            self._cooldown.last_motion,
            self.config.post_finalize_cooldown_frames,
        )
        logger.info("candidate reset")

        if finalized.parabola_fit is None and not finalized.insufficient_points_for_fit:
            return None

        if finalized.parabola_fit is not None or finalized.insufficient_points_for_fit:
            self.display_shot = finalized
            self.finalized_shots.append(finalized)
            return finalized
        return None


def _infer_last_motion(points: list[SparseBallDetection]) -> ShotMotionDirection:
    if len(points) < 2:
        return "unknown"

    previous = points[-2]
    latest = points[-1]
    dt = max(latest.frame_index - previous.frame_index, 1)
    vertical_velocity = (latest.y - previous.y) / dt
    if vertical_velocity > 0.5:
        return "descending"
    if vertical_velocity < -0.5:
        return "ascending"
    return "unknown"


def _near_rim_approach(point: SparseBallDetection, hoop_lock: HoopLock) -> bool:
    x1, _y1, x2, _y2 = hoop_lock.bbox_xyxy
    rim_half_width = (x2 - x1) / 2.0
    near_rim_x = abs(point.x - hoop_lock.rim_center_x) <= rim_half_width
    near_rim_y = point.y >= hoop_lock.rim_center_y - 60.0
    return near_rim_x and near_rim_y


def _ball_at_rim(point: SparseBallDetection, hoop_lock: HoopLock) -> bool:
    """True when the ball is at/below rim height and horizontally near the rim anchor."""

    x1, _y1, x2, _y2 = hoop_lock.bbox_xyxy
    rim_half_width = (x2 - x1) / 2.0
    near_rim_x = abs(point.x - hoop_lock.rim_center_x) <= rim_half_width
    at_or_below_rim = point.y >= hoop_lock.rim_center_y
    return near_rim_x and at_or_below_rim
