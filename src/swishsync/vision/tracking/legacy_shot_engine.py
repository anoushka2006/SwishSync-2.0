"""LegacyShotEngine — the WHOLESALE Phase-A wrap of the legacy shot engine.

Rationale (PLATFORM_REFACTOR step 3): ~90% of shot correctness is candidate
selection (detection gates + shot lifecycle), not the fitter. Fitting raw
per-frame detections can never reproduce the legacy output, so the exit gate
(byte-identical ``weighted_residual_rmse`` per CORE clip) is only meetable by
running the legacy sparse-buffer + ShotCandidateManager + finalize_shot path
unchanged. This class reproduces ``run_pipeline``'s per-frame *fit-relevant*
sequence exactly, importing and reusing ``swishsync_cv`` classes verbatim.

Consequence for the Tracker contract: a normal tracker consumes externally-run
per-frame detections. This engine CANNOT — it must control detection itself
(stride gating via ``SparseBallDetectionBuffer.should_detect``, dense/idle
flags, ROI recovery, hoop-model wiring). So ``update()`` IGNORES its
``detections`` argument. That argument is a temporary Phase-A shape artifact of
the shared ``Tracker`` ABC; it dissolves at MP-C when the stages actually split.

Omitted vs ``run_pipeline``: render-only paths (ball trail, rim-crop buffer /
rescan, outcome refinement, dual-pane rendering, video + json writing). Verified
against pipeline.py that none of these touch ``candidate_points``, fit
coefficients, RMSE, or confidence — they read the finalized candidate and mutate
only render-only fields (``rim_zone_points``, ``outcome``). ``finalize_shot``
(story/outcome/trusted-flight/confidence) runs inside ShotCandidateManager
exactly as before.
"""

from __future__ import annotations

from swishsync.core.registry import register_tracker
from swishsync.core.schemas import (
    Event,
    EventType,
    ObjectClass,
    Track,
    TrackState,
)
from swishsync.events.shot import _legacy_fit_to_ir
from swishsync.vision.detection.yolo import YoloDetector
from swishsync.vision.interfaces import Tracker

from swishsync_cv.config import (
    HoopLockConfig,
    PipelineConfig,
    ShotCandidateConfig,
    ShotStoryConfig,
    SparseDetectionConfig,
)
from swishsync_cv.data import ShotCandidate
from swishsync_cv.detection.ball_detection_gates import filter_basketball_detections
from swishsync_cv.detection.ball_roi_search import try_roi_ball_detection
from swishsync_cv.detection.yolo import YoloObjectDetector
from swishsync_cv.tracking.hoop_lock import HoopLockTracker
from swishsync_cv.tracking.shot_candidate import ShotCandidateManager
from swishsync_cv.tracking.sparse_detection import SparseBallDetectionBuffer


@register_tracker("legacy_shot_engine")
class LegacyShotEngine(Tracker):
    """Wholesale wrap: sparse buffer + hoop lock + ShotCandidateManager + fit."""

    def __init__(
        self,
        detector: YoloDetector,
        hoop_weights: str | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        # active_detector in run_pipeline IS a YoloObjectDetector; drive the
        # legacy one directly so stride/detect_crop paths match byte-for-byte.
        self._ball: YoloObjectDetector = detector.legacy
        self._detector = detector

        self._sparse_config: SparseDetectionConfig = (
            config.sparse_detection if config is not None else SparseDetectionConfig()
        )
        self._shot_config: ShotCandidateConfig = (
            config.shot_candidate if config is not None else ShotCandidateConfig()
        )
        self._hoop_config: HoopLockConfig = (
            config.hoop_lock if config is not None else HoopLockConfig()
        )
        self._story_config: ShotStoryConfig = (
            config.video_output.shot_story if config is not None else ShotStoryConfig()
        )

        self._hoop_detector: YoloObjectDetector | None = None
        if hoop_weights is not None:
            from dataclasses import replace

            self._hoop_detector = YoloObjectDetector(
                replace(self._ball.config, model_path=hoop_weights)
            )

        self._sparse_buffer = SparseBallDetectionBuffer(self._sparse_config)
        self._hoop_tracker = HoopLockTracker(self._hoop_config)
        # shot_manager needs frame_height (from the video); lazy-init on the
        # first frame, exactly like run_pipeline does inside the reader context.
        self._shot_manager: ShotCandidateManager | None = None

        self._seen_finalized = 0
        self._finalize_reasons: list[str | None] = []

    # -- Tracker contract ---------------------------------------------------- #
    def update(
        self,
        detections: list | None,  # noqa: ARG002 - IGNORED (see module docstring)
        frame_index: int,
        t_ms: float = 0.0,
        frame=None,
    ) -> None:
        if frame is None:
            raise ValueError(
                "LegacyShotEngine needs raw frame pixels; it controls detection "
                "itself (stride/ROI/hoop). Pass frame=<image>."
            )
        if self._shot_manager is None:
            self._shot_manager = ShotCandidateManager(
                config=self._shot_config,
                frame_height=frame.shape[0],
                hoop_lock_config=self._hoop_config,
                story_config=self._story_config,
            )

        shot_manager = self._shot_manager
        sparse_buffer = self._sparse_buffer
        hoop_tracker = self._hoop_tracker

        # --- detection scheduling (pipeline.py lines 152-225) --------------- #
        collecting = shot_manager.lifecycle_state == "collecting_shot"
        pre_first_shot = (
            shot_manager.lifecycle_state == "idle" and not shot_manager.finalized_shots
        )
        detection_ran = sparse_buffer.should_detect(
            frame_index, dense=collecting, idle=pre_first_shot
        )
        records: list = []

        if detection_ran:
            records = self._ball.detect(
                frame=frame, frame_index=frame_index, timestamp_ms=t_ms
            )
            records = filter_basketball_detections(
                records,
                hoop_lock=hoop_tracker.lock,
                floor_margin_px=self._shot_config.floor_below_rim_margin_px,
                enabled=self._sparse_config.floor_band_rejection_enabled,
            )
            sparse_point = sparse_buffer.extract_ball_detection(
                frame_index=frame_index, timestamp_ms=t_ms, detections=records
            )
            if sparse_point is None:
                tracking_points = shot_manager.tracking_context_points()
                validation_points = shot_manager.tracking_validation_points()
                if tracking_points and hasattr(self._ball, "detect_crop"):
                    roi_point = try_roi_ball_detection(
                        frame=frame,
                        frame_index=frame_index,
                        timestamp_ms=t_ms,
                        recent_points=tracking_points,
                        validation_points=validation_points,
                        detect_crop=self._ball.detect_crop,
                        sparse_config=self._sparse_config,
                        detection_config=self._ball.config,
                        shot_config=self._shot_config,
                    )
                    if roi_point is not None:
                        sparse_point = roi_point
                        sparse_buffer.register_detection(roi_point)
            if sparse_point is None and collecting:
                sparse_point = sparse_buffer.interpolate_at(
                    frame_index=frame_index, timestamp_ms=t_ms
                )
        else:
            sparse_point = sparse_buffer.interpolate_at(
                frame_index=frame_index, timestamp_ms=t_ms
            )

        # --- hoop lock (pipeline.py lines 227-252) -------------------------- #
        if hoop_tracker.should_process_frame(frame_index, detection_ran):
            yolo_hoops = [r for r in records if r.label == "hoop"]
            if self._hoop_detector is not None and (
                not hoop_tracker.is_locked or hoop_tracker.phase == "revalidation"
            ):
                yolo_hoops.extend(
                    r
                    for r in self._hoop_detector.detect(
                        frame=frame, frame_index=frame_index, timestamp_ms=t_ms
                    )
                    if r.label == "hoop"
                )
            hoop_tracker.update(
                frame_index=frame_index,
                frame=frame,
                yolo_hoop_detections=yolo_hoops,
            )

        # --- shot lifecycle (pipeline.py line 282) -------------------------- #
        shot_manager.update(
            frame_index=frame_index,
            point=sparse_point,
            hoop_lock=hoop_tracker.lock,
        )
        self._absorb_finalized()

    def close(self) -> None:
        """Finalize any candidate still open at end-of-stream (== end_of_video)."""

        if self._shot_manager is not None:
            self._shot_manager.finalize()
            self._absorb_finalized()

    # flush() alias for symmetry with other stream-closing APIs.
    flush = close

    def _absorb_finalized(self) -> None:
        assert self._shot_manager is not None
        finalized = self._shot_manager.finalized_shots
        while self._seen_finalized < len(finalized):
            cooldown = self._shot_manager._cooldown  # reason for the just-closed shot
            self._finalize_reasons.append(cooldown.reason if cooldown else None)
            self._seen_finalized += 1

    # -- IR output ----------------------------------------------------------- #
    @property
    def finalized_shots(self) -> list[ShotCandidate]:
        """Raw legacy candidates, finalized order (aligned with events/reasons)."""

        return self._shot_manager.finalized_shots if self._shot_manager else []

    def tracks(self) -> list[Track]:
        tracks: list[Track] = []
        for shot_id, candidate in enumerate(self.finalized_shots):
            states = [
                TrackState(
                    frame=point.frame_index,
                    t_ms=point.timestamp_ms,
                    bbox_xyxy=(point.x - 3, point.y - 3, point.x + 3, point.y + 3),
                    conf=point.confidence,
                    interpolated=point.interpolated,
                )
                for point in candidate.candidate_points
            ]
            tracks.append(Track(track_id=shot_id, cls=ObjectClass.BALL, states=states))
        return tracks

    def finalized_events(self) -> list[Event]:
        events: list[Event] = []
        for shot_id, candidate in enumerate(self.finalized_shots):
            points = candidate.candidate_points
            diagnostics = candidate.fit_diagnostics
            rmse = (
                diagnostics.weighted_residual_rmse if diagnostics is not None else None
            )
            flight_points = (
                diagnostics.fit_point_count
                if diagnostics is not None and diagnostics.fit_point_count is not None
                else len(points)
            )
            confidence = (
                candidate.confidence.overall_confidence
                if candidate.confidence is not None
                else 0.0
            )
            evidence = {
                "weighted_residual_rmse": rmse,
                "flight_points": flight_points,
                "frame_range": [candidate.start_frame, candidate.end_frame],
            }
            reason = (
                self._finalize_reasons[shot_id]
                if shot_id < len(self._finalize_reasons)
                else None
            )
            if reason is not None:
                evidence["finalize_reason"] = reason
            payload = (
                {"parabola_fit": _legacy_fit_to_ir(candidate.parabola_fit)}
                if candidate.parabola_fit is not None
                else {}
            )
            events.append(
                Event(
                    type=EventType.SHOT,
                    t_start_ms=points[0].timestamp_ms if points else 0.0,
                    t_end_ms=points[-1].timestamp_ms if points else 0.0,
                    actors=[shot_id],
                    confidence=confidence,
                    evidence=evidence,
                    payload=payload,
                )
            )
        return events
