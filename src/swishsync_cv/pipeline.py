"""End-to-end orchestration for the SwishSync shot reconstruction pipeline."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import cv2

from swishsync_cv.config import PipelineConfig
from swishsync_cv.data import DetectionRecord, FrameDetections, SparseBallDetection
from swishsync_cv.detection import YoloObjectDetector
from swishsync_cv.detection.ball_detection_gates import filter_basketball_detections
from swishsync_cv.detection.ball_roi_search import try_roi_ball_detection
from swishsync_cv.io import VideoReader, VideoWriter
from swishsync_cv.tracking.hoop_lock import HoopLockTracker
from swishsync_cv.tracking.hoop_selection import select_hoop_bbox_interactive
from swishsync_cv.tracking.shot_candidate import ShotCandidateManager
from swishsync_cv.tracking.shot_outcome import refine_outcome_with_rim_zone
from swishsync_cv.tracking.sparse_detection import SparseBallDetectionBuffer
from swishsync_cv.utils import (
    write_detections_csv,
    write_detections_jsonl,
    write_finalized_shots_json,
)
from swishsync_cv.visualization.dual_pane import compose_dual_pane, render_trajectory_panel
from swishsync_cv.visualization.overlay import draw_posture_overlay, render_debug_panel

logger = logging.getLogger("swishsync_cv.pipeline")

RIM_RESCAN_FRAMES = 48  # post-finalize window re-scanned for the ball at the rim
# (early-finalized shots reach the rim 20-40 frames after finalize)
RIM_RESCAN_BUFFER_FRAMES = 72  # rolling rim-crop buffer for shots that only
# finalize long after flight end (reacquisition wait / end of video)
BALL_TRAIL_WINDOW_SECONDS = 2.5  # render-only rolling ball-dot trail length
BALL_TRAIL_MAXLEN = 240  # bound trail memory during dense collection
RIM_RESCAN_MIN_CONFIDENCE = 0.35  # real balls score 0.85+; static rim clutter ~0.2


class Detector(Protocol):
    """Minimal detector interface used by the pipeline."""

    def detect(
        self,
        frame,
        frame_index: int,
        timestamp_ms: float,
    ) -> list[DetectionRecord]:
        ...


@dataclass(frozen=True)
class PipelineResult:
    """Paths and counts produced by a pipeline run."""

    output_video_path: Path
    detections_jsonl_path: Path
    detections_csv_path: Path
    shots_json_path: Path
    processed_frames: int
    detection_count: int
    sparse_detection_count: int
    shot_count: int


def run_pipeline(
    config: PipelineConfig,
    detector: Detector | None = None,
) -> PipelineResult:
    """Run sparse detection, shot collection, one-shot parabola fit, and export."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.video_output.save_debug_frames:
        config.debug_frames_dir.mkdir(parents=True, exist_ok=True)

    active_detector = detector or YoloObjectDetector(config.detection)
    hoop_detector: YoloObjectDetector | None = None
    if config.detection.hoop_model_path is not None:
        hoop_detector = YoloObjectDetector(
            replace(config.detection, model_path=config.detection.hoop_model_path)
        )
    sparse_buffer = SparseBallDetectionBuffer(config.sparse_detection)
    hoop_tracker = HoopLockTracker(config.hoop_lock)

    if config.hoop_lock.manual_bbox_xywh is not None:
        x, y, width, height = config.hoop_lock.manual_bbox_xywh
        hoop_tracker.lock_manual_bbox(0, x, y, width, height)
        logger.info(
            "manual hoop lock bbox=(%.1f, %.1f, %.1f, %.1f)",
            x,
            y,
            width,
            height,
        )

    frame_records: list[FrameDetections] = []
    flat_detections: list[DetectionRecord] = []
    sparse_detections: list[SparseBallDetection] = []
    processed_frames = 0

    with VideoReader(config.input_video) as reader:
        if reader.metadata is None:
            raise RuntimeError("Video metadata was not initialized.")

        shot_manager = ShotCandidateManager(
            config=config.shot_candidate,
            frame_height=reader.metadata.height,
            hoop_lock_config=config.hoop_lock,
            story_config=config.video_output.shot_story,
        )
        output_frame_size = _output_frame_size(
            reader.metadata.frame_size,
            dual_pane=config.video_output.dual_pane,
        )
        first_frame_image = None
        hoop_selection_attempted = config.hoop_lock.manual_bbox_xywh is not None
        rescan_candidate = None
        rescan_frames_left = 0
        seen_finalized = 0
        rim_crop_buffer: deque = deque(maxlen=RIM_RESCAN_BUFFER_FRAMES)
        pose_estimator = None
        last_posture: tuple = (None, None)
        if config.video_output.draw_posture:
            from swishsync_cv.pose.pose_estimator import PoseEstimator
            from swishsync_cv.pose.posture import compute_posture  # noqa: F401

            pose_estimator = PoseEstimator()
        # render-only ball-dot trail (dribbles + pre/post-shot); bypasses the
        # floor gate so low dribbles show, never feeds the shot fit
        ball_trail: deque = deque(maxlen=BALL_TRAIL_MAXLEN)
        trail_window_frames = int(reader.metadata.fps * BALL_TRAIL_WINDOW_SECONDS)

        with VideoWriter(
            output_path=config.output_video_path,
            fps=reader.metadata.fps,
            frame_size=output_frame_size,
            codec=config.video_output.codec,
        ) as writer:
            for packet in reader:
                if first_frame_image is None:
                    first_frame_image = packet.image.copy()

                if (
                    config.hoop_lock.select_hoop_on_first_frame
                    and packet.index == 0
                    and not hoop_tracker.is_locked
                ):
                    selected = select_hoop_bbox_interactive(first_frame_image)
                    if selected is not None:
                        hoop_tracker.lock_manual_bbox(0, *selected)
                        logger.info("interactive hoop lock applied from first frame")
                    hoop_selection_attempted = True

                collecting = shot_manager.lifecycle_state == "collecting_shot"
                pre_first_shot = (
                    shot_manager.lifecycle_state == "idle"
                    and not shot_manager.finalized_shots
                )
                detection_ran = sparse_buffer.should_detect(
                    packet.index,
                    dense=collecting,
                    idle=pre_first_shot,
                )
                detections: list[DetectionRecord] = []

                if detection_ran:
                    detections = active_detector.detect(
                        frame=packet.image,
                        frame_index=packet.index,
                        timestamp_ms=packet.timestamp_ms,
                    )
                    # capture the raw ball center for the trail BEFORE the floor
                    # gate drops low dribbles
                    _append_ball_trail(ball_trail, detections, packet.index)
                    detections = filter_basketball_detections(
                        detections,
                        hoop_lock=hoop_tracker.lock,
                        floor_margin_px=config.shot_candidate.floor_below_rim_margin_px,
                        enabled=config.sparse_detection.floor_band_rejection_enabled,
                    )
                    frame_records.append(
                        FrameDetections(
                            frame_index=packet.index,
                            timestamp_ms=packet.timestamp_ms,
                            detections=detections,
                        )
                    )
                    flat_detections.extend(detections)

                    sparse_point = sparse_buffer.extract_ball_detection(
                        frame_index=packet.index,
                        timestamp_ms=packet.timestamp_ms,
                        detections=detections,
                    )
                    if sparse_point is None:
                        tracking_points = shot_manager.tracking_context_points()
                        validation_points = shot_manager.tracking_validation_points()
                        if tracking_points and hasattr(active_detector, "detect_crop"):
                            roi_point = try_roi_ball_detection(
                                frame=packet.image,
                                frame_index=packet.index,
                                timestamp_ms=packet.timestamp_ms,
                                recent_points=tracking_points,
                                validation_points=validation_points,
                                detect_crop=active_detector.detect_crop,
                                sparse_config=config.sparse_detection,
                                detection_config=config.detection,
                                shot_config=config.shot_candidate,
                            )
                            if roi_point is not None:
                                sparse_point = roi_point
                                sparse_buffer.register_detection(roi_point)
                                logger.info(
                                    "ROI ball recovery frame=%s conf=%.2f",
                                    packet.index,
                                    roi_point.confidence,
                                )
                    if sparse_point is None and collecting:
                        sparse_point = sparse_buffer.interpolate_at(
                            frame_index=packet.index,
                            timestamp_ms=packet.timestamp_ms,
                        )
                else:
                    sparse_point = sparse_buffer.interpolate_at(
                        frame_index=packet.index,
                        timestamp_ms=packet.timestamp_ms,
                    )

                if hoop_tracker.should_process_frame(packet.index, detection_ran):
                    yolo_hoops = [
                        detection
                        for detection in detections
                        if detection.label == "hoop"
                    ]
                    # ponytail: hoop model only runs pre-lock/revalidation; once
                    # locked, existing color+geometry revalidation carries it.
                    if hoop_detector is not None and (
                        not hoop_tracker.is_locked
                        or hoop_tracker.phase == "revalidation"
                    ):
                        yolo_hoops.extend(
                            detection
                            for detection in hoop_detector.detect(
                                frame=packet.image,
                                frame_index=packet.index,
                                timestamp_ms=packet.timestamp_ms,
                            )
                            if detection.label == "hoop"
                        )
                    hoop_tracker.update(
                        frame_index=packet.index,
                        frame=packet.image,
                        yolo_hoop_detections=yolo_hoops,
                    )

                if (
                    config.hoop_lock.select_hoop_if_unlocked
                    and not hoop_selection_attempted
                    and not hoop_tracker.is_locked
                    and packet.index >= config.hoop_lock.acquisition_frames - 1
                    and first_frame_image is not None
                ):
                    selected = select_hoop_bbox_interactive(
                        first_frame_image,
                        window_title="Select hoop (auto-detection failed)",
                    )
                    if selected is not None:
                        hoop_tracker.lock_manual_bbox(0, *selected)
                        logger.info("interactive hoop lock applied after failed acquisition")
                    hoop_selection_attempted = True

                if sparse_point is not None and not sparse_point.interpolated:
                    sparse_detections.append(sparse_point)

                if hoop_tracker.lock is not None:
                    _buffer_rim_crop(
                        rim_crop_buffer,
                        frame=packet.image,
                        frame_index=packet.index,
                        timestamp_ms=packet.timestamp_ms,
                        hoop_lock=hoop_tracker.lock,
                    )

                shot_manager.update(
                    frame_index=packet.index,
                    point=sparse_point,
                    hoop_lock=hoop_tracker.lock,
                )
                if len(shot_manager.finalized_shots) > seen_finalized:
                    seen_finalized = len(shot_manager.finalized_shots)
                    rescan_candidate = shot_manager.finalized_shots[-1]
                    rescan_frames_left = RIM_RESCAN_FRAMES
                    # candidates can linger long past flight end (e.g. waiting
                    # for reacquisition), so scan the buffered crops backward
                    # to the flight end before continuing live
                    _rescan_buffered_crops(
                        candidate=rescan_candidate,
                        buffer=rim_crop_buffer,
                        detector=hoop_detector or active_detector,
                    )
                    # settle the verdict now from buffered evidence (the ball
                    # often already passed the rim before finalize) so the arc
                    # colours correctly immediately, not 48 frames later
                    _finish_rim_rescan(rescan_candidate, hoop_tracker.lock)

                elif rescan_frames_left > 0 and rescan_candidate is not None:
                    _collect_rim_zone_point(
                        candidate=rescan_candidate,
                        frame=packet.image,
                        frame_index=packet.index,
                        timestamp_ms=packet.timestamp_ms,
                        hoop_lock=hoop_tracker.lock,
                        detector=hoop_detector or active_detector,
                    )
                    # re-settle every frame so the arc colour flips the moment
                    # the decisive rim crossing lands, not at window end
                    _finish_rim_rescan(rescan_candidate, hoop_tracker.lock)
                if rescan_frames_left > 0 and rescan_candidate is not None:
                    rescan_frames_left -= 1
                    if rescan_frames_left == 0:
                        rescan_candidate = None

                left_panel = render_debug_panel(
                    frame=packet.image,
                    detections=detections,
                    hoop_lock=hoop_tracker.lock,
                    sparse_point=sparse_point,
                    config=config.video_output,
                    frame_index=packet.index,
                    detection_ran=detection_ran,
                    hoop_phase=hoop_tracker.phase,
                    collecting_shot=shot_manager.active,
                    lifecycle_state=shot_manager.lifecycle_state,
                    candidate_point_count=shot_manager.candidate_point_count,
                    display_shot=shot_manager.display_shot,
                    preview_pickup_points=shot_manager.preview_pickup_points(),
                    ball_trail=[
                        (fx, x, y)
                        for (fx, x, y) in ball_trail
                        if packet.index - fx <= trail_window_frames
                    ],
                )
                if pose_estimator is not None:
                    if packet.index % config.video_output.posture_stride == 0:
                        landmarks = pose_estimator.landmarks(packet.image)
                        posture = (
                            compute_posture(landmarks) if landmarks is not None else None
                        )
                        last_posture = (landmarks, posture)
                    draw_posture_overlay(left_panel, *last_posture)
                right_panel = render_trajectory_panel(
                    frame_size=reader.metadata.frame_size,
                    collecting_shot=shot_manager.active,
                    display_shot=shot_manager.display_shot,
                    hoop_lock=hoop_tracker.lock,
                    lifecycle_state=shot_manager.lifecycle_state,
                    candidate_point_count=shot_manager.candidate_point_count,
                    background_color=config.video_output.right_panel_background,
                    finalized_shots=shot_manager.finalized_shots,
                    video_config=config.video_output,
                    preview_pickup_points=shot_manager.preview_pickup_points(),
                    frame_index=packet.index,
                )
                output_frame = (
                    compose_dual_pane(left_panel, right_panel)
                    if config.video_output.dual_pane
                    else left_panel
                )
                writer.write(output_frame)

                if _should_save_debug_frame(
                    frame_index=packet.index,
                    save_debug_frames=config.video_output.save_debug_frames,
                    stride=config.video_output.debug_frame_stride,
                ):
                    debug_path = config.debug_frames_dir / f"frame_{packet.index:06d}.jpg"
                    cv2.imwrite(str(debug_path), output_frame)

                processed_frames += 1

        shot_manager.finalize()
        if len(shot_manager.finalized_shots) > seen_finalized:
            # candidate closed only at end of stream: scan buffered crops
            end_candidate = shot_manager.finalized_shots[-1]
            _rescan_buffered_crops(
                candidate=end_candidate,
                buffer=rim_crop_buffer,
                detector=hoop_detector or active_detector,
            )
            _finish_rim_rescan(end_candidate, hoop_tracker.lock)
        if rescan_candidate is not None and rescan_frames_left > 0:
            _finish_rim_rescan(rescan_candidate, hoop_tracker.lock)

    write_detections_jsonl(config.detections_jsonl_path, frame_records)
    write_detections_csv(config.detections_csv_path, flat_detections)
    write_finalized_shots_json(config.shots_json_path, shot_manager.finalized_shots)

    return PipelineResult(
        output_video_path=config.output_video_path,
        detections_jsonl_path=config.detections_jsonl_path,
        detections_csv_path=config.detections_csv_path,
        shots_json_path=config.shots_json_path,
        processed_frames=processed_frames,
        detection_count=len(flat_detections),
        sparse_detection_count=len(sparse_detections),
        shot_count=len(shot_manager.finalized_shots),
    )


def _append_ball_trail(ball_trail, detections, frame_index: int) -> None:
    """Record the highest-confidence raw ball center for the render-only trail."""

    balls = [d for d in detections if d.label == "basketball"]
    if not balls:
        return
    best = max(balls, key=lambda d: d.confidence)
    cx, cy = best.center
    ball_trail.append((frame_index, cx, cy))


def _rim_crop_bounds(hoop_lock, frame_shape) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = hoop_lock.bbox_xyxy
    width = max(x2 - x1, 40.0)
    center_x = (x1 + x2) / 2.0
    frame_height, frame_width = frame_shape[:2]
    return (
        int(max(0, center_x - width * 1.6)),
        int(max(0, y1 - width * 0.8)),
        int(min(frame_width, center_x + width * 1.6)),
        int(min(frame_height, y2 + width * 2.2)),
    )


def _buffer_rim_crop(buffer, frame, frame_index: int, timestamp_ms: float, hoop_lock) -> None:
    left, top, right, bottom = _rim_crop_bounds(hoop_lock, frame.shape)
    if right - left < 4 or bottom - top < 4:
        return
    buffer.append(
        (frame_index, timestamp_ms, frame[top:bottom, left:right].copy(), (left, top))
    )


def _record_rim_zone_ball(candidate, records, frame_index, timestamp_ms) -> None:
    balls = [record for record in records if record.label == "basketball"]
    if not balls:
        return
    best = max(balls, key=lambda record: record.confidence)
    bx1, by1, bx2, by2 = best.bbox_xyxy
    candidate.rim_zone_points.append(
        SparseBallDetection(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            x=(bx1 + bx2) / 2.0,
            y=(by1 + by2) / 2.0,
            confidence=best.confidence,
            interpolated=False,
        )
    )


def _collect_rim_zone_point(
    candidate,
    frame,
    frame_index: int,
    timestamp_ms: float,
    hoop_lock,
    detector,
) -> None:
    """Detect the ball in a rim-centered crop and record it for outcome refinement."""

    if hoop_lock is None or not hasattr(detector, "detect_crop"):
        return
    crop = _rim_crop_bounds(hoop_lock, frame.shape)
    records = detector.detect_crop(
        frame=frame,
        crop_xyxy=crop,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        min_confidence=RIM_RESCAN_MIN_CONFIDENCE,
    )
    _record_rim_zone_ball(candidate, records, frame_index, timestamp_ms)


def _rescan_buffered_crops(candidate, buffer, detector) -> None:
    """Run ball detection over buffered rim crops from after the flight end."""

    if not hasattr(detector, "detect_crop") or candidate.end_frame is None:
        return
    seen_frames = {point.frame_index for point in candidate.rim_zone_points}
    for frame_index, timestamp_ms, crop, (left, top) in buffer:
        if frame_index <= candidate.end_frame or frame_index in seen_frames:
            continue
        height, width = crop.shape[:2]
        records = detector.detect_crop(
            frame=crop,
            crop_xyxy=(0, 0, width, height),
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            min_confidence=RIM_RESCAN_MIN_CONFIDENCE,
        )
        shifted = [
            replace(
                record,
                bbox_xyxy=(
                    record.bbox_xyxy[0] + left,
                    record.bbox_xyxy[1] + top,
                    record.bbox_xyxy[2] + left,
                    record.bbox_xyxy[3] + top,
                ),
            )
            for record in records
        ]
        _record_rim_zone_ball(candidate, shifted, frame_index, timestamp_ms)
    candidate.rim_zone_points.sort(key=lambda point: point.frame_index)


def _finish_rim_rescan(candidate, hoop_lock) -> None:
    refined = refine_outcome_with_rim_zone(
        candidate.outcome,
        candidate.rim_zone_points,
        hoop_lock,
        candidate.parabola_fit,
    )
    if candidate.outcome is None or refined != candidate.outcome:
        logger.info(
            "rim rescan verdict=%s method=%s rim_zone_points=%d",
            refined.verdict,
            refined.method,
            len(candidate.rim_zone_points),
        )
    candidate.outcome = refined


def _output_frame_size(
    input_frame_size: tuple[int, int],
    dual_pane: bool,
) -> tuple[int, int]:
    width, height = input_frame_size
    if dual_pane:
        return (width * 2, height)
    return input_frame_size


def _should_save_debug_frame(
    frame_index: int,
    save_debug_frames: bool,
    stride: int,
) -> bool:
    if not save_debug_frames:
        return False
    if stride <= 0:
        raise ValueError("debug_frame_stride must be positive when saving debug frames.")
    return frame_index % stride == 0
