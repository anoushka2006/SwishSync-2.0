"""End-to-end orchestration for the SwishSync shot reconstruction pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2

from swishsync_cv.config import PipelineConfig
from swishsync_cv.data import DetectionRecord, FrameDetections, SparseBallDetection
from swishsync_cv.detection import YoloObjectDetector
from swishsync_cv.io import VideoReader, VideoWriter
from swishsync_cv.tracking.hoop_lock import HoopLockTracker
from swishsync_cv.tracking.hoop_selection import select_hoop_bbox_interactive
from swishsync_cv.tracking.shot_candidate import ShotCandidateManager
from swishsync_cv.tracking.sparse_detection import SparseBallDetectionBuffer
from swishsync_cv.utils import (
    write_detections_csv,
    write_detections_jsonl,
    write_finalized_shots_json,
)
from swishsync_cv.visualization.dual_pane import compose_dual_pane, render_trajectory_panel
from swishsync_cv.visualization.overlay import render_debug_panel

logger = logging.getLogger("swishsync_cv.pipeline")


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

                shot_manager.update(
                    frame_index=packet.index,
                    point=sparse_point,
                    hoop_lock=hoop_tracker.lock,
                )

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
                )
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
