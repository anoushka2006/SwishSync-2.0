"""End-to-end orchestration for the foundational SwishSync CV pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2

from swishsync_cv.config import PipelineConfig
from swishsync_cv.data import DetectionRecord, FrameDetections, TrajectoryPoint
from swishsync_cv.detection import YoloObjectDetector
from swishsync_cv.io import VideoReader, VideoWriter
from swishsync_cv.tracking import BallTrajectoryTracker
from swishsync_cv.utils import (
    write_detections_csv,
    write_detections_jsonl,
    write_trajectory_json,
)
from swishsync_cv.visualization import annotate_frame


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
    trajectory_json_path: Path
    processed_frames: int
    detection_count: int
    trajectory_point_count: int


def run_pipeline(
    config: PipelineConfig,
    detector: Detector | None = None,
) -> PipelineResult:
    """Run video extraction, detection, tracking, overlay, and export."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.video_output.save_debug_frames:
        config.debug_frames_dir.mkdir(parents=True, exist_ok=True)

    active_detector = detector or YoloObjectDetector(config.detection)
    tracker = BallTrajectoryTracker(
        min_confidence=config.detection.confidence_threshold,
    )

    frame_records: list[FrameDetections] = []
    flat_detections: list[DetectionRecord] = []
    processed_frames = 0

    with VideoReader(config.input_video) as reader:
        if reader.metadata is None:
            raise RuntimeError("Video metadata was not initialized.")

        with VideoWriter(
            output_path=config.output_video_path,
            fps=reader.metadata.fps,
            frame_size=reader.metadata.frame_size,
            codec=config.video_output.codec,
        ) as writer:
            for packet in reader:
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

                tracker.update(
                    frame_index=packet.index,
                    timestamp_ms=packet.timestamp_ms,
                    detections=detections,
                )
                annotated = annotate_frame(
                    frame=packet.image,
                    detections=detections,
                    trajectory=tracker.points,
                    config=config.video_output,
                    frame_index=packet.index,
                )
                writer.write(annotated)

                if _should_save_debug_frame(
                    frame_index=packet.index,
                    save_debug_frames=config.video_output.save_debug_frames,
                    stride=config.video_output.debug_frame_stride,
                ):
                    debug_path = config.debug_frames_dir / f"frame_{packet.index:06d}.jpg"
                    cv2.imwrite(str(debug_path), annotated)

                processed_frames += 1

    write_detections_jsonl(config.detections_jsonl_path, frame_records)
    write_detections_csv(config.detections_csv_path, flat_detections)
    write_trajectory_json(config.trajectory_json_path, tracker.points)

    return PipelineResult(
        output_video_path=config.output_video_path,
        detections_jsonl_path=config.detections_jsonl_path,
        detections_csv_path=config.detections_csv_path,
        trajectory_json_path=config.trajectory_json_path,
        processed_frames=processed_frames,
        detection_count=len(flat_detections),
        trajectory_point_count=len(tracker.points),
    )


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
