"""Configuration objects for SwishSync's foundational CV pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DetectionConfig:
    """YOLOv8 detection settings.

    The default aliases work with stock COCO YOLOv8 for basketball-like balls
    via ``sports ball``. Hoop aliases are intended for custom YOLO weights.
    """

    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cpu"
    basketball_aliases: tuple[str, ...] = ("basketball", "ball", "sports ball")
    hoop_aliases: tuple[str, ...] = (
        "hoop",
        "basketball hoop",
        "rim",
        "basketball rim",
        "backboard",
    )


@dataclass(frozen=True)
class VideoOutputConfig:
    """Video export and visual debugging settings."""

    codec: str = "mp4v"
    draw_confidence: bool = True
    draw_frame_index: bool = True
    trajectory_max_points: int | None = None
    save_debug_frames: bool = False
    debug_frame_stride: int = 30


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level processing config for one local basketball video."""

    input_video: Path
    output_dir: Path = Path("outputs")
    output_video_name: str = "processed.mp4"
    detections_jsonl_name: str = "detections.jsonl"
    detections_csv_name: str = "detections.csv"
    trajectory_json_name: str = "trajectory.json"
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    video_output: VideoOutputConfig = field(default_factory=VideoOutputConfig)

    @property
    def output_video_path(self) -> Path:
        return self.output_dir / self.output_video_name

    @property
    def detections_jsonl_path(self) -> Path:
        return self.output_dir / self.detections_jsonl_name

    @property
    def detections_csv_path(self) -> Path:
        return self.output_dir / self.detections_csv_name

    @property
    def trajectory_json_path(self) -> Path:
        return self.output_dir / self.trajectory_json_name

    @property
    def debug_frames_dir(self) -> Path:
        return self.output_dir / "debug_frames"
