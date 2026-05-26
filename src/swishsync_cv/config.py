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
class SparseDetectionConfig:
    """Settings for running ball/hoop detection on a stride."""

    detection_stride: int = 3
    min_confidence: float = 0.25


@dataclass(frozen=True)
class HoopLockConfig:
    """Hoop acquisition, hybrid detection, and lock behavior."""

    acquisition_frames: int = 30
    min_acquisition_observations: int = 3
    lock_confidence: float = 0.45
    unlock_confidence_ratio: float = 0.55
    revalidation_miss_frames: int = 6
    max_candidate_center_drift_px: float = 35.0
    smoothing_alpha: float = 0.18
    orange_hue_min: int = 5
    orange_hue_max: int = 25
    orange_sat_min: int = 80
    orange_sat_max: int = 255
    orange_val_min: int = 80
    orange_val_max: int = 255
    min_hoop_width_px: int = 8
    max_hoop_width_ratio: float = 0.40
    min_hoop_height_px: int = 6
    max_hoop_height_ratio: float = 0.45
    min_aspect_ratio: float = 0.6
    max_aspect_ratio: float = 6.0
    upper_frame_ratio: float = 0.85
    manual_bbox_xywh: tuple[float, float, float, float] | None = None
    select_hoop_on_first_frame: bool = False
    select_hoop_if_unlocked: bool = False
    rim_anchor_weight: float = 2.0


@dataclass(frozen=True)
class GapRecoveryConfig:
    """Short-gap parabolic continuity for active shot collection."""

    max_short_gap_fill_frames: int = 6
    min_measured_for_gap_predict: int = 3
    gap_predict_tail_points: int = 5
    synthetic_gap_confidence: float = 0.35
    extend_idle_with_gap_predict: bool = True


@dataclass(frozen=True)
class ShotCandidateConfig:
    """Heuristic thresholds for shot start/end and motion validation."""

    gap_recovery: GapRecoveryConfig = field(default_factory=GapRecoveryConfig)
    min_points_to_start: int = 3
    upward_velocity_threshold: float = 2.0
    upper_body_y_ratio: float = 0.55
    start_below_rim_margin_px: float = 350.0
    max_horizontal_jump_px: float = 120.0
    max_vertical_accel_px: float = 80.0
    max_parabola_deviation_px: float = 45.0
    post_rim_extension_frames: int = 12
    post_rim_measured_cap: int = 5
    max_idle_frames: int = 8
    min_validated_points_for_fit: int = 4
    min_measured_points_for_fit: int = 5
    reacquisition_gap_frames: int = 15
    horizontal_jump_end_px: float = 60.0
    post_finalize_cooldown_frames: int = 25
    floor_below_rim_margin_px: float = 100.0
    rim_anchor_rmse_regression_ratio: float = 1.15
    max_visual_extension_ratio: float = 2.5


@dataclass(frozen=True)
class AnalyticalViewConfig:
    """Right-panel analytical shot-space rendering settings."""

    background_color: tuple[int, int, int] = (24, 24, 28)
    viewport_margin: float = 0.12
    active_opacity: float = 1.0
    completed_opacity: float = 0.45
    active_arc_thickness: int = 3
    completed_arc_thickness: int = 2


@dataclass(frozen=True)
class ShotStoryConfig:
    """Render-only shot lifecycle visualization settings."""

    pickup_max_horizontal_span_px: float = 200.0
    pickup_min_points: int = 2
    pickup_min_frame_span: int = 3
    max_pickup_to_release_frame_gap: int = 6
    max_pickup_to_release_distance_px: float = 90.0
    min_pickup_connector_distance_px: float = 5.0
    pickup_connector_opacity_scale: float = 0.55
    max_flight_diagnostic_dots: int = 12


@dataclass(frozen=True)
class VideoOutputConfig:
    """Video export and visual debugging settings."""

    codec: str = "mp4v"
    draw_confidence: bool = True
    draw_frame_index: bool = True
    save_debug_frames: bool = False
    debug_frame_stride: int = 30
    dual_pane: bool = True
    right_panel_background: tuple[int, int, int] = (24, 24, 28)
    show_shot_history: bool = True
    shot_story: ShotStoryConfig = field(default_factory=ShotStoryConfig)
    analytical_view: AnalyticalViewConfig = field(default_factory=AnalyticalViewConfig)


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level processing config for one local basketball video."""

    input_video: Path
    output_dir: Path = Path("outputs")
    output_video_name: str = "processed.mp4"
    detections_jsonl_name: str = "detections.jsonl"
    detections_csv_name: str = "detections.csv"
    trajectory_json_name: str = "trajectory.json"
    shots_json_name: str = "shots.json"
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    sparse_detection: SparseDetectionConfig = field(default_factory=SparseDetectionConfig)
    hoop_lock: HoopLockConfig = field(default_factory=HoopLockConfig)
    shot_candidate: ShotCandidateConfig = field(default_factory=ShotCandidateConfig)
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
    def shots_json_path(self) -> Path:
        return self.output_dir / self.shots_json_name

    @property
    def debug_frames_dir(self) -> Path:
        return self.output_dir / "debug_frames"
