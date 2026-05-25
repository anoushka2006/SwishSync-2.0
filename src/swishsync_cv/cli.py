"""Command-line interface for running the SwishSync CV pipeline locally."""

from __future__ import annotations

import argparse
from pathlib import Path

from swishsync_cv.config import (
    DetectionConfig,
    HoopLockConfig,
    PipelineConfig,
    ShotCandidateConfig,
    SparseDetectionConfig,
    VideoOutputConfig,
)
from swishsync_cv.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SwishSync's sparse detection -> shot reconstruction -> "
            "dual-pane visualization pipeline."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Local input video path.")
    parser.add_argument(
        "--output-dir",
        default=Path("outputs"),
        type=Path,
        help="Directory where processed video and debug artifacts are written.",
    )
    parser.add_argument(
        "--output-video-name",
        default="processed.mp4",
        help="Output video filename inside --output-dir.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 model path/name. Use custom weights for hoop detection.",
    )
    parser.add_argument(
        "--confidence",
        default=0.25,
        type=float,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        default=0.45,
        type=float,
        help="YOLO non-max suppression IOU threshold.",
    )
    parser.add_argument(
        "--detection-stride",
        default=3,
        type=int,
        help="Run YOLO every N frames (sparse detection).",
    )
    parser.add_argument(
        "--no-dual-pane",
        action="store_true",
        help="Disable dual-pane output and write debug panel only.",
    )
    parser.add_argument(
        "--save-debug-frames",
        action="store_true",
        help="Save periodic annotated JPEG frames for inspection.",
    )
    parser.add_argument(
        "--debug-frame-stride",
        default=30,
        type=int,
        help="Save one debug frame every N frames when --save-debug-frames is set.",
    )
    parser.add_argument(
        "--hoop-bbox",
        type=str,
        help="Manual hoop bbox as x,y,w,h in pixels. Locks hoop immediately.",
    )
    parser.add_argument(
        "--select-hoop-on-first-frame",
        action="store_true",
        help="Prompt to draw a hoop bbox on the first frame when no manual bbox is set.",
    )
    parser.add_argument(
        "--select-hoop-if-unlocked",
        action="store_true",
        help="Prompt to draw a hoop bbox if automatic hoop lock fails during acquisition.",
    )
    return parser


def _parse_manual_hoop_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    from swishsync_cv.tracking.hoop_geometry import parse_hoop_bbox_arg

    return parse_hoop_bbox_arg(value)


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig(
        input_video=args.input,
        output_dir=args.output_dir,
        output_video_name=args.output_video_name,
        detection=DetectionConfig(
            model_path=args.model,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            device="cpu",
        ),
        sparse_detection=SparseDetectionConfig(
            detection_stride=args.detection_stride,
            min_confidence=args.confidence,
        ),
        hoop_lock=HoopLockConfig(
            manual_bbox_xywh=_parse_manual_hoop_bbox(args.hoop_bbox),
            select_hoop_on_first_frame=args.select_hoop_on_first_frame,
            select_hoop_if_unlocked=args.select_hoop_if_unlocked,
        ),
        shot_candidate=ShotCandidateConfig(),
        video_output=VideoOutputConfig(
            dual_pane=not args.no_dual_pane,
            save_debug_frames=args.save_debug_frames,
            debug_frame_stride=args.debug_frame_stride,
        ),
    )
    result = run_pipeline(config)
    print("SwishSync CV pipeline complete")
    print(f"processed_frames={result.processed_frames}")
    print(f"detection_count={result.detection_count}")
    print(f"sparse_detection_count={result.sparse_detection_count}")
    print(f"shot_count={result.shot_count}")
    print(f"output_video={result.output_video_path}")
    print(f"detections_jsonl={result.detections_jsonl_path}")
    print(f"detections_csv={result.detections_csv_path}")
    print(f"shots_json={result.shots_json_path}")


if __name__ == "__main__":
    main()
