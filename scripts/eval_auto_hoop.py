#!/usr/bin/env python3
"""Benchmark automatic hoop lock (custom hoop YOLO weights) against manual bboxes.

Runs ONLY the hoop-lock path on each Testing clip: hoop model on a stride over
the early frames, fed into HoopLockTracker with no manual bbox. Compares the
resulting lock against the manual bboxes recorded in eval_rerun_results.json.

Reported per clip: lock frame, lock confidence, center-x delta, bottom-y (rim
line) delta, and IoU vs the manual bbox. Exits non-zero if any clip fails to
lock. Note: manual bboxes often include backboard/pole, so low IoU with a small
|dx| is expected — dx and rim-line delta are the meaningful metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from swishsync_cv.config import DetectionConfig, HoopLockConfig
from swishsync_cv.detection.yolo import YoloObjectDetector
from swishsync_cv.tracking.hoop_lock import HoopLockTracker
from scripts.evaluate_testing_clips import CLIP_LABELS

TESTING_DIR = ROOT / "videos" / "Testing"
RESULTS_JSON = ROOT / "outputs" / "eval" / "shot_story" / "eval_rerun_results.json"

DETECTION_STRIDE = 3
MAX_FRAMES = 90


def bbox_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def run_clip(
    video_path: Path,
    detector: YoloObjectDetector,
) -> tuple[HoopLockTracker, int | None]:
    tracker = HoopLockTracker(HoopLockConfig())
    lock_frame: int | None = None
    cap = cv2.VideoCapture(str(video_path))
    try:
        for frame_index in range(MAX_FRAMES):
            ok, frame = cap.read()
            if not ok:
                break
            detection_ran = frame_index % DETECTION_STRIDE == 0
            if not tracker.should_process_frame(frame_index, detection_ran):
                continue
            hoops = []
            if detection_ran or not tracker.is_locked:
                hoops = [
                    detection
                    for detection in detector.detect(
                        frame=frame,
                        frame_index=frame_index,
                        timestamp_ms=0.0,
                    )
                    if detection.label == "hoop"
                ]
            tracker.update(
                frame_index=frame_index,
                frame=frame,
                yolo_hoop_detections=hoops,
            )
            if lock_frame is None and tracker.is_locked:
                lock_frame = frame_index
    finally:
        cap.release()
    return tracker, lock_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatic hoop lock benchmark")
    parser.add_argument(
        "--hoop-model",
        default=str(ROOT / "models" / "hoop_ball_yolov8.pt"),
        help="Hoop YOLO weights path.",
    )
    args = parser.parse_args()

    bbox_map = json.loads(RESULTS_JSON.read_text())["bbox_map"]
    detector = YoloObjectDetector(DetectionConfig(model_path=args.hoop_model))

    print(f"{'clip':4} {'locked':>6} {'frame':>5} {'conf':>5} {'dx':>7} {'d_rim_y':>8} {'IoU':>5}")
    failures = []
    for filename, label in sorted(CLIP_LABELS.items(), key=lambda kv: kv[1]):
        video_path = TESTING_DIR / filename
        if not video_path.exists() or label not in bbox_map:
            continue
        tracker, lock_frame = run_clip(video_path, detector)
        lock = tracker.lock
        if lock is None or not lock.is_locked:
            failures.append(label)
            print(f"{label:4} {'NO':>6}")
            continue
        mx, my, mw, mh = bbox_map[label]
        manual = (mx, my, mx + mw, my + mh)
        dx = lock.center_x - (manual[0] + manual[2]) / 2.0
        d_rim_y = lock.bbox_xyxy[3] - manual[3]
        print(
            f"{label:4} {'yes':>6} {lock_frame if lock_frame is not None else '-':>5} "
            f"{lock.confidence:>5.2f} {dx:>7.1f} {d_rim_y:>8.1f} "
            f"{bbox_iou(lock.bbox_xyxy, manual):>5.2f}"
        )

    if failures:
        print(f"\nFAILED to lock: {', '.join(failures)}")
        sys.exit(1)
    print("\nAll clips locked automatically.")


if __name__ == "__main__":
    main()
