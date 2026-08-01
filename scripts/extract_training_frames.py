#!/usr/bin/env python3
"""Sample annotation frames from our Testing clips for hoop/ball training.

Writes JPEGs to datasets/own_clips/images/ with a bias toward rim-interaction
moments (flight end ± 30 frames from existing eval shots.json) plus uniform
coverage. Annotate with rim-only + basketball boxes (e.g. in Roboflow or
labelImg), then merge with the CC-BY Universe datasets listed in
docs/training_plan.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_testing_clips import CLIP_LABELS, slugify

TESTING_DIR = ROOT / "videos" / "Testing"
EVAL_DIR = ROOT / "outputs" / "temp" / "outcome_check"
OUT_DIR = ROOT / "datasets" / "own_clips" / "images"


def sample_indices(total: int, label: str) -> list[int]:
    uniform = list(range(0, total, max(total // 12, 1)))  # ~12 uniform frames
    rim_window: list[int] = []
    shots_path = EVAL_DIR / slugify(
        {v: k for k, v in CLIP_LABELS.items()}[label]
    ) / "shots.json"
    if shots_path.exists():
        shots = json.loads(shots_path.read_text())
        for shot in shots:
            end = shot.get("end_frame")
            if end is None:
                continue
            rim_window.extend(range(max(0, end - 6), min(total, end + 30), 3))
    return sorted(set(uniform + rim_window))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract annotation frames")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for filename, label in sorted(CLIP_LABELS.items(), key=lambda kv: kv[1]):
        video = TESTING_DIR / filename
        if not video.exists():
            continue
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in sample_indices(total, label):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            out = args.out_dir / f"{label}_{index:05d}.jpg"
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            written += 1
        cap.release()
        print(f"{label}: done")
    print(f"wrote {written} frames to {args.out_dir}")


if __name__ == "__main__":
    main()
