#!/usr/bin/env python3
"""Shooting-form posture metrics at each clip's release frame.

Standalone (NOT in the core pipeline — pose runs once per shot, not per frame,
so it never slows detection or touches the fit). Reads the release frame from
each clip's shots.json, runs YOLO-pose, prints elbow/knee/back-bend angles, and
saves one annotated frame per clip under the eval dir.
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

from swishsync_cv.pose.pose_estimator import PoseEstimator
from swishsync_cv.pose.posture import compute_posture
from scripts.evaluate_testing_clips import CLIP_LABELS, slugify

TESTING = ROOT / "videos" / "Testing"
EVAL = ROOT / "outputs" / "temp" / "outcome_check"
_SKELETON = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
             (23, 25), (25, 27), (24, 26), (26, 28), (11, 12), (23, 24)]


def _release_frame(shot: dict) -> int:
    return (shot.get("story") or {}).get("release_frame") or shot["start_frame"]


def _annotate(frame, landmarks, metrics) -> None:
    for a, b in _SKELETON:
        if a in landmarks and b in landmarks:
            pa, pb = landmarks[a], landmarks[b]
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 255, 0), 2)
    for x, y, _c in landmarks.values():
        cv2.circle(frame, (int(x), int(y)), 4, (0, 200, 255), -1)
    lines = [
        f"Elbow: {metrics.elbow_angle_deg:.0f} deg" if metrics.elbow_angle_deg else "Elbow: n/a",
        f"Knee:  {metrics.knee_bend_deg:.0f} deg" if metrics.knee_bend_deg else "Knee: n/a",
        f"Back:  {metrics.back_bend_deg:.0f} deg" if metrics.back_bend_deg else "Back: n/a",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (30, 60 + i * 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (40, 220, 40), 3, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Posture metrics at release")
    parser.add_argument("--eval-dir", type=Path, default=EVAL)
    parser.add_argument("--only", nargs="*", help="Clip labels")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    estimator = PoseEstimator()
    print(f"{'clip':4} {'relF':>5} {'side':>5} {'elbow':>6} {'knee':>6} {'back':>6}")
    for filename, label in sorted(CLIP_LABELS.items(), key=lambda kv: kv[1]):
        if args.only and label not in args.only:
            continue
        shots_path = args.eval_dir / slugify(filename) / "shots.json"
        if not shots_path.exists():
            continue
        shots = json.loads(shots_path.read_text())
        if not shots:
            print(f"{label:4}  (no shot)")
            continue
        shot = max(shots, key=lambda s: len(s.get("candidate_points", [])))
        rf = _release_frame(shot)
        cap = cv2.VideoCapture(str(TESTING / filename))
        cap.set(cv2.CAP_PROP_POS_FRAMES, rf)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        landmarks = estimator.landmarks(frame)
        if landmarks is None:
            print(f"{label:4} {rf:>5}  (no person)")
            continue
        m = compute_posture(landmarks)

        def fmt(v):
            return f"{v:.0f}" if v is not None else "n/a"

        print(f"{label:4} {rf:>5} {m.side:>5} {fmt(m.elbow_angle_deg):>6} "
              f"{fmt(m.knee_bend_deg):>6} {fmt(m.back_bend_deg):>6}")
        if not args.no_save:
            _annotate(frame, landmarks, m)
            out = args.eval_dir / slugify(filename) / "posture_release.jpg"
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
