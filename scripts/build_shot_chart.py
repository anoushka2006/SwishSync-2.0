"""Build a make/miss shot chart for a calibrated clip.

Shooter court position = midpoint of the ankles from YOLO-pose at each shot's
release frame, projected to court coords via the calibration. Make/miss comes
from the shot's outcome verdict. Renders a half-court chart PNG.

Requires a calibration JSON (scripts/calibrate_court.py) for the clip's camera.
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

from swishsync_cv.court.homography import CourtCalibration
from swishsync_cv.court.shot_chart import ShotMark, render_shot_chart
from swishsync_cv.pose.pose_estimator import PoseEstimator
from swishsync_cv.pose.posture import L_ANKLE, R_ANKLE


def _feet_point(landmarks) -> tuple[float, float] | None:
    la, ra = landmarks.get(L_ANKLE), landmarks.get(R_ANKLE)
    pts = [p for p in (la, ra) if p is not None and (len(p) < 3 or p[2] >= 0.3)]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a make/miss shot chart")
    parser.add_argument("--input", required=True, type=Path, help="Video path")
    parser.add_argument("--shots", required=True, type=Path, help="shots.json")
    parser.add_argument("--calib", required=True, type=Path, help="Calibration JSON")
    parser.add_argument("--out", required=True, type=Path, help="Chart PNG out")
    args = parser.parse_args()

    calib = CourtCalibration.from_json(args.calib)
    shots = json.loads(args.shots.read_text())
    estimator = PoseEstimator()

    marks: list[ShotMark] = []
    cap = cv2.VideoCapture(str(args.input))
    for shot in shots:
        verdict = (shot.get("outcome") or {}).get("verdict")
        if verdict not in ("make", "miss"):
            continue
        rf = (shot.get("story") or {}).get("release_frame") or shot["start_frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, rf)
        ok, frame = cap.read()
        if not ok:
            continue
        landmarks = estimator.landmarks(frame)
        if landmarks is None:
            continue
        feet = _feet_point(landmarks)
        if feet is None:
            continue
        court_x, court_y = calib.project(*feet)
        marks.append(ShotMark(court_x, court_y, make=verdict == "make"))
    cap.release()

    chart = render_shot_chart(marks)
    cv2.imwrite(str(args.out), chart)
    print(f"wrote {args.out} — {len(marks)} shots plotted")


if __name__ == "__main__":
    main()
