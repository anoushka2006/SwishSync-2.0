#!/usr/bin/env python3
"""Confirm-step court calibration: click known court points on the first frame.

Per camera setup (static camera), run once. Click >=4 court landmarks whose
real court coordinates (feet, hoop-origin convention) you know, confirm each,
and save a calibration JSON for use by build_shot_chart.py.

Deliberately NOT fully automatic: auto line-detection can misfire, and NEX
Team's automatic leg-projection court mapping is patented — a human confirm
step is both safer and licensing-clean.

Example landmarks (hoop-origin, +x to half-court, +y to camera-right):
  hoop floor point        -> (0, 0)
  lane corners            -> (0, +8) and (0, -8)
  free-throw line corners -> (19, +8) and (19, -8)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from swishsync_cv.court.homography import CourtCalibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive court calibration")
    parser.add_argument("--input", required=True, type=Path, help="Video path")
    parser.add_argument("--out", required=True, type=Path, help="Calibration JSON out")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.input))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read first frame")

    clicks: list[tuple[float, float]] = []

    def on_click(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((float(x), float(y)))
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
            cv2.imshow("calibrate", frame)

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_click)
    cv2.imshow("calibrate", frame)
    print("Click >=4 known court points, then press any key. Order matters —")
    print("enter their court (x,y) feet when prompted.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(clicks) < 4:
        raise SystemExit(f"need >=4 points, got {len(clicks)}")

    court_points = []
    for i, (px, py) in enumerate(clicks):
        raw = input(f"court coords (feet) for click {i} at px({px:.0f},{py:.0f}) as 'x,y': ")
        x_str, y_str = raw.split(",")
        court_points.append((float(x_str), float(y_str)))

    calib = CourtCalibration(
        image_points=tuple(clicks),
        court_points=tuple(court_points),
    )
    calib.to_json(args.out)
    print(f"wrote {args.out} ({len(clicks)} points)")


if __name__ == "__main__":
    main()
