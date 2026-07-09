#!/usr/bin/env python3
"""Upload frames to the Roboflow project for annotation (data flywheel).

Two modes feed the own-weights training set (task #2):

  --own-clips   upload the sampled own-clip frames (datasets/own_clips/images/)
  --low-conf DIR  upload frames from a processed run where ball/hoop detection
                  was weak — the active-learning signal (far-court misses).

Reads the key from env ROBOFLOW_API_KEY (never hard-code it). Target project
defaults to the forked CC-BY set in the basketball-strategy workspace.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = "basketball-strategy/cv-cnfd4-eaond"


def main() -> int:
    parser = argparse.ArgumentParser(description="Roboflow annotation uploader")
    parser.add_argument("--own-clips", action="store_true")
    parser.add_argument("--low-conf", type=Path, help="Dir of frames to upload")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--batch", default="swishsync-own", help="Roboflow batch name")
    args = parser.parse_args()

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("Set ROBOFLOW_API_KEY env var first.", file=sys.stderr)
        return 2

    if args.own_clips:
        src = ROOT / "datasets" / "own_clips" / "images"
    elif args.low_conf:
        src = args.low_conf
    else:
        print("Pass --own-clips or --low-conf DIR", file=sys.stderr)
        return 2

    images = sorted(src.glob("*.jpg"))
    if not images:
        print(f"No .jpg frames in {src}", file=sys.stderr)
        return 1

    # REST upload — deliberately NO roboflow SDK (its install clobbered opencv
    # once; see DECISION_LOG). requests ships with ultralytics. Roboflow
    # dedupes by content hash, so reruns are safe/resumable.
    import requests

    project_slug = args.project.split("/", 1)[1]
    url = (
        f"https://api.roboflow.com/dataset/{project_slug}/upload"
        f"?api_key={key}&batch={args.batch}"
    )
    ok = failed = 0
    for image in images:
        with open(image, "rb") as fh:
            response = requests.post(url, files={"file": (image.name, fh)}, timeout=60)
        body = response.json() if response.ok else {}
        if body.get("success") or body.get("duplicate"):
            ok += 1
        else:
            failed += 1
            print(f"FAIL {image.name}: {response.text[:160]}", file=sys.stderr)
    print(f"\n{ok} uploaded / {failed} failed → {args.project} (batch '{args.batch}'). "
          "Annotate rim-only + ball.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
