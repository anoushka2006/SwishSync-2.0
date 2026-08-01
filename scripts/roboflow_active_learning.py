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

    try:
        from roboflow import Roboflow
    except ImportError:
        print("pip install roboflow", file=sys.stderr)
        return 2

    workspace, project_slug = args.project.split("/", 1)
    rf = Roboflow(api_key=key)
    project = rf.workspace(workspace).project(project_slug)
    for image in images:
        project.upload(str(image), batch_name=args.batch)
        print(f"uploaded {image.name}")
    print(f"\n{len(images)} frames → {args.project} (batch '{args.batch}'). Annotate rim-only + ball.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
