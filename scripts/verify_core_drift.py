#!/usr/bin/env python3
"""Compare CORE RMSE pre-PR1 vs PR1 with matched hoop bboxes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRE = Path("/tmp/swishsync-pre-pr1")
OUT = ROOT / "outputs" / "core_drift_verify"

BASELINE_RMSE = {"C": 0.443, "P": 0.91, "R": 0.43, "S": 0.82}
BASELINE_START = {"C": 66, "P": 92, "R": 24, "S": 79}

CLIP_FILES = {
    "C": "IMG_1961.MOV",
    "P": "IMG_2029 7.MOV",
    "R": "IMG_2029 9.MOV",
    "S": "IMG_2029.MOV",
}

BBOXES = {
    "C_clipc_rim": (371.5, 233.0, 122.0, 173.0),
    "C_extracted": (376.0, 240.0, 122.0, 173.0),
    "P_extracted": (1292.0, 198.0, 137.0, 209.0),
    "P_rim_std": (1300.5, 273.0, 120.0, 127.0),
    "R_extracted": (1248.0, 216.0, 147.0, 189.0),
    "R_rim_std": (1261.5, 272.0, 120.0, 127.0),
    "S_extracted": (1249.0, 214.0, 150.0, 191.0),
    "S_rim_std": (1264.0, 272.0, 120.0, 127.0),
}

CLIP_BBOX = {
    "C": ["C_clipc_rim", "C_extracted"],
    "P": ["P_extracted", "P_rim_std"],
    "R": ["R_extracted", "R_rim_std"],
    "S": ["S_extracted", "S_rim_std"],
}


def run_pipeline(code_root: Path, label: str, bbox_name: str, bbox: tuple[float, ...]) -> dict:
    video = ROOT / "videos" / "Testing" / CLIP_FILES[label]
    out = OUT / code_root.name / label / bbox_name
    out.mkdir(parents=True, exist_ok=True)
    script = f"""
import json
from pathlib import Path
import sys
sys.path.insert(0, {str(code_root / "src")!r})
from swishsync_cv.config import HoopLockConfig, PipelineConfig
from swishsync_cv.pipeline import run_pipeline

out = Path({str(out)!r})
config = PipelineConfig(
    input_video=Path({str(video)!r}),
    output_dir=out,
    hoop_lock=HoopLockConfig(manual_bbox_xywh={bbox!r}),
)
run_pipeline(config)
shots = json.loads((out / "shots.json").read_text())
if not shots:
    print(json.dumps({{"shots": 0}}))
else:
    s = shots[0]
    d = s.get("fit_diagnostics") or {{}}
    print(json.dumps({{
        "shots": len(shots),
        "start": s.get("start_frame"),
        "end": s.get("end_frame"),
        "rmse": d.get("weighted_residual_rmse"),
        "measured": len([p for p in s.get("candidate_points", []) if not p.get("interpolated")]),
    }}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(code_root / "src")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {"error": proc.stderr[-800:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    if not PRE.exists():
        subprocess.check_call(["git", "worktree", "add", str(PRE), "HEAD"], cwd=ROOT)

    rows = []
    print("tag,clip,bbox,rmse,start,shots,delta_rmse,delta_start,pr1_minus_pre")
    for label in ["C", "P", "R", "S"]:
        base_rmse = BASELINE_RMSE[label]
        base_start = BASELINE_START[label]
        for bbox_name in CLIP_BBOX[label]:
            bbox = BBOXES[bbox_name]
            results = {}
            for code_root, tag in [(PRE, "pre_pr1"), (ROOT, "pr1")]:
                results[tag] = run_pipeline(code_root, label, bbox_name, bbox)

            pre = results["pre_pr1"]
            pr1 = results["pr1"]
            if "error" in pre or "error" in pr1:
                print(f"ERR {label} {bbox_name}")
                if "error" in pre:
                    print(" pre:", pre["error"][:200])
                if "error" in pr1:
                    print(" pr1:", pr1["error"][:200])
                continue

            pr1_minus_pre = None
            if pre.get("rmse") is not None and pr1.get("rmse") is not None:
                pr1_minus_pre = pr1["rmse"] - pre["rmse"]

            row = {
                "label": label,
                "bbox_name": bbox_name,
                "bbox": bbox,
                "baseline_rmse": base_rmse,
                "baseline_start": base_start,
                "pre_pr1": pre,
                "pr1": pr1,
                "pr1_minus_pre_rmse": pr1_minus_pre,
            }
            rows.append(row)

            def fmt(r: dict) -> str:
                rmse = r.get("rmse")
                return f"{rmse:.3f}" if rmse is not None else "n/a"

            print(
                f"pre_pr1,{label},{bbox_name},{fmt(pre)},{pre.get('start')},{pre.get('shots')},"
                f"{(pre.get('rmse') or 0)-base_rmse:+.3f},{(pre.get('start') or 0)-base_start:+},"
                f"0"
            )
            delta_str = f"{pr1_minus_pre:+.3f}" if pr1_minus_pre is not None else "n/a"
            print(
                f"pr1,{label},{bbox_name},{fmt(pr1)},{pr1.get('start')},{pr1.get('shots')},"
                f"{(pr1.get('rmse') or 0)-base_rmse:+.3f},{(pr1.get('start') or 0)-base_start:+},"
                f"{delta_str}"
            )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
