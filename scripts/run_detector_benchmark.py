#!/usr/bin/env python3
"""Full A-S benchmark: baseline detector vs new trained weights.

Runs the pipeline under two detector configs, then reports per clip:
shot detected, measured points (recall proxy), trajectory completeness,
weighted RMSE, make/miss verdict + correctness. Flags improvements,
regressions, and any CORE clip whose RMSE got worse.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from swishsync_cv.config import DetectionConfig, PipelineConfig
from swishsync_cv.detection import YoloObjectDetector
from swishsync_cv.pipeline import run_pipeline
from scripts.evaluate_testing_clips import CLIP_LABELS, slugify
from scripts.run_full_eval_rerun import OUTCOME_GROUND_TRUTH, BASELINE_RMSE

TESTING = ROOT / "videos" / "Testing"
CORE = set(BASELINE_RMSE)

# Both configs share the frozen baseline hoop so the ONLY variable between the
# two columns is the ball detector (model + confidence). CORE pins were produced
# with a yolov8n ball; candidate-as-hoop is byte-identical on CORE, so pairing
# the candidate ball with the baseline hoop is the clean single-variable isolation.
BASELINE = dict(ball="yolov8n.pt", hoop=str(ROOT / "models/hoop_ball_yolov8_baseline.pt"))
TRAINED = dict(ball=str(ROOT / "models/hoop_ball_yolov8.pt"),
               hoop=str(ROOT / "models/hoop_ball_yolov8_baseline.pt"))


def _metrics(out_dir: Path) -> dict:
    shots = json.loads((out_dir / "shots.json").read_text())
    if not shots:
        return {"shot": False, "meas": 0, "complete": None, "rmse": None, "verdict": "n/a"}
    s = max(shots, key=lambda x: len(x.get("candidate_points", [])))
    meas = [p for p in s.get("candidate_points", []) if not p.get("interpolated")]
    span = (s["end_frame"] - s["start_frame"] + 1) if s.get("end_frame") and meas else None
    return {
        "shot": not s.get("insufficient_points_for_fit", False),
        "meas": len(meas),
        "complete": (len(meas) / span) if span else None,
        "rmse": (s.get("fit_diagnostics") or {}).get("weighted_residual_rmse"),
        "verdict": (s.get("outcome") or {}).get("verdict") or "unknown",
    }


def _run(
    cfg: dict,
    tag: str,
    only: set[str] | None = None,
    ball_conf: float | None = None,
) -> dict[str, dict]:
    logging.disable(logging.INFO)
    rows = {}
    for filename, label in sorted(CLIP_LABELS.items(), key=lambda kv: kv[1]):
        if only and label not in only:
            continue
        video = TESTING / filename
        if not video.exists():
            continue
        out = ROOT / "outputs" / "temp" / f"bench_{tag}" / slugify(filename)
        detection = DetectionConfig(model_path=cfg["ball"], hoop_model_path=cfg["hoop"])
        ball_detector = None
        if ball_conf is not None:
            # Raise ONLY the ball detector's confidence, and strip its hoop
            # aliases so the candidate's `rim` class never feeds the hoop tracker
            # — the baseline hoop model (built inside run_pipeline at its default
            # 0.25 conf) alone locks the rim. Single-variable CORE isolation.
            ball_detector = YoloObjectDetector(
                replace(
                    detection,
                    confidence_threshold=ball_conf,
                    hoop_model_path=None,
                    hoop_aliases=(),
                )
            )
        run_pipeline(
            PipelineConfig(input_video=video, output_dir=out, detection=detection),
            detector=ball_detector,
        )
        rows[label] = _metrics(out)
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=None,
        help="Confidence threshold for the candidate ball detector only "
        "(hoop detector stays at its default 0.25). Omit to use detector default.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Clip labels to run, e.g. A C P R S (default: all 19).",
    )
    args = parser.parse_args()
    only = {x.upper() for x in args.only} if args.only else None
    new_tag = f"cand{args.ball_conf:.2f}" if args.ball_conf is not None else "trained"

    base = _run(BASELINE, "baseline", only=only)
    new = _run(TRAINED, new_tag, only=only, ball_conf=args.ball_conf)

    def correct(label, m):
        gt = OUTCOME_GROUND_TRUTH.get(label)
        return m["verdict"] == gt if gt and m["verdict"] != "n/a" else None

    print(f"\n{'clip':4} | {'baseline':^28} | {'trained':^28} |")
    print(f"{'':4} | {'shot meas rmse  verdict':^28} | {'shot meas rmse  verdict':^28} |")
    improved, regressed, core_worse, still_fail = [], [], [], []
    for label in sorted(base):
        b, n = base[label], new[label]
        cb, cn = correct(label, b), correct(label, n)

        def cell(m, c):
            rmse = f"{m['rmse']:.2f}" if m["rmse"] is not None else "  - "
            mark = "" if c is None else ("OK" if c else "X")
            return f"{'Y' if m['shot'] else '-':>2} {m['meas']:>4} {rmse:>5} {m['verdict']:>7}{mark:>3}"

        print(f"{label:4} | {cell(b, cb)} | {cell(n, cn)} |")

        # verdict correctness change
        if cb is False and cn is True:
            improved.append(f"{label}(verdict {b['verdict']}->{n['verdict']})")
        if cb is True and cn is False:
            regressed.append(f"{label}(verdict {b['verdict']}->{n['verdict']})")
        # detection improvement (more measured points)
        if n["meas"] > b["meas"] + 2:
            improved.append(f"{label}(+{n['meas']-b['meas']} pts)")
        if b["shot"] and not n["shot"]:
            regressed.append(f"{label}(lost shot)")
        if cn is False or (not n["shot"] and label in OUTCOME_GROUND_TRUTH):
            still_fail.append(label)
        # CORE RMSE regression
        if label in CORE and b["rmse"] is not None and n["rmse"] is not None:
            if n["rmse"] > b["rmse"] + 0.15:
                core_worse.append(f"{label}({b['rmse']:.2f}->{n['rmse']:.2f})")

    def agree(rows):
        ok = tot = 0
        for label, m in rows.items():
            c = correct(label, m)
            if c is not None:
                tot += 1; ok += c
        return f"{ok}/{tot}"

    print(f"\nmake/miss agreement — baseline {agree(base)}  trained {agree(new)}")
    print(f"IMPROVED: {improved or 'none'}")
    print(f"REGRESSED: {regressed or 'none'}")
    print(f"STILL FAILING: {sorted(set(still_fail)) or 'none'}")
    print(f"CORE clips worse: {core_worse or 'NONE'}")


if __name__ == "__main__":
    main()
