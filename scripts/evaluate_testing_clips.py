#!/usr/bin/env python3
"""Batch runner and metrics extractor for videos/Testing/ evaluation."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTING_DIR = ROOT / "videos" / "Testing"
EVAL_DIR = ROOT / "outputs" / "eval"

CLIP_LABELS = {
    "IMG_1962.MOV": "A",
    "IMG_1963.MOV": "B",
    "IMG_1961.MOV": "C",
    "IMG_2027 2.MOV": "D",
    "IMG_2027 3.MOV": "E",
    "IMG_2027 4.MOV": "F",
    "IMG_2027 6.MOV": "G",
    "IMG_2027.MOV": "H",
    "IMG_2028 2.mov": "I",
    "IMG_2028 3.mov": "J",
    "IMG_2028.mov": "K",
    "IMG_2029 11.MOV": "L",
    "IMG_2029 3.MOV": "M",
    "IMG_2029 4.MOV": "N",
    "IMG_2029 6.MOV": "O",
    "IMG_2029 7.MOV": "P",
    "IMG_2029 8.MOV": "Q",
    "IMG_2029 9.MOV": "R",
    "IMG_2029.MOV": "S",
}


def slugify(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^\w\-]+", "_", stem).strip("_")


def processed_video_name(label: str) -> str:
    return f"processed_{label.lower()}.mp4"


def run_clip(video_path: Path, output_dir: Path, label: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "swishsync_cv.cli",
        "--input",
        str(video_path),
        "--output-dir",
        str(output_dir),
        "--output-video-name",
        processed_video_name(label),
        "--select-hoop-on-first-frame",
    ]
    print(f"\n>>> Running: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def frame_gaps(points: list[dict]) -> list[int]:
    frames = [p["frame_index"] for p in points if not p.get("interpolated", False)]
    if len(frames) < 2:
        return []
    return [frames[i + 1] - frames[i] for i in range(len(frames) - 1)]


def analyze_shot(shot: dict) -> dict:
    points = shot.get("candidate_points", [])
    measured = [p for p in points if not p.get("interpolated", False)]
    fit = shot.get("parabola_fit")
    diag = shot.get("fit_diagnostics") or {}
    conf = shot.get("confidence") or {}
    arc = shot.get("arc_render") or {}
    excluded = shot.get("excluded_debug_points", [])

    gaps = frame_gaps(measured)
    max_gap = max(gaps) if gaps else 0
    ball_lost = max_gap > 15

    bounce_pollution = len(excluded) > 0 or (
        len(measured) >= 5
        and measured[-1]["y"] > min(p["y"] for p in measured) + 80
    )

    start_late = len(measured) >= 3 and measured[0]["y"] < measured[1]["y"] - 40
    end_early = (
        fit is not None
        and len(measured) >= 3
        and measured[-1]["y"] > min(p["y"] for p in measured) + 25
    )

    insufficient = shot.get("insufficient_points_for_fit", False)
    rmse = diag.get("weighted_residual_rmse")
    traj_conf = conf.get("trajectory_confidence")
    overall = conf.get("overall_confidence")

    if insufficient or fit is None:
        traj_usable = 1 if len(measured) >= 3 else 1
    elif rmse is not None and rmse <= 1.0 and (traj_conf or 0) >= 0.9:
        traj_usable = 5
    elif rmse is not None and rmse <= 2.0 and (traj_conf or 0) >= 0.8:
        traj_usable = 4
    elif rmse is not None and (traj_conf or 0) >= 0.6:
        traj_usable = 3
    else:
        traj_usable = 2

    if bounce_pollution and traj_usable > 2:
        traj_usable = min(traj_usable, 3)
    if ball_lost and traj_usable > 2:
        traj_usable = min(traj_usable, 2)

    hoop_usable = arc.get("rim_center") is not None

    notes: list[str] = []
    if insufficient:
        notes.append(f"insufficient fit ({len(measured)} pts)")
    if bounce_pollution:
        notes.append("bounce/floor points in segment")
    if ball_lost:
        notes.append(f"mid-flight gap {max_gap}f")
    if excluded:
        notes.append(f"{len(excluded)} excluded debug pts")
    if start_late:
        notes.append("possible late start")
    if end_early:
        notes.append("ends before apex descent")

    return {
        "shot_detected": len(measured) >= 2,
        "trajectory_usable": traj_usable,
        "hoop_lock_usable": hoop_usable,
        "ball_lost_mid_flight": ball_lost,
        "bounce_floor_pollution": bounce_pollution,
        "shot_starts_too_late": start_late,
        "shot_ends_too_early": end_early,
        "overall_confidence_pct": round((overall or 0) * 100, 1) if overall else None,
        "trajectory_confidence_pct": round((traj_conf or 0) * 100, 1) if traj_conf else None,
        "weighted_rmse": round(rmse, 2) if rmse is not None else None,
        "fit_points": diag.get("fit_point_count") or len(measured),
        "start_frame": shot.get("start_frame"),
        "end_frame": shot.get("end_frame"),
        "notes": "; ".join(notes) if notes else "clean",
    }


def analyze_clip(output_dir: Path, filename: str, label: str) -> dict:
    shots_path = output_dir / "shots.json"
    processed_path = output_dir / processed_video_name(label)
    if not processed_path.exists():
        processed_path = output_dir / "processed.mp4"
    processed_exists = processed_path.exists()
    row = {
        "label": label,
        "filename": filename,
        "output_dir": str(output_dir.relative_to(ROOT)),
        "shot_detected": False,
        "num_shots": 0,
        "trajectory_usable": 1,
        "hoop_lock_usable": processed_exists,
        "ball_lost_mid_flight": False,
        "bounce_floor_pollution": False,
        "shot_starts_too_late": False,
        "shot_ends_too_early": False,
        "overall_confidence_pct": None,
        "trajectory_confidence_pct": None,
        "weighted_rmse": None,
        "notes": "no shots.json",
        "category": "STRESS TEST",
    }
    if not shots_path.exists():
        if processed_exists:
            row["notes"] = "pipeline completed; no shots.json"
        return row

    shots = json.loads(shots_path.read_text())
    row["num_shots"] = len(shots)
    if not shots:
        row["notes"] = "zero shot objects; sparse detections likely missed ball"
        return row

    primary = max(
        shots,
        key=lambda s: (
            0 if s.get("insufficient_points_for_fit") else 1000,
            len(s.get("candidate_points", [])),
            (s.get("confidence") or {}).get("trajectory_confidence") or 0,
        ),
    )
    metrics = analyze_shot(primary)
    row.update(metrics)
    row["num_shots"] = len(shots)
    row["hoop_lock_usable"] = metrics["hoop_lock_usable"] or processed_exists
    row["shot_detected"] = metrics["shot_detected"] or len(shots) > 0
    if len(shots) > 1 and row["notes"] == "clean":
        row["notes"] = f"{len(shots)} shot objects; best-quality shot scored"
    elif len(shots) > 1:
        row["notes"] = f"{len(shots)} shots; {row['notes']}"

    # Heuristic CORE vs STRESS from aggregate behavior
    measured = len(primary.get("candidate_points", []))
    if (
        row["trajectory_usable"] >= 4
        and not row["bounce_floor_pollution"]
        and not row["ball_lost_mid_flight"]
        and measured >= 10
    ):
        row["category"] = "CORE"
    else:
        row["category"] = "STRESS TEST"

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run pipeline on clips")
    parser.add_argument(
        "--only",
        nargs="*",
        help="Only run/analyze these clip labels (e.g. D E F)",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=True,
        help="Skip A B C when running (default true)",
    )
    parser.add_argument("--report", action="store_true", help="Write markdown report")
    args = parser.parse_args()

    videos = sorted(TESTING_DIR.iterdir(), key=lambda p: p.name.lower())
    videos = [v for v in videos if v.suffix.lower() in {".mov", ".mp4"}]

    labels_filter = {x.upper() for x in args.only} if args.only else None

    for video in videos:
        label = CLIP_LABELS.get(video.name)
        if label is None:
            continue
        if labels_filter and label not in labels_filter:
            continue
        if args.skip_completed and label in {"A", "B", "C"} and args.run:
            continue

        out_dir = EVAL_DIR / slugify(video.name)
        if args.run:
            code = run_clip(video, out_dir, label)
            if code != 0:
                print(f"WARNING: {video.name} exited {code}", flush=True)

    rows = []
    for video in sorted(videos, key=lambda p: CLIP_LABELS.get(p.name, "Z")):
        label = CLIP_LABELS.get(video.name)
        if label is None:
            continue
        if labels_filter and label not in labels_filter:
            continue
        out_dir = EVAL_DIR / slugify(video.name)
        rows.append(analyze_clip(out_dir, video.name, label))

    rows.sort(key=lambda r: r["label"])

    if args.report or not args.run:
        report_path = EVAL_DIR / "EVAL_SUMMARY.md"
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        report_path.write_text(build_markdown(rows), encoding="utf-8")
        print(f"\nWrote {report_path}")


def build_markdown(rows: list[dict]) -> str:
    failure_counts = {
        "no_shot_detected": sum(1 for r in rows if not r["shot_detected"]),
        "trajectory_unusable_lte2": sum(1 for r in rows if r["trajectory_usable"] <= 2),
        "hoop_lock_not_usable": sum(1 for r in rows if not r["hoop_lock_usable"]),
        "ball_lost_mid_flight": sum(1 for r in rows if r["ball_lost_mid_flight"]),
        "bounce_floor_pollution": sum(1 for r in rows if r["bounce_floor_pollution"]),
        "shot_starts_too_late": sum(1 for r in rows if r["shot_starts_too_late"]),
        "shot_ends_too_early": sum(1 for r in rows if r["shot_ends_too_early"]),
        "insufficient_fit": sum(
            1 for r in rows if "insufficient fit" in r.get("notes", "")
        ),
    }

    most_common = max(failure_counts.items(), key=lambda kv: kv[1])

    lines = [
        "# SwishSync Testing Clip Evaluation",
        "",
        "Pipeline: `swishsync-cv --select-hoop-on-first-frame` (no threshold tuning between clips).",
        "",
        "Methodology: Clips A–C seeded from prior runs; D–S batch-processed with interactive hoop ROI.",
        "Trajectory usability scored 1–5 from fit quality, RMSE, confidence, and lifecycle heuristics.",
        "",
        "## Summary Table",
        "",
        "| Clip | File | Category | Shot? | #Shots | Traj (1-5) | Hoop OK | Ball lost | Bounce | Late start | Early end | Overall conf | RMSE | Notes |",
        "|---|---|---|---|---:|---:|---|---|---|---|---|---:|---:|---|",
    ]

    for r in rows:
        lines.append(
            "| {label} | {filename} | {category} | {shot} | {n} | {traj} | {hoop} | {lost} | {bounce} | {late} | {early} | {conf} | {rmse} | {notes} |".format(
                label=r["label"],
                filename=r["filename"],
                category=r["category"],
                shot="yes" if r["shot_detected"] else "no",
                n=r["num_shots"],
                traj=r["trajectory_usable"],
                hoop="yes" if r["hoop_lock_usable"] else "no",
                lost="yes" if r["ball_lost_mid_flight"] else "no",
                bounce="yes" if r["bounce_floor_pollution"] else "no",
                late="yes" if r["shot_starts_too_late"] else "no",
                early="yes" if r["shot_ends_too_early"] else "no",
                conf=(
                    f"{r['overall_confidence_pct']:.1f}%"
                    if r["overall_confidence_pct"] is not None
                    else "n/a"
                ),
                rmse=f"{r['weighted_rmse']:.2f}" if r["weighted_rmse"] is not None else "n/a",
                notes=r["notes"].replace("|", "/"),
            )
        )

    lines.extend(
        [
            "",
            "## Failure Mode Counts",
            "",
        ]
    )
    for key, count in sorted(failure_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- **{key}**: {count}")

    lines.extend(
        [
            "",
            "## Most Common Failure Reason",
            "",
            f"**{most_common[0]}** ({most_common[1]} clips)",
            "",
            "## Recommended Next Engineering Priority",
            "",
            _recommendation(rows, failure_counts),
            "",
            "## Clip Categories",
            "",
            "### CORE clips",
            "",
        ]
    )
    core = [r for r in rows if r["category"] == "CORE"]
    stress = [r for r in rows if r["category"] == "STRESS TEST"]
    lines.append(
        ", ".join(f"{r['label']} ({r['filename']})" for r in core) or "_none yet_"
    )
    lines.extend(["", "### STRESS TEST clips", ""])
    lines.append(
        ", ".join(f"{r['label']} ({r['filename']})" for r in stress) or "_none_"
    )
    return "\n".join(lines) + "\n"


def _recommendation(rows: list[dict], failure_counts: dict[str, int]) -> str:
    if failure_counts["no_shot_detected"] >= failure_counts["bounce_floor_pollution"]:
        return (
            "1. **Sparse ball detection / shot-start sensitivity** — "
            f"{failure_counts['no_shot_detected']} clips produced zero shot objects; "
            "YOLO stride-3 misses short or low-visibility shots.\n"
            "2. **Lifecycle split / bounce exclusion** — on detected shots, bounce-floor "
            "pollution and early/late lifecycle bounds degrade trajectory quality.\n"
            "3. **Custom hoop YOLO weights** — interactive hoop ROI still required every run; "
            "auto lock remains non-functional on stock COCO weights."
        )
    return (
        "1. **Lifecycle split / bounce exclusion** — bounce-floor pollution is the most "
        "frequent failure; strengthen post-rim and long-gap finalization.\n"
        "2. **Custom hoop YOLO weights** for automatic lock without manual ROI.\n"
        "3. **Mid-flight reacquisition** for ball-lost gaps during ascent."
    )


if __name__ == "__main__":
    main()
