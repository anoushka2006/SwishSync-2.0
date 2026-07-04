#!/usr/bin/env python3
"""Build a static HTML browser for SwishSync eval outputs under outputs/eval/shot_story/."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_testing_clips import (  # noqa: E402
    CLIP_LABELS,
    DEFAULT_EVAL_DIR,
    TESTING_DIR,
    analyze_clip,
    preview_video_name,
    processed_video_name,
    slugify,
)

LOW_CONFIDENCE_PCT = 70.0
HIGH_RMSE_PX = 2.0


def display_category(row: dict) -> str:
    if row.get("num_shots", 0) == 0 or not row.get("shot_detected"):
        return "FAILURE"
    if row.get("category") == "CORE":
        return "CORE"
    return "STRESS"


def collect_clip_rows(eval_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for video in sorted(TESTING_DIR.iterdir(), key=lambda path: path.name.lower()):
        label = CLIP_LABELS.get(video.name)
        if label is None:
            continue
        out_dir = eval_dir / slugify(video.name)
        row = analyze_clip(out_dir, video.name, label)
        slug = slugify(video.name)
        video_name = processed_video_name(label)
        video_path = out_dir / video_name
        if not video_path.exists():
            video_path = out_dir / "processed.mp4"

        video_name = processed_video_name(label)
        video_path = out_dir / video_name
        if not video_path.exists():
            video_path = out_dir / "processed.mp4"

        outcome_verdict = None
        outcome_margin = None
        shots_path = out_dir / "shots.json"
        if shots_path.exists():
            try:
                shots = json.loads(shots_path.read_text())
            except json.JSONDecodeError:
                shots = []
            if shots:
                primary = max(shots, key=lambda s: len(s.get("candidate_points", [])))
                outcome = primary.get("outcome") or {}
                outcome_verdict = outcome.get("verdict")
                outcome_margin = outcome.get("margin_ratio")

        preview_path = out_dir / preview_video_name(label)
        rel_dir = slug
        processed_href = (
            f"{rel_dir}/{video_path.name}".replace("\\", "/") if video_path.exists() else None
        )
        preview_href = (
            f"{rel_dir}/{preview_path.name}".replace("\\", "/") if preview_path.exists() else None
        )
        embed_href = preview_href or processed_href
        rows.append(
            {
                "label": label,
                "filename": video.name,
                "slug": slug,
                "display_category": display_category(row),
                "source_category": row.get("category", "STRESS TEST"),
                "shot_detected": bool(row.get("shot_detected")),
                "num_shots": row.get("num_shots", 0),
                "trajectory_usable": row.get("trajectory_usable"),
                "overall_confidence_pct": row.get("overall_confidence_pct"),
                "trajectory_confidence_pct": row.get("trajectory_confidence_pct"),
                "weighted_rmse": row.get("weighted_rmse"),
                "outcome_verdict": outcome_verdict,
                "outcome_margin": outcome_margin,
                "notes": row.get("notes", ""),
                "start_frame": row.get("start_frame"),
                "end_frame": row.get("end_frame"),
                "flags": build_flags(row),
                "links": {
                    "video": processed_href,
                    "preview": preview_href,
                    "embed": embed_href,
                    "shots_json": f"{rel_dir}/shots.json".replace("\\", "/")
                    if (out_dir / "shots.json").exists()
                    else None,
                    "detections_csv": f"{rel_dir}/detections.csv".replace("\\", "/")
                    if (out_dir / "detections.csv").exists()
                    else None,
                },
                "has_video": video_path.exists(),
                "has_preview": preview_path.exists(),
            }
        )
    rows.sort(key=lambda row: row["label"])
    return rows


def build_flags(row: dict) -> list[str]:
    flags: list[str] = []
    display = display_category(row)
    if display == "CORE":
        flags.append("core")
    if display == "FAILURE":
        flags.append("failure")
    if row.get("num_shots", 0) == 0 or not row.get("shot_detected"):
        flags.append("missing_shot")
    overall = row.get("overall_confidence_pct")
    if overall is not None and overall < LOW_CONFIDENCE_PCT:
        flags.append("low_confidence")
    rmse = row.get("weighted_rmse")
    if rmse is not None and rmse > HIGH_RMSE_PX:
        flags.append("high_rmse")
    return flags


def render_html(clips: list[dict], eval_dir: Path) -> str:
    payload = json.dumps(clips, indent=2)
    eval_name = html.escape(eval_dir.name)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SwishSync Eval Browser — {eval_name}</title>
  <style>
    :root {{
      --bg: #111318;
      --panel: #1a1d24;
      --border: #2b3140;
      --text: #e8ebf2;
      --muted: #9aa3b5;
      --core: #3ecf8e;
      --stress: #f0b429;
      --failure: #ff6b6b;
      --link: #7db4ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    header {{
      padding: 20px 24px 12px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, #171a22, var(--bg));
    }}
    h1 {{ margin: 0 0 6px; font-size: 1.5rem; }}
    .subtitle {{ color: var(--muted); font-size: 0.95rem; }}
    .server-note {{
      margin-top: 10px;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #202532;
      color: var(--muted);
      font-size: 0.85rem;
    }}
    .server-note code {{
      color: var(--text);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.82rem;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      padding: 14px 24px;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
    }}
    .toolbar label {{ color: var(--muted); font-size: 0.9rem; }}
    .filter-btn {{
      border: 1px solid var(--border);
      background: #222733;
      color: var(--text);
      padding: 6px 12px;
      border-radius: 999px;
      cursor: pointer;
      font-size: 0.85rem;
    }}
    .filter-btn.active {{
      background: #2f3b55;
      border-color: var(--link);
    }}
    .stats {{
      margin-left: auto;
      color: var(--muted);
      font-size: 0.85rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 16px;
      padding: 20px 24px 40px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    .card.hidden {{ display: none; }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px 10px;
      border-bottom: 1px solid var(--border);
    }}
    .clip-letter {{
      font-size: 1.6rem;
      font-weight: 700;
      line-height: 1;
    }}
    .filename {{ color: var(--muted); font-size: 0.85rem; word-break: break-word; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }}
    .badge {{
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      padding: 3px 8px;
      border-radius: 999px;
      border: 1px solid transparent;
    }}
    .badge.core {{ color: var(--core); border-color: color-mix(in srgb, var(--core) 50%, transparent); }}
    .badge.stress {{ color: var(--stress); border-color: color-mix(in srgb, var(--stress) 50%, transparent); }}
    .badge.failure {{ color: var(--failure); border-color: color-mix(in srgb, var(--failure) 50%, transparent); }}
    .badge.flag {{ color: #ffb4b4; border-color: #5a3030; background: #2a1818; }}
    .badge.flag.warn {{ color: #ffd27a; border-color: #5a4520; background: #2a2218; }}
    .badge.make {{ color: var(--core); border-color: color-mix(in srgb, var(--core) 50%, transparent); }}
    .badge.miss {{ color: var(--failure); border-color: color-mix(in srgb, var(--failure) 50%, transparent); }}
    .badge.unknown {{ color: var(--muted); border-color: var(--border); }}
    .video-wrap {{
      background: #000;
      aspect-ratio: 16 / 9;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    video {{ width: 100%; height: 100%; object-fit: contain; background: #000; }}
    .no-video {{ color: var(--muted); font-size: 0.9rem; padding: 24px; text-align: center; }}
    .video-fallback {{
      padding: 8px 16px 0;
      text-align: center;
      background: #000;
    }}
    .video-fallback a {{
      color: var(--link);
      font-size: 0.82rem;
      text-decoration: none;
    }}
    .video-fallback a:hover {{ text-decoration: underline; }}
    .muted-inline {{ color: var(--muted); font-size: 0.78rem; }}
    .metrics {{
      padding: 12px 16px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px 14px;
      font-size: 0.85rem;
    }}
    .metric-label {{ color: var(--muted); }}
    .metric-value {{ font-variant-numeric: tabular-nums; }}
    .notes {{
      padding: 0 16px 12px;
      color: var(--muted);
      font-size: 0.82rem;
    }}
    .links {{
      padding: 12px 16px 16px;
      border-top: 1px solid var(--border);
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .links a {{
      color: var(--link);
      text-decoration: none;
      font-size: 0.85rem;
      border: 1px solid var(--border);
      padding: 5px 10px;
      border-radius: 8px;
      background: #202532;
    }}
    .links a:hover {{ border-color: var(--link); }}
    .links a.disabled {{
      pointer-events: none;
      opacity: 0.35;
    }}
  </style>
</head>
<body>
  <header>
    <h1>SwishSync Eval Browser</h1>
    <div class="subtitle">Benchmark clips A–S · {html.escape(str(eval_dir.relative_to(ROOT)))}</div>
    <div class="server-note">
      Serve from the repo root for reliable video playback (avoid <code>file://</code>).
      Generate browser-safe previews first with
      <code>python scripts/prepare_eval_browser_videos.py</code>, then run
      <code>python -m http.server 8000</code>
      and open
      <code>http://localhost:8000/{html.escape(str(eval_dir.relative_to(ROOT)).replace(chr(92), "/"))}/index.html</code>
    </div>
  </header>
  <div class="toolbar">
    <label>Filter:</label>
    <button class="filter-btn active" data-filter="ALL">All</button>
    <button class="filter-btn" data-filter="CORE">CORE</button>
    <button class="filter-btn" data-filter="STRESS">STRESS</button>
    <button class="filter-btn" data-filter="FAILURE">FAILURE</button>
    <div class="stats" id="stats"></div>
  </div>
  <div class="grid" id="grid"></div>
  <script>
    const CLIPS = {payload};
    const FLAG_LABELS = {{
      core: "CORE",
      failure: "FAILURE",
      missing_shot: "No shot",
      low_confidence: "Low conf",
      high_rmse: "High RMSE",
    }};

    function fmt(value, suffix = "") {{
      if (value === null || value === undefined || value === "") return "n/a";
      return `${{value}}${{suffix}}`;
    }}

    function badgeClass(flag) {{
      if (flag === "low_confidence" || flag === "high_rmse") return "flag warn";
      if (flag === "core" || flag === "failure" || flag === "missing_shot") return "flag";
      return "flag";
    }}

    function renderCard(clip) {{
      const cat = clip.display_category.toLowerCase();
      const flagBadges = clip.flags
        .filter((flag) => !["core", "failure"].includes(flag))
        .map((flag) => `<span class="badge ${{badgeClass(flag)}}">${{FLAG_LABELS[flag] || flag}}</span>`)
        .join("");
      const embedSrc = clip.links.embed;
      const videoBlock = embedSrc
        ? `<div class="video-wrap">
             <video controls preload="metadata" playsinline>
               <source src="${{embedSrc}}" type="video/mp4">
             </video>
           </div>
           <div class="video-fallback">
             <a href="${{embedSrc}}" target="_blank" rel="noopener">Open video file</a>
             ${{clip.links.preview && clip.links.video && clip.links.preview !== clip.links.video
               ? `<span class="muted-inline"> · browser preview</span>` : ""}}
           </div>`
        : `<div class="video-wrap"><div class="no-video">processed video not found</div></div>`;
      const link = (href, label) => href
        ? `<a href="${{href}}" target="_blank" rel="noopener">${{label}}</a>`
        : `<a class="disabled">${{label}}</a>`;

      return `
        <article class="card" data-category="${{clip.display_category}}">
          <div class="card-head">
            <div>
              <div class="clip-letter">${{clip.label}}</div>
              <div class="filename">${{clip.filename}}</div>
            </div>
            <div class="badges">
              <span class="badge ${{cat}}">${{clip.display_category}}</span>
              ${{clip.outcome_verdict
                ? `<span class="badge ${{clip.outcome_verdict}}">${{clip.outcome_verdict}}</span>` : ""}}
              ${{flagBadges}}
            </div>
          </div>
          ${{videoBlock}}
          <div class="metrics">
            <div><span class="metric-label">Shot detected</span><div class="metric-value">${{clip.shot_detected ? "yes" : "no"}}</div></div>
            <div><span class="metric-label">Shot count</span><div class="metric-value">${{clip.num_shots}}</div></div>
            <div><span class="metric-label">RMSE</span><div class="metric-value">${{fmt(clip.weighted_rmse, " px")}}</div></div>
            <div><span class="metric-label">Overall conf</span><div class="metric-value">${{fmt(clip.overall_confidence_pct, "%")}}</div></div>
            <div><span class="metric-label">Traj usable</span><div class="metric-value">${{fmt(clip.trajectory_usable, " / 5")}}</div></div>
            <div><span class="metric-label">Traj conf</span><div class="metric-value">${{fmt(clip.trajectory_confidence_pct, "%")}}</div></div>
            <div><span class="metric-label">Frames</span><div class="metric-value">${{clip.start_frame != null ? `${{clip.start_frame}}–${{clip.end_frame}}` : "n/a"}}</div></div>
            <div><span class="metric-label">Source cat</span><div class="metric-value">${{clip.source_category}}</div></div>
            <div><span class="metric-label">Make/Miss</span><div class="metric-value">${{fmt(clip.outcome_verdict)}}</div></div>
            <div><span class="metric-label">Rim margin</span><div class="metric-value">${{clip.outcome_margin != null ? clip.outcome_margin.toFixed(2) : "n/a"}}</div></div>
          </div>
          <div class="notes">${{clip.notes || "—"}}</div>
          <div class="links">
            ${{link(clip.links.video, "processed mp4")}}
            ${{link(clip.links.shots_json, "shots.json")}}
            ${{link(clip.links.detections_csv, "detections.csv")}}
          </div>
        </article>`;
    }}

    function updateStats(visibleCount) {{
      const total = CLIPS.length;
      const core = CLIPS.filter((clip) => clip.display_category === "CORE").length;
      const failure = CLIPS.filter((clip) => clip.display_category === "FAILURE").length;
      document.getElementById("stats").textContent =
        `Showing ${{visibleCount}} / ${{total}} · CORE ${{core}} · FAILURE ${{failure}}`;
    }}

    function applyFilter(filter) {{
      const cards = document.querySelectorAll(".card");
      let visible = 0;
      cards.forEach((card) => {{
        const show = filter === "ALL" || card.dataset.category === filter;
        card.classList.toggle("hidden", !show);
        if (show) visible += 1;
      }});
      updateStats(visible);
    }}

    const grid = document.getElementById("grid");
    grid.innerHTML = CLIPS.map(renderCard).join("");
    updateStats(CLIPS.length);

    document.querySelectorAll(".filter-btn").forEach((button) => {{
      button.addEventListener("click", () => {{
        document.querySelectorAll(".filter-btn").forEach((btn) => btn.classList.remove("active"));
        button.classList.add("active");
        applyFilter(button.dataset.filter);
      }});
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static eval browser HTML")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help="Evaluation output root (default: outputs/eval/shot_story)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: <eval-dir>/index.html)",
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir.resolve()
    output_path = (args.output or eval_dir / "index.html").resolve()
    clips = collect_clip_rows(eval_dir)
    output_path.write_text(render_html(clips, eval_dir), encoding="utf-8")
    print(f"Wrote {output_path} ({len(clips)} clips)")


if __name__ == "__main__":
    main()
