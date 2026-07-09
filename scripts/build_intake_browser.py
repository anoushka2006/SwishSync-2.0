#!/usr/bin/env python3
"""Build a labeling page for a clip-intake batch (clip-intake step 2 helper).

Transcodes each intake clip's processed.mp4 to a browser-playable preview and
writes an index.html where the user watches each clip and types the ordered
per-shot outcomes (`make, miss, ...`). A copy button assembles all labels into
the exact `LETTER: outcomes` lines to paste back into chat.

    python scripts/build_intake_browser.py            # default intake dir
    open http://localhost:8000/outputs/temp/intake/index.html
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTAKE = ROOT / "outputs" / "temp" / "intake"


def transcode(src: Path, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
        return True
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vcodec", "libx264",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dest)]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def clip_card(letter: str, clip_dir: Path) -> str:
    shots = []
    sj = clip_dir / "shots.json"
    if sj.exists():
        shots = json.loads(sj.read_text())
    detected = ", ".join(
        f"shot@f{s['start_frame']}→{(s.get('outcome') or {}).get('verdict') or '?'}"
        for s in shots
    ) or "no shot detected"
    return f"""
  <div class="card">
    <h3>{html.escape(letter)} <small>({html.escape(clip_dir.name)})</small></h3>
    <video src="{html.escape(letter)}/preview.mp4" controls preload="metadata"></video>
    <p class="detected">pipeline saw: {html.escape(detected)}</p>
    <label>ordered outcomes (e.g. <code>make, miss, make</code>):
      <input type="text" data-letter="{html.escape(letter)}" placeholder="make, miss, ...">
    </label>
  </div>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Intake labeling page")
    parser.add_argument("--intake-dir", type=Path, default=INTAKE)
    args = parser.parse_args()

    clip_dirs = sorted(
        (d for d in args.intake_dir.iterdir() if (d / "processed.mp4").exists()),
        key=lambda d: (len(d.name), d.name),  # T..Z before AA..AI
    )
    cards = []
    for d in clip_dirs:
        if not transcode(d / "processed.mp4", d / "preview.mp4"):
            print(f"{d.name}: transcode FAILED", file=sys.stderr)
            continue
        cards.append(clip_card(d.name, d))

    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Clip intake — label the new batch</title>
<style>
 body {{ font-family: -apple-system, sans-serif; background:#1a1a1e; color:#eee; margin:2rem; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:1.2rem; }}
 .card {{ background:#26262c; border-radius:10px; padding:1rem; }}
 video {{ width:100%; border-radius:6px; background:#000; }}
 input {{ width:100%; padding:.5rem; border-radius:6px; border:1px solid #444;
          background:#141417; color:#eee; margin-top:.3rem; }}
 .detected {{ color:#9a9; font-size:.85rem; }}
 #copyall {{ position:fixed; bottom:1.2rem; right:1.2rem; padding:.8rem 1.4rem;
   background:#2f7d46; color:#fff; border:0; border-radius:8px; font-size:1rem; cursor:pointer; }}
 h1 small {{ color:#888; font-size:.6em; }}
</style></head><body>
<h1>Label the new clips <small>watch → type ordered outcomes → Copy all → paste to Claude</small></h1>
<div class="grid">{''.join(cards)}
</div>
<button id="copyall">Copy all labels</button>
<script>
document.getElementById('copyall').onclick = () => {{
  const lines = [...document.querySelectorAll('input[data-letter]')]
    .filter(i => i.value.trim())
    .map(i => i.dataset.letter + ': ' + i.value.trim());
  navigator.clipboard.writeText(lines.join('\\n'));
  const btn = document.getElementById('copyall');
  btn.textContent = 'Copied ' + lines.length + ' clips ✓';
  setTimeout(() => btn.textContent = 'Copy all labels', 2000);
}};
</script></body></html>"""
    out = args.intake_dir / "index.html"
    out.write_text(page)
    print(f"wrote {out} ({len(cards)} clips)")


if __name__ == "__main__":
    main()
