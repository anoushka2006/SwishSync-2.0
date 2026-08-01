#!/usr/bin/env python3
"""Create browser-compatible preview_<letter>.mp4 files from processed eval videos."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_testing_clips import (  # noqa: E402
    CLIP_LABELS,
    DEFAULT_EVAL_DIR,
    TESTING_DIR,
    preview_video_name,
    processed_video_name,
    slugify,
)

FFMPEG_MISSING_HELP = """
ffmpeg is required to create browser-compatible preview videos.

Install ffmpeg, then rerun:
  python scripts/prepare_eval_browser_videos.py
  python scripts/build_eval_browser.py

macOS (Homebrew):  brew install ffmpeg
Ubuntu/Debian:   sudo apt install ffmpeg
"""


def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def resolve_processed_video(out_dir: Path, label: str) -> Path | None:
    preferred = out_dir / processed_video_name(label)
    if preferred.exists():
        return preferred
    fallback = out_dir / "processed.mp4"
    if fallback.exists():
        return fallback
    return None


def source_has_audio(ffmpeg: str, source: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return False
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def build_ffmpeg_command(ffmpeg: str, source: Path, dest: Path, *, with_audio: bool) -> list[str]:
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if with_audio:
        cmd.extend(["-acodec", "aac"])
    else:
        cmd.append("-an")
    cmd.append(str(dest))
    return cmd


def convert_preview(ffmpeg: str, source: Path, dest: Path) -> None:
    with_audio = source_has_audio(ffmpeg, source)
    cmd = build_ffmpeg_command(ffmpeg, source, dest, with_audio=with_audio)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return

    if with_audio:
        retry = build_ffmpeg_command(ffmpeg, source, dest, with_audio=False)
        retry_result = subprocess.run(retry, capture_output=True, text=True, check=False)
        if retry_result.returncode == 0:
            return
        raise RuntimeError(retry_result.stderr.strip() or retry_result.stdout.strip())

    raise RuntimeError(result.stderr.strip() or result.stdout.strip())


def should_convert(source: Path, dest: Path, force: bool) -> bool:
    if force or not dest.exists():
        return True
    return source.stat().st_mtime > dest.stat().st_mtime


def prepare_previews(eval_dir: Path, *, force: bool = False) -> tuple[int, int, int]:
    converted = 0
    skipped = 0
    missing = 0

    for video in sorted(TESTING_DIR.iterdir(), key=lambda path: path.name.lower()):
        label = CLIP_LABELS.get(video.name)
        if label is None:
            continue

        out_dir = eval_dir / slugify(video.name)
        source = resolve_processed_video(out_dir, label)
        if source is None:
            missing += 1
            print(f"[{label}] skip — processed video not found in {out_dir.name}/")
            continue

        dest = out_dir / preview_video_name(label)
        if not should_convert(source, dest, force):
            skipped += 1
            print(f"[{label}] up to date — {dest.name}")
            continue

        print(f"[{label}] converting {source.name} -> {dest.name}")
        convert_preview(find_ffmpeg() or "ffmpeg", source, dest)
        converted += 1

    return converted, skipped, missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create browser-compatible preview_<letter>.mp4 files for the eval browser",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help="Evaluation output root (default: outputs/eval/shot_story)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode all previews even when up to date",
    )
    args = parser.parse_args()

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        print(FFMPEG_MISSING_HELP.strip())
        sys.exit(1)

    eval_dir = args.eval_dir.resolve()
    converted, skipped, missing = prepare_previews(eval_dir, force=args.force)
    print(
        f"\nDone: converted={converted} skipped={skipped} missing_source={missing} "
        f"eval_dir={eval_dir.relative_to(ROOT)}"
    )
    print("Next: python scripts/build_eval_browser.py")


if __name__ == "__main__":
    main()
