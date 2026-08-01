#!/usr/bin/env bash
# Run make/miss eval → build the static browser → serve it locally.
# Usage: scripts/eval_browser.sh [eval_dir] [port]
set -euo pipefail
cd "$(dirname "$0")/.."

EVAL_DIR="${1:-outputs/temp/outcome_check}"
PORT="${2:-8000}"

python scripts/eval_shot_outcome.py --out-dir "$EVAL_DIR"
python scripts/prepare_eval_browser_videos.py --eval-dir "$EVAL_DIR"
python scripts/build_eval_browser.py --eval-dir "$EVAL_DIR"

echo "Serving on http://localhost:${PORT}/${EVAL_DIR}/index.html  (Ctrl-C to stop)"
python -m http.server "$PORT"
