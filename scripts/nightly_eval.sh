#!/usr/bin/env bash
# Nightly regression run (LOCAL — the Testing videos are git-ignored and only
# exist on this machine, so this cannot run as a cloud agent). Runs the full
# eval + CORE drift check and appends a dated line to outputs/nightly_log.txt.
# Enable via launchd: see docs/automations.md.
set -uo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date "+%Y-%m-%d %H:%M")
LOG=outputs/nightly_log.txt
mkdir -p outputs

AGREE=$(python scripts/eval_shot_outcome.py 2>/dev/null | grep -E "^agreement" || echo "agreement: ERROR")
python scripts/check_core_drift.py >/dev/null 2>&1 && CORE="CORE ok" || CORE="CORE DRIFT"

echo "$STAMP | $AGREE | $CORE" >> "$LOG"
echo "$STAMP | $AGREE | $CORE"
