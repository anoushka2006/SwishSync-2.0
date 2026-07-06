# Automations

Five automations, respecting the guardrails in CLAUDE.md (AI never merges or
deploys itself; verify before code leaves local).

## 1. Test-on-stop (`.claude/settings.json`)
Stop hook runs `pytest -q` after every Claude turn and shows the tail. Fast
(<1s suite) — catches breakage the moment it happens.

## 2. `/eval` skill (`.claude/skills/eval/`)
One command runs the full make/miss eval → builds the static browser → serves
it. Wraps `scripts/eval_browser.sh`. Use after any pipeline change.

## 3. Pre-push CORE-drift gate (`.githooks/pre-push` + `scripts/check_core_drift.py`)
Before a push, IF `src/swishsync_cv/tracking|detection/` changed, runs the CORE
clips and fails the push on RMSE drift > 0.15 vs the pinned auto-lock baseline.
Skips cleanly when local videos/weights are absent. Bypass: `git push --no-verify`.
**Enable once:** `git config core.hooksPath .githooks`

## 4. Nightly local eval (`scripts/nightly_eval.sh`)
Appends `date | agreement | CORE ok/DRIFT` to `outputs/nightly_log.txt`.
LOCAL only — the Testing videos are git-ignored and exist only on this machine,
so this can NOT be a cloud agent. **Enable via launchd** (macOS):

```bash
# ~/Library/LaunchAgents/com.swishsync.nightly.plist runs scripts/nightly_eval.sh
# at 02:00 daily. Create with: launchctl load <plist>. Template:
#   ProgramArguments: [/bin/bash, <repo>/scripts/nightly_eval.sh]
#   StartCalendarInterval: {Hour: 2, Minute: 0}
```

## 5. Roboflow active-learning uploader (`scripts/roboflow_active_learning.py`)
Feeds the own-weights training flywheel (task #2). Reads `ROBOFLOW_API_KEY` from
env (never hard-coded). Two modes:
- `--own-clips` → upload `datasets/own_clips/images/` for annotation.
- `--low-conf DIR` → upload weak-detection frames (far-court misses = the
  active-learning signal).
Target: forked CC-BY project `basketball-strategy/cv-cnfd4-eaond`. Annotate
rim-only + ball. **Note:** true closed-loop active learning (auto-collect during
inference) activates only after a model is trained and deployed.
