---
name: milestone
description: Ship a SwishSync milestone — run the quality gates, update docs, commit with a plain-English message, push the feature branch, rebuild the eval browser, and report. Use when a unit of work is done and verified, when the user says "ship it", "milestone", "commit and push", or at any natural completion point of a roadmap milestone.
---

# milestone — the ship ritual

Runs the full gate → document → ship → report sequence. This is the most
repeated workflow in the repo; do the steps IN ORDER and stop at the first
failing gate.

## 1. Gates (stop on first failure — report it verbatim, do not ship)

```bash
pytest -q                                   # must be 100% pass
python scripts/check_core_drift.py         # must print "within tolerance"
```

If the change touched detection, tracking, config, or pipeline:

```bash
python scripts/eval_shot_outcome.py         # wrong-verdict count must not
                                            # increase vs the last shipped run
```

Run long evals in the background; write full output to a scratchpad file and
filter on read (never pipe the live run through grep).

## 2. Document

- Append a `docs/DECISION_LOG.md` entry if the work made a choice a future
  session would re-litigate. Format: date heading, **Decision**, **Context /
  alternatives**, **Trade-off/Verification** — match the existing entries.
- Update `docs/ROADMAP.md` milestone status if a milestone completed.
- If eval artifacts changed: rebuild the browser —
  `python scripts/prepare_eval_browser_videos.py --eval-dir <dir>` then
  `python scripts/build_eval_browser.py --eval-dir <dir>`, verify
  `curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/<dir>/index.html`
  returns 200.

## 3. Ship

- Stage deliberately (`git add <paths>` — never blind `git add -A` when
  scratch/experiment files might be lying around; check `git status` first).
- Commit message: plain English, derived from the ACTUAL diff — what changed,
  why, verification numbers (test count, agreement, CORE status). End with the
  session's Co-Authored-By line.
- Push the current `feature/*` branch. The pre-push CORE hook may run — if it
  fails, the push is correctly blocked; fix, don't `--no-verify`.
- **NEVER merge, never open a PR, never touch main/dev** without an explicit
  user "go" for that specific action.

## 4. Report (caveman style if active)

One short block: what shipped (commit hash), gate numbers (tests / CORE /
agreement), what's next per the roadmap, and anything that needs the user
(labels, calibration, credits, review).

## 5. Retro — the ritual iterates itself

After every ship, answer two questions in one line each:

1. **What was traded off?** Anything accepted-but-imperfect (a threshold
   guessed, a proxy metric, a deferred edge case) → record it in the
   DECISION_LOG entry's **Trade-off** field. If the milestone made no entry,
   a real trade-off forces one.
2. **What slowed this milestone or nearly caused a mistake?** If it matches an
   existing failure mode in CLAUDE.md → cite it in the report (reinforcement).
   If it's NEW → same commit gets:
   - a new row in CLAUDE.md's failure-mode table (name it, state the rule), and
   - if this ritual itself was the gap, an edit to THIS skill file fixing it.

Skill/manual edits ship inside the milestone commit — self-iteration is part
of the milestone, not a separate chore. Never delete history from the
DECISION_LOG or the failure-mode table; supersede with a new row/entry that
points back (same rule as the decision log).

Change log of this skill (newest first):
- 2026-07-07: added self-iteration retro (this section).
- 2026-07-06: initial ritual (gates → document → ship → report).
