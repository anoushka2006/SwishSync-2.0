---
name: eval
description: Run the SwishSync make/miss eval on the 19 Testing clips, build the static eval browser, and serve it locally. Use when the user asks to run the eval, check outcome agreement, rebuild or open the eval browser, or verify make/miss after a change.
---

# eval

Run the full evaluation + browser in one step:

```bash
scripts/eval_browser.sh
```

This runs `eval_shot_outcome.py` (full pipeline with auto hoop lock on all 19
clips) → `prepare_eval_browser_videos.py` → `build_eval_browser.py`, then serves
on http://localhost:8000/outputs/temp/outcome_check/index.html.

- Outcome ground truth + agreement target (14/17, zero wrong verdicts) live in
  `scripts/run_full_eval_rerun.py::OUTCOME_GROUND_TRUTH`.
- CORE RMSE baselines are in `run_full_eval_rerun.py::BASELINE_RMSE`; a
  regression there is a real problem, not a metric artifact.
- The eval is CPU-only and takes a few minutes. Run it in the background and
  report the final agreement table.
- Do NOT commit `outputs/` — it is git-ignored workspace.
