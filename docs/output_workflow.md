# Output workflow

How to place pipeline artifacts under `outputs/` so debug, evaluation, and showcase runs stay separated.

## Directory guide

| Folder | Purpose | Canonical? |
|--------|---------|------------|
| `outputs/eval/shot_story/` | Latest full 19-clip Testing eval | **Yes** — current reference |
| `outputs/eval/pr1/` | Short-gap recovery PR snapshot | Reference for PR1 comparison |
| `outputs/eval/baseline/` | Pre-PR / pre-feature baseline runs | Reference when populated |
| `outputs/eval/pr1_test/` | Small PR1 smoke tests | No — disposable |
| `outputs/showcase/` | Curated demo videos | **Yes** — shareable |
| `outputs/debug/bbox/` | Hoop ROI / bbox experiments | No |
| `outputs/debug/instrumentation/` | Agent logs, traces, one-off probes | No |
| `outputs/debug/core_drift_verify/` | RMSE drift A/B outputs | No |
| `outputs/temp/` | Legacy clips, scratch CLI runs | No — safe to delete |

## Where new work should go

### Full evaluation (19 Testing clips)

```bash
python scripts/run_full_eval_rerun.py
# writes to outputs/eval/shot_story/ by default

python scripts/run_full_eval_rerun.py --eval-dir outputs/eval/my_experiment
# isolated experiment without overwriting canonical eval
```

Report-only analysis (no pipeline rerun):

```bash
python scripts/evaluate_testing_clips.py --report
python scripts/evaluate_testing_clips.py --report --eval-dir outputs/eval/pr1
```

### Single-clip or ad-hoc CLI runs

```bash
swishsync-cv \
  --input videos/Testing/IMG_1961.MOV \
  --output-dir outputs/temp/clip_c_retry \
  --select-hoop-on-first-frame
```

Do not use `outputs/` root directly; older root-level `processed.mp4` / `shots.json` files were moved to `outputs/temp/root_pipeline_run/`.

### Showcase clips

Copy or render polished outputs to:

```text
outputs/showcase/processed_a.mp4
outputs/showcase/processed_h_bounce_demo.mp4
```

Use descriptive suffixes when the letter alone is ambiguous. Keep source eval paths in commit messages or PR notes, not duplicated inside showcase filenames unless helpful.

## Naming conventions

### Evaluation clip outputs

Each Testing clip maps to a letter **A–S** (see `scripts/evaluate_testing_clips.py` → `CLIP_LABELS`).

Per-clip directory (slugified filename):

```text
outputs/eval/shot_story/IMG_2027_2/
  processed_d.mp4      # lettered processed video
  shots.json
  detections.jsonl
  detections.csv
```

Letter → file examples:

| Letter | Video | Processed name |
|--------|-------|----------------|
| A | IMG_1962.MOV | `processed_a.mp4` |
| C | IMG_1961.MOV | `processed_c.mp4` |
| D | IMG_2027 2.MOV | `processed_d.mp4` |
| H | IMG_2027.MOV | `processed_h.mp4` |
| Q | IMG_2029 8.MOV | `processed_q.mp4` |

Legacy unlettered `processed.mp4` may exist in older runs; scripts prefer `processed_<letter>.mp4` when present.

### Eval summaries

| File | Location |
|------|----------|
| `EVAL_SUMMARY.md` | Same eval root (e.g. `outputs/eval/shot_story/`) |
| `eval_rerun_results.json` | Written by `run_full_eval_rerun.py` |

## Safe to delete

- `outputs/temp/**` — scratch and legacy runs
- `outputs/debug/**` — experiments after review
- `outputs/eval/pr1_test/**` — smoke runs
- Extra eval roots created with `--eval-dir` once results are merged or abandoned

## Keep (canonical references)

- `outputs/eval/shot_story/` — current full eval + `EVAL_SUMMARY.md`
- `outputs/eval/pr1/` — PR1 comparison snapshot
- `outputs/eval/baseline/` — when used for regression baselines
- `outputs/showcase/` — demo-ready exports

## Script path defaults

| Script | Default output root |
|--------|---------------------|
| `scripts/run_full_eval_rerun.py` | `outputs/eval/shot_story` |
| `scripts/evaluate_testing_clips.py` | `outputs/eval/shot_story` |
| `scripts/verify_core_drift.py` | `outputs/debug/core_drift_verify` |
| `swishsync-cv` CLI | `--output-dir` argument (prefer `outputs/temp/...`) |

Override eval location:

```bash
python scripts/run_full_eval_rerun.py --eval-dir outputs/eval/baseline
python scripts/evaluate_testing_clips.py --report --eval-dir outputs/eval/pr1
```

Hoop bbox extraction for reruns searches, in order: the active `--eval-dir`, then `shot_story`, `pr1`, `baseline`, and `pr1_test`.

## Eval browser (local review)

After a full eval (or any run that populates per-clip folders), create browser-safe preview encodes and build the HTML index:

```bash
python scripts/prepare_eval_browser_videos.py
python scripts/build_eval_browser.py
# optional: --eval-dir outputs/eval/my_experiment on both commands
```

`prepare_eval_browser_videos.py` writes `preview_<letter>.mp4` (H.264 + AAC via ffmpeg) next to each `processed_<letter>.mp4` without modifying pipeline outputs. If ffmpeg is missing, it prints install instructions.

Output:

```text
outputs/eval/shot_story/index.html
outputs/eval/shot_story/IMG_1962/preview_a.mp4
```

Serve from the **repo root** so embedded videos play reliably:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/outputs/eval/shot_story/index.html
```

The page embeds `preview_<letter>.mp4` when available (fallback: original processed video), keeps links to the full `processed_<letter>.mp4`, and lists shot metrics, notes, and artifact links. Regenerate previews and the index after rerunning eval clips.
