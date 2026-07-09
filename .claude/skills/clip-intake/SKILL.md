---
name: clip-intake
description: Absorb a new batch of filmed clips into the benchmark and training flywheel — inventory, label collection, crash-run, frame extraction, categorization (roadmap M8). Use when new videos land in videos/, when the user says they filmed/uploaded clips, or when extending the benchmark suite.
---

# clip-intake — new clips → benchmark + training data

Every clip double-dips: a benchmark entry AND detector-training frames. Do the
steps in order; the user-labels step gates the benchmark half but NOT the
training half — never block frame extraction on labels.

## 1. Inventory (cheap, do immediately)

- List new files in `videos/` (root — `videos/Testing/` holds the lettered
  suite). Probe each with cv2: WxH, fps, duration. Duration > ~15s at 30fps
  usually means multi-shot.
- Assign the next free letters (after S: T, U, V... then AA, AB...). Propose
  the letter↔file mapping to the user in the report; letters become permanent
  once labels attach — do not renumber later.

## 2. Ask the user for per-shot labels (blocking for benchmark only)

Ordered outcomes per clip, e.g. `T: make, miss, make`. One label per shot, in
shot order (filming spec told them to jot these). Rules:
- User-confirmed labels ONLY enter ground truth (failure mode #10) — never
  infer labels from montages without a confirmation pass.
- Multi-shot clips go to `OUTCOME_GROUND_TRUTH_V2: dict[label, list[verdict]]`
  (per-shot schema, M8); single-shot clips may also live in the legacy
  single-verdict table.

## 3. Crash-run (no labels needed)

Run every new clip through the pipeline (auto hoop lock, current weights) as a
background Bash sweep — full logs to the CURRENT session scratchpad
(`mkdir -p` first; failure mode #14). Success bar: zero crashes; every clip
produces shots.json (empty is fine — that's data, not failure).
Record per clip: shots found, points, fit?, verdict, hoop lock frame.

## 4. Training frames (no labels needed)

`python scripts/extract_training_frames.py` (extend its CLIP_LABELS coverage or
pass the new dir) → rim-interaction-biased JPEGs into
`datasets/own_clips/images/`. Then push frames (never raw video) to Roboflow:
`python scripts/roboflow_active_learning.py --own-clips` with
`ROBOFLOW_API_KEY` from env. User annotates: `basketball` + `rim` (ring only).

## 5. Categorize (after labels + crash-run)

- Side-view, clean single/multi-shot, tracks well → CORE candidate (Opus
  judges; CORE additions re-pin baselines via DECISION_LOG entry).
- New angles / hard lighting / far court → STRESS (verdict-exempt if the
  geometry is non-side-view — angle-aware geometry is parked).
- Undetectable-ball clips → FAILURE (honest-zero expectation).
- Update `CLIP_LABELS`, ground truth tables, docs/benchmark_suite.md, and the
  eval browser build so new clips appear.

## 6. Report

Letters↔files table, crash-run results, frames extracted/uploaded count, what
awaits the user (labels? annotation?), and which clips look CORE-worthy.
