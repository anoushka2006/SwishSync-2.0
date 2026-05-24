# SwishSync Repository Workflow

SwishSync uses a three-layer branch model:

```text
main      stable, demo-ready code only
dev       active integration branch
feature/* isolated feature/module work
```

## Branch roles

### `main`

- Keep deployable or demo-ready.
- Merge from `dev` only after the CV pipeline is verified.
- Do not commit experiments, raw videos, outputs, or model weights directly here.

### `dev`

- Use for active integration work across modules.
- Merge feature branches here first.
- Run tests before promoting to `main`.

### `feature/*`

- Use for focused work such as:
  - `feature/ball-tracking`
  - `feature/shot-detection`
  - `feature/mediapipe-pose`
  - `feature/trajectory-fitting`
  - `feature/insight-engine`
  - `feature/frontend-ui`

## Local artifact policy

The following should stay local and untracked:

- `videos/`
- `outputs/`
- `models/`
- YOLO weights such as `*.pt`
- processed videos such as `*.mp4` and `*.MOV`
- generated detection exports and debug frames

Use small synthetic fixtures in tests instead of committing real videos.

## Recommended flow

```bash
git checkout dev
git pull origin dev
git checkout -b feature/ball-tracking

# make focused changes
pytest

git add <changed source/test/doc files>
git commit -m "Improve ball tracking"
git push -u origin feature/ball-tracking
```

After review, merge the feature branch into `dev`. Promote `dev` to `main` only
when the repository is stable and demo-ready.
