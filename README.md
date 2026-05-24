# SwishSync 2.0

Foundational computer vision module for SwishSync:

```text
local basketball video -> frame extraction -> YOLOv8 detections -> ball trajectory -> annotated video
```

This repository intentionally contains only the first stable CV pipeline. It does
not include a full app, pose estimation, or frontend work.

## Scope

The module supports:

1. Loading a local basketball video.
2. Extracting frames with OpenCV.
3. Running YOLOv8 object detection on every frame with CPU inference.
4. Detecting basketball and hoop classes when the configured model exposes them.
5. Saving frame-by-frame detection coordinates.
6. Tracking the basketball center across frames.
7. Drawing detection boxes and trajectory overlays.
8. Exporting a processed video.

> Important model note: stock COCO YOLOv8 models include `sports ball`, which is
> used as the default basketball alias. They do not include a dedicated hoop
> class. Hoop detection requires custom YOLOv8 weights with a class name such as
> `hoop`, `basketball hoop`, `rim`, or another configured alias.

## Folder structure

```text
src/swishsync_cv/
  cli.py                  # Local command-line entry point
  config.py               # Pipeline, detection, and visualization config
  data.py                 # Shared frame/detection/trajectory data contracts
  pipeline.py             # End-to-end orchestration
  detection/
    yolo.py               # YOLOv8 adapter and class-name filtering
  io/
    video.py              # OpenCV video reader/writer
  tracking/
    trajectory.py         # Basketball trajectory tracking
  visualization/
    overlay.py            # OpenCV drawing utilities
  utils/
    serialization.py      # JSONL/CSV/JSON debug exports
tests/                    # Independently testable module tests
```

## Processing pipeline

1. `PipelineConfig` validates the input/output intent.
2. `VideoReader` opens the local video and streams `FramePacket` objects.
3. `YoloObjectDetector` runs YOLOv8 on each frame using `device="cpu"`.
4. YOLO outputs are normalized into `DetectionRecord` objects for:
   - `basketball`
   - `hoop`
5. `BallTrajectoryTracker` selects the highest-confidence basketball detection
   per frame and records its center.
6. `annotate_frame` draws:
   - basketball bounding boxes
   - hoop bounding boxes
   - accumulated ball trajectory
   - optional frame index
7. `VideoWriter` exports the annotated video.
8. Serialization utilities save:
   - `detections.jsonl`: one record per frame
   - `detections.csv`: one row per detected object
   - `trajectory.json`: tracked ball center points

## Data flow

```text
input video path
  -> VideoReader
  -> FramePacket(index, timestamp_ms, image)
  -> YoloObjectDetector
  -> DetectionRecord[]
  -> BallTrajectoryTracker
  -> TrajectoryPoint[]
  -> annotate_frame
  -> VideoWriter
  -> processed video + detection artifacts
```

## Component guide

### `io.video`

- **Why it exists:** isolates OpenCV file handling from detection and tracking.
- **Inputs:** local video path, output video path, FPS, frame size, codec.
- **Outputs:** `FramePacket` objects and processed video files.
- **Common failures:** missing video path, unsupported codec, zero FPS, invalid
  frame dimensions, output frame size mismatch.
- **Debugging strategy:** inspect OpenCV metadata first; verify FPS/width/height;
  run on a short clip; confirm the writer codec works on the target machine.

### `detection.yolo`

- **Why it exists:** wraps Ultralytics YOLOv8 and converts model-specific output
  into stable SwishSync detection records.
- **Inputs:** frame image, frame index, timestamp, `DetectionConfig`.
- **Outputs:** `DetectionRecord` objects with label, class name, confidence, bbox,
  and center coordinates.
- **Common failures:** missing model file, first-run weight download, class alias
  mismatch, low confidence threshold, hoop class absent from stock weights.
- **Debugging strategy:** inspect `detections.csv`, lower confidence threshold for
  experiments, check raw model class names, and use custom hoop-trained weights.

### `tracking.trajectory`

- **Why it exists:** keeps temporal ball trajectory state independent from YOLO.
- **Inputs:** ordered detections for each frame.
- **Outputs:** `TrajectoryPoint` objects for selected basketball centers.
- **Common failures:** no basketball detection in a frame, multiple basketball-like
  detections, detector briefly locks onto another round object.
- **Debugging strategy:** compare selected point confidence with all basketball
  detections in `detections.csv`; inspect frame-level overlays around misses.

### `visualization.overlay`

- **Why it exists:** centralizes drawing logic for repeatable visual debugging.
- **Inputs:** a frame, detections, trajectory points, `VideoOutputConfig`.
- **Outputs:** an annotated frame ready for video export.
- **Common failures:** bbox coordinates outside frame bounds, empty trajectory,
  output video dimensions that do not match the original frame.
- **Debugging strategy:** enable `--save-debug-frames` and inspect JPEG snapshots
  without scrubbing through the full video.

### `utils.serialization`

- **Why it exists:** makes detections and trajectories inspectable outside video.
- **Inputs:** `FrameDetections`, `DetectionRecord`, and `TrajectoryPoint` objects.
- **Outputs:** JSONL, CSV, and JSON artifacts.
- **Common failures:** missing output directory permissions or unexpected empty
  detections from the model.
- **Debugging strategy:** start with `detections.jsonl` to confirm frame coverage,
  then use `detections.csv` for filtering/sorting by class and confidence.

### `pipeline`

- **Why it exists:** wires independent modules into the required end-to-end flow.
- **Inputs:** `PipelineConfig` and, optionally, any detector implementing
  `detect(frame, frame_index, timestamp_ms)`.
- **Outputs:** `PipelineResult` with output paths and count summaries.
- **Common failures:** any upstream I/O/model issue, very slow CPU inference on
  long videos, empty trajectory when the ball is not detected.
- **Debugging strategy:** run on a trimmed clip, enable debug frames, inspect
  counts printed by the CLI, and review exported detection artifacts.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

Run with the default small YOLOv8 model:

```bash
swishsync-cv \
  --input path/to/basketball_clip.mp4 \
  --output-dir outputs
```

Run with custom weights that include hoop/rim classes:

```bash
swishsync-cv \
  --input path/to/basketball_clip.mp4 \
  --model path/to/custom_hoop_ball_yolov8.pt \
  --output-dir outputs \
  --save-debug-frames \
  --debug-frame-stride 15
```

Outputs:

```text
outputs/
  processed.mp4
  detections.jsonl
  detections.csv
  trajectory.json
  debug_frames/          # only when --save-debug-frames is used
```

## Tests

```bash
pytest
```

The tests use synthetic frames and fake detectors where possible so individual
modules remain testable without downloading YOLO weights.
