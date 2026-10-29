# Whole-Person Tracking

CPU-friendly webcam tracking that associates a verified face with a full-body
track. When the face turns away or is briefly occluded, the tracker can keep
following the body as `TEMPORARY`; it returns to `UNKNOWN` when confidence
decays too far.

```text
camera/video -> person detector -> body tracker
                         \-> face detector -> ArcFace identity verification
body + face association -> identity state -> pan/tilt centring command
```

The repository also includes a deterministic, model-free simulator for head
turns, occlusion, crossings, identity swaps, and vertical movement. Simulator
logs use the same CSV format as live runs.

## Features

- Multi-person tracking with HOG, Haar upper-body, YOLOv8 ONNX, or YOLOX ONNX.
- ArcFace ONNX verification with YuNet or Haar face detection.
- `CONFIRMED`, `TEMPORARY`, and `UNKNOWN` identity states.
- Appearance, overlap, mismatch, and confidence-decay safeguards.
- Virtual-camera pan/tilt centring with optional MQTT motor commands.
- CSV logs, metrics, Markdown reports, and timeline plots.
- Logic tests and simulation that run without a webcam or ONNX models.

## Requirements and installation

- Python 3.10–3.13
- A webcam for live tracking
- `opencv-python`, `numpy`, `onnxruntime`, `mediapipe`, `matplotlib`
- `paho-mqtt` when using `--mqtt`

From the repository root, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install opencv-python numpy onnxruntime mediapipe matplotlib paho-mqtt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

The default live configuration expects these local model files:

```text
models/embedder_arcface.onnx
models/face_detection_yunet_2023mar.onnx
models/yolox_nano.onnx
```

Model files, face databases, enrollment crops, and captured frames are not
included. Treat enrolled face data as private and do not commit it.

## Live tracking

Enroll the target with the ArcFace workflow, then start tracking:

```powershell
python -m src.enroll
python track_person.py --target Christella
```

Useful alternatives:

```powershell
# Model-free, weaker OpenCV descriptor backend
python track_person.py --target Christella --backend simple

# Track a video file and stop after 300 frames
python track_person.py --target Christella --source path\to\video.mp4 --max-frames 300

# Publish pan/tilt commands to the local MQTT broker
python track_person.py --target Christella --mqtt
```

Live runs write `logs/run_<timestamp>.csv` and print the matching evaluation
command on exit. The default person detector is YOLOX; try
`--person-detector hog` or `--person-detector upper` as model-free fallbacks.
The script tries camera index `1` first and falls back when needed; use
`--camera` to select another device.

### Live controls

| Key | Action |
| --- | --- |
| `q` | Quit |
| `r` | Reset identity and centring |
| `f` | Toggle fullscreen |
| `v` | Toggle virtual-camera picture-in-picture |
| `e` | Enrol with the simple backend |
| `1` | Label the current segment `normal` |
| `2` | Label a head-turn event |
| `3` | Label an occlusion event |
| `4` | Label a crossing event |
| `w` | Flag that the tracker is on the wrong person |

## Face-recognition utilities

The original face-recognition modules remain available independently:

```powershell
python -m src.camera       # Webcam smoke test
python -m src.detect       # Face-detection preview
python -m src.align        # Five-point alignment preview
python -m src.embed        # ArcFace embedding preview
python -m src.enroll       # Build/update data/db/face_db.npz
python -m src.recognize    # Multi-face recognition
python -m src.evaluate     # Evaluate enrolled samples
```

The ArcFace database is stored at `data/db/face_db.npz`, metadata at
`data/db/face_db.json`, and aligned enrollment crops under `data/enroll/`.

## Tests, simulation, and evaluation

Run the model-free test suite:

```powershell
python -m unittest test_whole_person -v
```

The simulator is exposed through `src.sim.run_scenario` and includes
`head_turn_short`, `head_turn_long`, `occlusion`, `crossing`,
`crossing_similar`, `crossing_swap`, `crossing_bounce`, and `vertical`.

Run the crossing stress comparison:

```powershell
python run_stress.py --seeds 10
```

This writes `reports/stress.md` and compares the normal three-state logic with
a deliberately unsafe “never let go” baseline.

Evaluate live or simulator CSV logs:

```powershell
python evaluate_run.py logs\run_20261022_164253.csv
python evaluate_run.py runs\sim_*.csv --out reports
```

The evaluator can produce `<name>_metrics.json`, `<name>_report.md`, and
`<name>_timeline.png`. Reports cover state time, wrong-person honesty,
continuity during events, confidence decay, centring error, and command jitter.

## MQTT motor control

`--mqtt` connects to `127.0.0.1:1883` and publishes changed commands:

```text
face_centering/motor/horizontal -> PAN_LEFT | PAN_RIGHT | PAN_STOP
face_centering/motor/vertical   -> TILT_UP | TILT_DOWN | TILT_STOP
```

The tracker stops both axes when identity is not actively locked and during
shutdown. Change the broker settings in `src/mqtt_controller.py` if needed.

## Project layout

```text
whole_person_tracking/
├── track_person.py       Live webcam/video entry point
├── evaluate_run.py       Run-log metrics and report generator
├── run_stress.py         Crossing stress-test runner
├── test_whole_person.py  Model-free test suite
├── src/
│   ├── whole_person.py   Detection, association, identity, and pipeline
│   ├── person_tracker.py Body detectors and multi-person tracking
│   ├── identity_state.py CONFIRMED/TEMPORARY/UNKNOWN state machine
│   ├── centering.py      Virtual-camera and pan/tilt control
│   ├── backends.py       ArcFace, YuNet, Haar, and simple backends
│   ├── sim.py            Scripted scenarios and synthetic frames
│   └── runlog.py         Shared CSV logging
├── models/               Local ONNX models (not committed)
├── data/                 Enrollment database and samples
├── logs/                 Live run logs and reports
├── runs/                 Simulator run logs
└── reports/              Evaluation reports, metrics, and plots
```

## Limitations and privacy

This is a research/demo tracker, not an access-control or safety system.
Lighting, camera placement, detector quality, similar clothing, and people
crossing can affect identity continuity. Validate thresholds and failure modes
on your own camera before connecting motors.

The application processes camera frames locally, but embeddings, enrollment
crops, logs, and reports may still be biometric or identifying data. Protect or
delete them when no longer needed.
