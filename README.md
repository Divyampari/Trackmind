# FactoryGuard AI

**AI-Powered Factory Safety Monitoring System**

> HackNex 2026 — Problem Statement PS07: Autonomous Vision & Behaviour Understanding

---

## Overview

FactoryGuard AI is an AI-powered factory safety monitoring system that analyses factory video footage or live webcam streams to detect and track workers, understand their behaviour, identify safety violations, and generate evidence-backed incidents.

The project is built collaboratively across **4 phases**:

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1** | Computer Vision Foundation (Detection & Tracking) | ✅ Video & Live Webcam Support |
| **Phase 2** | Behaviour Intelligence | 🔲 Pending |
| **Phase 3** | Evidence & Incident Intelligence | 🔲 Pending |
| **Phase 4** | Dashboard & User Interface | 🔲 Pending |

---

## Phase 1 — Computer Vision Foundation

Phase 1 provides the foundational computer vision pipeline:

- **Person detection** using a pretrained YOLOv8 Small model (`yolov8s.pt` by Ultralytics)
- **Worker tracking** with persistent IDs using tuned ByteTrack
- **Dual Input Modes:** Pre-recorded video files (`.mp4`, `.avi`, etc.) and **Live Webcam streams**
- **Annotated video / live display** with bounding boxes, worker IDs, and confidence scores
- **Structured tracking data** (JSON) for downstream consumption by Phase 2+

### Pipeline

```
Factory Video / Live Webcam
        ↓
   OpenCV (frame-by-frame reading / webcam capture)
        ↓
   YOLOv8s Person Detection (COCO Class 0, conf=0.45)
        ↓
   Tuned ByteTrack Multi-Object Tracking (track_buffer=90, new_track_thresh=0.55)
        ↓
   Persistent Worker IDs
        ↓
   ┌──────────────┬──────────────────────────────┐
   │ Annotated    │ Structured Tracking          │
   │ Video/Stream │ Data (JSON)                  │
   └──────────────┴──────────────────────────────┘
```

> **Note:** Phase 1 detects and tracks people only. It does **not** perform behaviour analysis, zone detection, or incident generation. Those capabilities are implemented in subsequent phases.

---

## Technology Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core language |
| **Ultralytics YOLO** | Person detection (pretrained YOLOv8s) |
| **ByteTrack** | Multi-object tracking (tuned parameters for factory video & live webcam) |
| **OpenCV** | Video I/O, webcam capture, live display, and frame annotation |
| **NumPy** | Numerical and geometric operations |

---

## Project Structure

```
FactoryGuard-AI/
│
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation
├── .gitignore                      # Git ignore rules
│
├── detection/
│   ├── __init__.py                 # Package exports
│   └── tracker.py                  # Core detection & tracking module (video + webcam)
│
├── data/
│   └── tracking/
│       ├── tracking_data.json      # [Generated] Pre-recorded video tracking output
│       └── webcam_tracking_data.json # [Generated] Live webcam tracking output
│
├── videos/                         # Place pre-recorded input videos here
│   └── my_test.mp4
│
├── outputs/
│   ├── tracked_output.mp4          # [Generated] Annotated video file output
│   └── webcam_output.mp4           # [Generated] Optional recorded webcam session
│
├── models/
│   └── trackers/
│       └── bytetrack_factoryguard.yaml # Tuned ByteTrack configuration
│
└── tests/
    ├── __init__.py
    ├── test_integration.py         # Phase 1 output contract test
    ├── test_webcam.py              # Webcam stream & schema verification tests
    ├── test_configurations.py      # Diagnostic grid evaluation script
    ├── create_sample_video.py      # Synthetic factory video generator
    └── create_person_test_video.py # Real person test video generator
```

---

## Installation

```bash
# 1. Activate virtual environment
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running the Program

### Mode 1: Video File Tracking

```bash
# Specify video file directly
python app.py videos/my_test.mp4

# Or using the --source flag
python app.py --source videos/my_test.mp4

# Auto-detects first video in videos/ directory
python app.py
```

### Mode 2: Live Webcam Tracking

```bash
# Start live webcam tracking (default laptop/USB camera 0)
python app.py --source webcam

# Shortcut flag
python app.py --webcam

# Specify a specific camera index (e.g., external USB camera 1)
python app.py --source webcam --camera-id 1

# Optional: Record the live webcam session to an MP4 video
python app.py --source webcam --record-webcam
```

> **Live Controls:** When the live webcam window is active, press **`q`** or **`ESC`** at any time to stop tracking and automatically save `data/tracking/webcam_tracking_data.json`.

### All Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `video` | Auto-detect | Path to input video file |
| `--source` | `None` | Input source: `'webcam'` (or `'0'`) or path to video file |
| `--webcam` | `False` | Shortcut to launch live webcam tracking |
| `--camera-id` | `0` | Camera device index for webcam mode |
| `--record-webcam` | `False` | Record annotated live webcam session to MP4 |
| `--no-preview` | `False` | Headless mode (disables OpenCV GUI window) |
| `--model` | `yolov8s.pt` | YOLO model path or name |
| `--confidence` | `0.45` | Minimum detection confidence (0.0–1.0) |
| `--tracker-config` | `models/trackers/bytetrack_factoryguard.yaml` | ByteTrack tracker config |
| `--output-video` | `outputs/tracked_output.mp4` | Custom output video path |
| `--output-tracking` | `data/tracking/tracking_data.json` | Custom tracking JSON path |
| `--verbose` / `-v` | `False` | Enable debug logging |

---

## Tracking Data Schema

Both video-file mode (`tracking_data.json`) and webcam mode (`webcam_tracking_data.json`) output the **exact same structured schema**:

```json
{
  "video": "webcam_0",
  "fps": 30.0,
  "total_frames": 150,
  "resolution": [640, 480],
  "frames": [
    {
      "frame": 0,
      "timestamp": 0.033,
      "workers": [
        {
          "id": 1,
          "bbox": [120.0, 80.5, 260.3, 400.1],
          "center": [190.2, 240.3],
          "confidence": 0.8842
        }
      ]
    }
  ]
}
```

- **`bbox` format:** `[x1, y1, x2, y2]` (top-left to bottom-right in pixels)
- **`center` format:** `[cx, cy]`
- **`timestamp`:** Derived from video FPS or session elapsed time (seconds)
- **`id`:** Integer persistent track ID assigned by ByteTrack (`-1` if unassigned)

---

## Programmatic API

```python
from detection.tracker import process_video, process_webcam, get_tracking_data

# 1. Process video file
video_data = process_video("videos/my_test.mp4")

# 2. Process live webcam stream
webcam_data = process_webcam(camera_index=0, show_preview=True)

# 3. Load previously saved data (Phase 2 consumption)
data = get_tracking_data("data/tracking/webcam_tracking_data.json")
```

---

## Running Automated Tests

```bash
# Test contract validation & Phase 2 consumption demo
python tests/test_integration.py

# Test webcam integration & schema validation
python -m unittest tests/test_webcam.py
```

---

## Pretrained Models & Attribution

- **Detection:** Ultralytics YOLOv8s (pretrained on COCO dataset). *Not trained by our team.*
- **Tracking:** ByteTrack algorithm via Ultralytics API with custom factory monitoring parameters.

---

## Limitations

- **Webcam Framerate:** Real-time processing speed depends on CPU/GPU hardware.
- **Lighting Conditions:** Poor or low-light factory webcam streams may reduce detection recall.
- **Prolonged Occlusions (>3s):** If a worker leaves camera field of view for >3 seconds, a new track ID may be assigned upon return.
