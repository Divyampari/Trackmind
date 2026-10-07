# FactoryGuard AI

**AI-Powered Factory Safety Monitoring System**

> HackNex 2026 — Problem Statement PS07: Autonomous Vision & Behaviour Understanding

---

## Overview

FactoryGuard AI is an AI-powered factory safety monitoring system that analyses factory video footage or live webcam streams to detect and track workers, inspect PPE compliance, understand behavior, identify safety violations, and generate evidence-backed incidents.

The project is built collaboratively across **4 phases**:

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1** | Computer Vision Foundation (Detection & Tracking) | ✅ Video & Live Webcam Support |
| **PPE Detection** | Worker PPE Understanding (Helmet, Vest, Boots via Roboflow) | ✅ Integrated & Cached |
| **Phase 2** | Behaviour Intelligence & Zone Monitoring | 🔲 In Progress |
| **Phase 3** | Evidence & Incident Intelligence | 🔲 Pending |
| **Phase 4** | Dashboard & User Interface | 🔲 Pending |

---

## Architecture

```
Factory Video / Live Webcam
        ↓
   OpenCV (frame-by-frame stream / webcam capture)
        ↓
   YOLOv8s Person Detection (COCO Class 0, conf=0.45)
        ↓
   Tuned ByteTrack Multi-Object Tracking (track_buffer=90, new_track_thresh=0.55)
        ↓
   Persistent Worker IDs
        ↓
   Worker BBox Crop -> Roboflow Hosted Inference ("helmet-vest-and-boots-detection/8")
        ↓
   Worker-Level PPE Cache (refreshed periodically, default: 1.0s)
        ↓
   ┌──────────────┬──────────────────────────────┐
   │ Annotated    │ Structured Tracking          │
   │ Video/Stream │ Data + PPE (JSON)            │
   └──────────────┴──────────────────────────────┘
```

---

## Technology Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core language |
| **Ultralytics YOLO** | Person detection (pretrained YOLOv8s) |
| **ByteTrack** | Multi-object tracking (persistent worker IDs across frames) |
| **Roboflow Hosted Inference** | PPE detection (`helmet-vest-and-boots-detection/8`) via `inference-sdk` |
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
├── .gitignore                      # Git ignore rules (.env, .pt, outputs, data)
├── .env.example                    # Environment variable template
│
├── detection/
│   ├── __init__.py                 # Package exports
│   ├── tracker.py                  # Detection & ByteTrack tracking pipeline
│   └── ppe_detector.py             # Roboflow PPE detection & worker-level caching
│
├── data/
│   └── tracking/
│       ├── tracking_data.json      # [Generated] Pre-recorded video tracking + PPE output
│       └── webcam_tracking_data.json # [Generated] Live webcam tracking + PPE output
│
├── videos/                         # Place pre-recorded input videos here
│   └── my_test.mp4
│
├── outputs/
│   ├── tracked_output.mp4          # [Generated] Annotated video file output with PPE badges
│   └── webcam_output.mp4           # [Generated] Optional recorded webcam session
│
├── models/
│   └── trackers/
│       └── bytetrack_factoryguard.yaml # Tuned ByteTrack configuration
│
└── tests/
    ├── __init__.py
    ├── test_integration.py         # Output contract & Phase 2 integration test
    ├── test_ppe.py                 # Roboflow PPE parser, caching & fallback tests
    ├── test_webcam.py              # Webcam stream & schema verification tests
    ├── test_configurations.py      # Diagnostic grid evaluation script
    ├── create_sample_video.py      # Synthetic factory video generator
    └── create_person_test_video.py # Real person test video generator
```

---

## Installation & Setup

### 1. Activate Virtual Environment

```bash
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Roboflow API Key (Optional for PPE Detection)

Copy `.env.example` to `.env` and set your key:

```bash
copy .env.example .env
```

Inside `.env`:
```ini
ROBOFLOW_API_KEY=your_actual_key_here
```

> **Note:** If `ROBOFLOW_API_KEY` is not provided, the pipeline continues running smoothly with PPE status set to `"unknown"`.

---

## Running the Program

### Mode 1: Pre-recorded Video File

```bash
# Process factory video with tracking and PPE detection
python app.py videos/my_test.mp4

# Configure PPE refresh interval (e.g., sample PPE every 1.5 seconds per worker)
python app.py videos/my_test.mp4 --ppe-interval 1.5

# Disable PPE detection (tracking only)
python app.py videos/my_test.mp4 --disable-ppe
```

### Mode 2: Live Webcam Stream

```bash
# Start live webcam tracking (default camera 0)
python app.py --source webcam

# Shortcut flag
python app.py --webcam

# Record live webcam session to MP4
python app.py --source webcam --record-webcam
```

> **Live Controls:** Press **`q`** or **`ESC`** in the preview window to stop tracking and save results.

---

## Tracking & PPE Data Schema

Output file: `data/tracking/tracking_data.json`

```json
{
  "video": "my_test.mp4",
  "fps": 30.0,
  "total_frames": 857,
  "resolution": [1280, 714],
  "frames": [
    {
      "frame": 52,
      "timestamp": 1.7333,
      "workers": [
        {
          "id": 1,
          "bbox": [928.9, 0.0, 963.3, 53.7],
          "center": [946.1, 26.9],
          "confidence": 0.5936,
          "ppe": {
            "helmet": true,
            "vest": true,
            "boots": "unknown"
          }
        }
      ]
    }
  ]
}
```

### PPE Field Contract

| PPE Item | Value | Meaning |
|---|---|---|
| `helmet` | `true` | Hard hat confirmed detected |
| | `false` | Missing helmet confirmed detected ('no helmet') |
| | `"unknown"` | Indeterminate or low confidence |
| `vest` | `true` / `false` / `"unknown"` | Safety vest compliance status |
| `boots` | `true` / `false` / `"unknown"` | Safety boots compliance status |

---

## PPE Detection Architecture & Optimizations

1. **Worker Bounding Box Cropping:** Only cropped worker regions are sent to Roboflow inference. Full-resolution frames are never transmitted unnecessarily.
2. **Worker-Level Caching:** PPE predictions are associated with persistent ByteTrack worker IDs.
3. **Temporal Sampling:** PPE inference is executed at a configurable sampling interval (`--ppe-interval`, default: `1.0s` per worker). Between intervals, cached results are reused, reducing API calls by ~97% at 30 FPS.
4. **Fault Tolerance:** If network latency, rate limits, or API outages occur, the detector logs a concise warning and falls back to `"unknown"`, ensuring uninterrupted video tracking.

---

## Running Automated Tests

```bash
# 1. Run PPE unit, caching & failure tests
python -m unittest tests/test_ppe.py

# 2. Run Webcam integration tests
python -m unittest tests/test_webcam.py

# 3. Run Output contract & Phase 2 consumption tests
python tests/test_integration.py
```

---

## Pretrained Models & Attribution

- **Person Detection:** Ultralytics YOLOv8s (pretrained on COCO dataset).
- **Tracking:** ByteTrack algorithm via Ultralytics API.
- **PPE Classification:** Roboflow Hosted Inference Model `helmet-vest-and-boots-detection/8`.
