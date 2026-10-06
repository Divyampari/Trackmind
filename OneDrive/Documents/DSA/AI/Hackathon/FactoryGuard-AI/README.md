# FactoryGuard AI

**AI-Powered Factory Safety Monitoring System**

> HackNex 2026 — Problem Statement PS07: Autonomous Vision & Behaviour Understanding

---

## Overview

FactoryGuard AI is an AI-powered factory safety monitoring system that analyses factory video footage to detect and track workers, understand their behaviour, identify safety violations, and generate evidence-backed incidents.

The project is built collaboratively across **4 phases**:

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1** | Computer Vision Foundation (Detection & Tracking) | ✅ Refined & Verified |
| **Phase 2** | Behaviour Intelligence | 🔲 Pending |
| **Phase 3** | Evidence & Incident Intelligence | 🔲 Pending |
| **Phase 4** | Dashboard & User Interface | 🔲 Pending |

---

## Phase 1 — Computer Vision Foundation

Phase 1 provides the foundational computer vision pipeline:

- **Person detection** using a pretrained YOLOv8 Small model (`yolov8s.pt` by Ultralytics)
- **Worker tracking** with persistent IDs using tuned ByteTrack
- **Annotated output video** with bounding boxes, worker IDs, and confidence scores
- **Structured tracking data** (JSON) for downstream consumption by Phase 2+

### Pipeline

```
Factory Video (MP4/AVI/etc.)
        ↓
   OpenCV (frame-by-frame reading)
        ↓
   YOLOv8s Person Detection (COCO Class 0, conf=0.45)
        ↓
   Tuned ByteTrack Multi-Object Tracking (track_buffer=90, new_track_thresh=0.55)
        ↓
   Persistent Worker IDs
        ↓
   ┌──────────────┬──────────────────────┐
   │ Annotated    │ Structured Tracking  │
   │ Output Video │ Data (JSON)          │
   └──────────────┴──────────────────────┘
```

> **Note:** Phase 1 detects and tracks people only. It does **not** perform behaviour analysis, zone detection, or incident generation. Those capabilities are implemented in subsequent phases.

---

## Technology Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core language |
| **Ultralytics YOLO** | Person detection (pretrained YOLOv8s) |
| **ByteTrack** | Multi-object tracking (tuned parameters for factory video) |
| **OpenCV** | Video I/O and frame annotation |
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
│   └── tracker.py                  # Core detection & tracking module
│
├── data/
│   └── tracking/
│       └── tracking_data.json      # [Generated] Tracking output
│
├── videos/                         # Place input videos here
│   └── my_test.mp4                 # Test video
│
├── outputs/
│   └── tracked_output.mp4          # [Generated] Annotated video
│
├── models/
│   └── trackers/
│       └── bytetrack_factoryguard.yaml # Tuned ByteTrack configuration
│
└── tests/
    ├── __init__.py
    ├── test_integration.py         # Phase 1 output contract test
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

### Run with Default Settings (YOLOv8s + Tuned ByteTrack)

```bash
python app.py videos/my_test.mp4
```

### Advanced Options

```bash
python app.py videos/my_test.mp4 --model yolov8s.pt --confidence 0.45 --tracker-config models/trackers/bytetrack_factoryguard.yaml
```

---

## Tracking Data Schema (`data/tracking/tracking_data.json`)

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
          "confidence": 0.5936
        }
      ]
    }
  ]
}
```

- **`bbox` format:** `[x1, y1, x2, y2]` (top-left to bottom-right in pixels)
- **`center` format:** `[cx, cy]`
- **`timestamp`:** Derived from `frame_number / fps`
- **`id`:** Integer tracking ID (`-1` if unassigned)

---

## Refinement Summary: False Positives & ID Switching

### Diagnostics on `videos/my_test.mp4` (857 frames)

1. **Problem 1 — Pole False Positive:**
   - *Baseline (YOLOv8n @ conf=0.35):* Generated 256 false detections on a vertical pole (Track ID 20 & 55).
   - *Root Cause:* The Nano model's low parameter capacity confused the vertical pole texture with a standing person at lower confidence levels (<0.40).
   - *Fix:* Upgrading to **YOLOv8s** (`yolov8s.pt`) and setting confidence threshold to **0.45** completely eliminated the pole false positive (0 stationary pole detections).

2. **Problem 2 — Track ID Switching & Flickers:**
   - *Baseline:* Generated 33 fragmented unique track IDs with 15 short flicker tracks (<10 frames).
   - *Root Cause:* Default ByteTrack uses `track_buffer: 30` (only 1 second buffer) and `new_track_thresh: 0.25`. When a person walked or rode a bicycle past obstacles or experienced brief confidence dips, the track died within 1 second and immediately spawned a new ID upon reappearance.
   - *Fix:* Tuned ByteTrack configuration (`models/trackers/bytetrack_factoryguard.yaml`):
     - `track_buffer: 90` (3 seconds memory window to bridge occlusions/cycling)
     - `new_track_thresh: 0.55` (prevents spawning spurious tracks from low-confidence noise)
     - `track_high_thresh: 0.45` & `track_low_thresh: 0.10` (recovers weak detections during movement)
     - `match_thresh: 0.85` (enhanced IoU association)

### Before vs After Comparison on `videos/my_test.mp4`

| Metric | Baseline (`yolov8n` + default tracker) | Refined (`yolov8s` + tuned ByteTrack) | Improvement |
|---|---|---|---|
| **Unique Track IDs** | 33 | **10** | **70% reduction in fragmentation** |
| **Static Pole Detections** | 256 | **0** | **100% eliminated** |
| **Flicker Tracks (<10 frames)** | 15 | **0** | **100% eliminated** |
| **Track Continuity** | 50% - 70% | **90.4% - 100%** | **Continuous tracking across entire trajectories** |

---

## Integration Test

Run the decoupled contract test:

```bash
python tests/test_integration.py
```

---

## Pretrained Models & Attribution

- **Detection:** Ultralytics YOLOv8s (pretrained on COCO dataset). *Not trained by our team.*
- **Tracking:** ByteTrack algorithm via Ultralytics API with custom factory monitoring parameters.

---

## Limitations

- **Prolonged Occlusions (>3s):** If a worker is occluded for more than 90 consecutive frames (~3 seconds) and reappears far away, a new tracking ID may be assigned.
- **Extreme Crowding:** Multiple overlapping workers moving in close proximity may occasionally experience identity swaps if bounding boxes severely overlap.
- **Hardware Requirement:** YOLOv8s runs at ~5.7 FPS on CPU and >60 FPS on GPU.
