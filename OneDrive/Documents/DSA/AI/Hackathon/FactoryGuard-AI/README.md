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
| **Phase 2** | Behaviour Intelligence (Video Streaming, Zone Entry/Exit, Dwell, Events) | ✅ Complete & Verified |
| **Phase 3** | Evidence & Incident Intelligence | 🔲 Pending |
| **Phase 4** | Dashboard & User Interface | 🔲 Pending |

---

## Phase 2 — Behaviour Intelligence (Video Pipeline)

Phase 2 builds directly upon Phase 1 tracking output without rebuilding or modifying YOLO + ByteTrack detection. It provides real-time spatial and temporal intelligence across video footage:

- **Full Video Frame-by-Frame Processing:** Streams video sequentially using `cv2.VideoCapture` and `cv2.VideoWriter` without loading the whole video into memory.
- **Interactive Multi-Zone Drawing:** Interactive OpenCV UI on the video's first frame allows operators to click polygon vertices to define custom zones.
- **Zone Configuration Persistence:** Saves zone coordinates, names, types (`restricted`, `warning`), and dwell thresholds into JSON (`data/zones/factory_zone_config.json`).
- **Point-in-Polygon Center Point Tracking:** Uses `cv2.pointPolygonTest` on each worker's midpoint `[cx, cy]` to determine whether they are inside/outside.
- **Temporal Event Detection:**
  - **Zone Entry:** Generates `restricted_zone_entry` or `warning_zone_entry` when transition occurs from outside to inside.
  - **Prolonged Presence / Dwell:** Measures stay duration (seconds) via video timestamps and triggers `prolonged_restricted_zone_presence` when `dwell_time >= DWELL_THRESHOLD_SECONDS` (default: 5.0s).
  - **Zone Exit:** Detects when worker leaves zone, resetting state to allow new entries.
  - **Independent Multi-Worker & Multi-Zone Tracking:** Concurrently tracks separate dwell timers and states for each worker across multiple safety zones.
- **Outputs Generated:**
  - **Real Annotated Output Video:** `outputs/phase2_output.mp4` with semi-transparent zone polygons, worker bounding boxes, worker IDs, center points, and safety status badges.
  - **Machine-Readable Behaviour Events JSON:** `data/behaviour/behaviour_events.json` structured for Phase 3 downstream incident handling.

### Phase 2 Pipeline

```
Factory Video (e.g. videos/test_factory.mp4)
       ↓
Display First Video Frame
       ↓
User Draws Safety Zones (or loads saved JSON)
       ↓
Save Polygon Coordinates (data/zones/factory_zone_config.json)
       ↓
Stream Video Frame-by-Frame
       ↓
Phase 1 YOLOv8s + ByteTrack Tracker
       ↓
Worker Midpoint [cx, cy] vs Polygon (cv2.pointPolygonTest)
       ↓
State Transitions (Entry / Exit / Prolonged Dwell)
       ↓
┌─────────────────────────────────┬──────────────────────────────────────┐
│ Real Annotated Output Video     │ Structured Behaviour Events (JSON)   │
│ outputs/phase2_output.mp4       │ data/behaviour/behaviour_events.json │
└─────────────────────────────────┴──────────────────────────────────────┘
```

---

## Phase 2 Usage & Commands

### 1. Run Complete Phase 2 on a Video

```bash
python phase2.py videos/test_factory.mp4
```

### 2. Draw / Re-draw Custom Safety Zones

To launch the interactive zone drawer on the first frame:

```bash
python phase2.py videos/test_factory.mp4 --draw-zones
```

**Interactive Zone Drawing Controls:**
- **Left Mouse Click:** Place polygon vertex points.
- **Key `r`:** Finalize current polygon as **RESTRICTED** zone (e.g. `Machine Zone A`).
- **Key `w`:** Finalize current polygon as **WARNING** zone (e.g. `Warning Corridor B`).
- **Key `c`:** Clear current points in progress.
- **Key `d`:** Delete / clear all defined zones.
- **Key `s` / ESC / `q`:** Save configuration and proceed to video analysis.

### 3. Customize Dwell Threshold & Output Paths

```bash
python phase2.py videos/test_factory.mp4 --dwell-threshold 5.0 --output-video outputs/phase2_output.mp4 --events data/behaviour/behaviour_events.json
```

### 4. Standalone Zone Configuration Utility

```bash
# Interactively define zones for a video
python zone_config.py --video videos/test_factory.mp4

# Inspect saved zones
python zone_config.py --list
```

---

## Zone Configuration Schema (`data/zones/factory_zone_config.json`)

```json
{
  "zones": [
    {
      "name": "Machine Zone A",
      "type": "restricted",
      "points": [
        [120, 150],
        [500, 150],
        [500, 450],
        [120, 450]
      ],
      "dwell_threshold": 5.0
    },
    {
      "name": "Warning Corridor B",
      "type": "warning",
      "points": [
        [550, 200],
        [850, 200],
        [850, 600],
        [550, 600]
      ],
      "dwell_threshold": 10.0
    }
  ]
}
```

---

## Behaviour Events Schema (`data/behaviour/behaviour_events.json`)

```json
{
  "events": [
    {
      "worker_id": 1,
      "event": "restricted_zone_entry",
      "timestamp": 7.3,
      "zone": "Machine Zone A",
      "severity": "high",
      "duration": 0.0
    },
    {
      "worker_id": 1,
      "event": "prolonged_restricted_zone_presence",
      "timestamp": 9.3,
      "zone": "Machine Zone A",
      "severity": "critical",
      "duration": 2.0
    }
  ]
}
```

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
