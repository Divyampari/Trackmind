````markdown
# Trackmind

### Autonomous Vision & Behaviour Understanding for Factory Safety

Trackmind is an AI-powered factory safety monitoring system developed for **HackNex 2026 – Problem Statement PS07: Autonomous Vision & Behaviour Understanding**.

The project aims to analyze factory video footage, detect and track workers, understand their behaviour over time, identify safety violations, and generate evidence-backed incidents.

---

## Phase 1 – Computer Vision Foundation

Phase 1 establishes the computer vision foundation of Trackmind.

The current pipeline is:

**Factory Video → YOLO Person Detection → ByteTrack Tracking → Persistent Worker IDs → Annotated Video**

At this stage, the system focuses only on detecting and tracking people. Behaviour understanding, safety-zone analysis, incident generation, evidence management, and dashboard functionality are implemented in later phases.

---

## Technology Stack

- Python
- Ultralytics YOLO
- ByteTrack
- OpenCV
- NumPy

### Detection

A pretrained **YOLOv8s** model is used for person detection.

Only the COCO `person` class is processed in Phase 1.

### Tracking

**ByteTrack** is used through the Ultralytics tracking API to maintain persistent worker identities across video frames.

This allows the system to distinguish between different workers and maintain their identity over time.

---

## Project Structure

```text
Trackmind/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── detection/
│   ├── __init__.py
│   └── tracker.py
│
├── videos/
│
├── models/
│
├── data/
│   └── tracking/
│
├── outputs/
│
└── tests/
````

---

## Installation

Create and activate a Python virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The required YOLO model is downloaded automatically by Ultralytics when it is needed.

---

## Input Video

Place a test video containing people inside:

```text
videos/
```

For example:

```text
videos/test_factory.mp4
```

The application does not depend on a specific hardcoded filename. A video path can be provided when running the application.

---

## Running Trackmind

To process a video:

```bash
python app.py videos/test_factory.mp4
```

The system processes the video frame-by-frame and generates an annotated output.

The processed video is saved in:

```text
outputs/tracked_output.mp4
```

---

## Webcam Support

Trackmind also supports webcam input.

Run:

```bash
python app.py --source webcam
```

The webcam feed is processed in real time with YOLO detection and ByteTrack tracking.

Press `q` or `ESC` to stop the webcam processing.

An optional webcam recording can also be enabled:

```bash
python app.py --source webcam --record-webcam
```

---

## Detection and Tracking Pipeline

For every input frame:

1. OpenCV reads the frame.
2. YOLO detects people in the frame.
3. Detection confidence is obtained for each person.
4. ByteTrack associates detections across consecutive frames.
5. Each tracked person receives a persistent worker ID whenever possible.
6. The bounding box, center position, confidence, frame number, and timestamp are recorded.
7. The worker information is drawn onto the frame.
8. The annotated frame is written to the output video.

The system does not load the entire video into memory. Frames are processed sequentially.

---

## Annotated Output

Each detected worker is displayed with:

* Bounding box
* Worker tracking ID
* Person label
* Detection confidence

Example:

```text
Worker 3 | Person | 0.89
```

This provides a visual demonstration of both detection and tracking.

---

## Tracking Data

Trackmind also produces structured tracking information for future behaviour-analysis modules.

A typical worker record contains:

```json
{
    "id": 1,
    "bbox": [895.6, 260.3, 1079.4, 580.4],
    "center": [987.5, 420.4],
    "confidence": 0.8827,
    "frame": 0,
    "timestamp": 0.0
}
```

The tracking data is stored under:

```text
data/tracking/
```

This structured output allows later phases to consume worker movement information without modifying the detection layer.

---

## Why Tracking Is Important

Simple object detection only answers:

> "Is there a person in this frame?"

Tracking adds temporal understanding by answering:

> "Which worker is this, and where have they been across frames?"

Persistent worker IDs are important for the later behaviour-analysis phase because behaviours such as zone entry and prolonged presence depend on observing the same worker over time.

---

## Phase 1 Scope

Implemented in Phase 1:

* Person detection
* Bounding-box detection
* Detection confidence
* ByteTrack tracking
* Persistent worker IDs
* Worker center coordinates
* Frame numbers
* Timestamps
* Annotated video generation
* Structured tracking data
* Local video input
* Webcam input

Not implemented in Phase 1:

* Behaviour classification
* Restricted-zone detection
* Prolonged-presence detection
* Severity classification
* Incident generation
* Evidence management
* Streamlit dashboard
* General anomaly detection

---

## Testing

Phase 1 was tested for:

* Video input handling
* Person detection
* Tracking ID generation
* Tracking ID persistence across frames
* Multiple-worker tracking
* Output video generation
* Tracking data generation
* Invalid input handling
* Webcam input
* Webcam tracking
* Clean webcam shutdown

The webcam pipeline was also tested with a live camera feed, confirming that frames could be processed and worker tracking information generated successfully.

Automated webcam tests were run using:

```bash
python -m unittest tests/test_webcam.py
```

---

## Limitations

The accuracy of detection and tracking depends on:

* Camera quality
* Lighting conditions
* Worker visibility
* Occlusion
* Number of people in the scene
* Camera angle
* Movement between frames

ByteTrack may temporarily lose or change an ID when a worker is heavily occluded or leaves and re-enters the scene.

Phase 1 is a prototype foundation and is not intended to replace certified industrial safety systems.

---

## Pretrained Model Attribution

Trackmind uses the **Ultralytics YOLO** framework and a pretrained YOLOv8 model for person detection.

YOLO is used as the perception component, while ByteTrack provides multi-object tracking through the Ultralytics tracking interface.

Users should follow the applicable licensing and attribution requirements of the libraries and pretrained models used in their deployment.

---

## Future Development

Trackmind is designed as a multi-phase system.

### Phase 2 – Behaviour Intelligence

The tracking output will be used to understand worker behaviour, including:

* Configurable safety zones
* Restricted-zone entry
* Prolonged presence
* Dwell-time analysis
* Severity classification

### Phase 3 – Evidence & Incident Intelligence

Behaviour events will be converted into structured incidents containing:

* Incident ID
* Worker ID
* Timestamp
* Location/zone
* Behaviour type
* Severity
* Duration
* Evidence frames
* Incident history

### Phase 4 – Dashboard

A Streamlit dashboard will provide:

* Worker count
* Safety alerts
* Incident history
* Severity information
* Evidence images
* Processed video
* Worker and incident summaries

These features are planned for later phases and are not part of the current Phase 1 implementation.

---

## Project Vision

Trackmind is designed to move beyond simple CCTV monitoring.

The overall system will follow:

**Detect → Track → Understand → Identify → Prove → Act**

The long-term objective is to transform raw factory video into meaningful, explainable safety information by identifying **who** is involved, **where** they are, **what** they are doing, **when** it happened, **how long** it continued, and **why** it should be considered a safety concern.

```
```
<img width="1600" height="754" alt="img1" src="https://github.com/user-attachments/assets/cbfb8541-85de-43e9-99f6-d3cb85ffec6b" />
<img width="749" height="506" alt="img2" src="https://github.com/user-attachments/assets/c78b8ed1-aa75-404c-9fa9-7698d39500e9" />
<img width="717" height="498" alt="img3" src="https://github.com/user-attachments/assets/11f48d57-7b69-408d-8975-32e17248a13a" />
<img width="338" height="155" alt="img4" src="https://github.com/user-attachments/assets/4c33eb42-d707-424c-930c-1ef31a744506" />
<img width="1541" height="839" alt="img5" src="https://github.com/user-attachments/assets/be52ecc8-6964-4220-8196-8db7539569e7" />
<img width="1411" height="380" alt="img6" src="https://github.com/user-attachments/assets/2319bbcb-e756-487b-9b26-f2c78f71e4b4" />
<img width="1600" height="618" alt="image" src="https://github.com/user-attachments/assets/ec1b9f11-101e-4ecc-bef6-a45ac5094636" />

