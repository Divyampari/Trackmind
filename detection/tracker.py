"""
FactoryGuard AI - Worker Detection & Tracking Module

Phase 1: Computer Vision Foundation

This module provides reliable person detection and tracking for factory
safety monitoring. It uses Ultralytics YOLO (pretrained YOLOv8s) for person
detection and tuned ByteTrack for persistent worker ID assignment across
frames, with low false positives and high ID stability during walking,
cycling, and brief occlusions.

Supports:
    - Pre-recorded video files (MP4, AVI, MKV, etc.)
    - Live webcam feeds (default laptop webcam or USB camera)

Output Contract (for Phase 2+ integration):
    - Structured tracking data as JSON (tracking_data.json or webcam_tracking_data.json)
    - Annotated output video with bounding boxes & worker IDs
    - Frame-by-frame worker records with:
        * worker ID (persistent across frames where possible)
        * bounding box [x1, y1, x2, y2] in pixel coordinates
        * center point [cx, cy]
        * frame number (0-indexed)
        * timestamp (derived from video FPS or session elapsed time, in seconds)
        * detection confidence (0.0 - 1.0)

Coordinate Conventions:
    - Bounding box: [x1, y1, x2, y2] where (x1,y1) is top-left, (x2,y2) is bottom-right
    - Center: [cx, cy] computed as midpoint of bounding box
    - All coordinates are in pixels relative to the original video/webcam resolution
    - Origin (0, 0) is the top-left corner of the frame

Usage:
    from detection.tracker import process_video, process_webcam, get_tracking_data

    # Video file processing
    tracking_data = process_video("videos/my_test.mp4")

    # Live webcam processing
    tracking_data = process_webcam(camera_index=0)

    # Load previously saved tracking data
    tracking_data = get_tracking_data("data/tracking/tracking_data.json")
"""

import json
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Union

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("factoryguard.tracker")

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "yolov8s.pt"          # Pretrained YOLOv8 Small (higher precision)
DEFAULT_CONFIDENCE = 0.45             # Tuned confidence threshold (suppresses noise/poles)
PERSON_CLASS_ID = 0                   # COCO class ID for 'person'
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_TRACKING_DIR = os.path.join("data", "tracking")
DEFAULT_OUTPUT_VIDEO = "tracked_output.mp4"
DEFAULT_TRACKING_FILE = "tracking_data.json"
DEFAULT_WEBCAM_TRACKING_FILE = "webcam_tracking_data.json"
DEFAULT_WEBCAM_OUTPUT_VIDEO = "webcam_output.mp4"
DEFAULT_TRACKER_CFG = os.path.join("models", "trackers", "bytetrack_factoryguard.yaml")

# Annotation colours & style
BOX_COLOR = (0, 255, 128)             # Green bounding box (BGR)
TEXT_COLOR = (255, 255, 255)          # White text
LABEL_BG_COLOR = (40, 40, 40)        # Dark background for labels
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2


# ---------------------------------------------------------------------------
# Helper: Ensure tuned ByteTrack configuration exists
# ---------------------------------------------------------------------------
def ensure_default_tracker_config(cfg_path: str = DEFAULT_TRACKER_CFG) -> str:
    """Create the tuned ByteTrack configuration file if it does not exist."""
    if not os.path.isfile(cfg_path):
        os.makedirs(os.path.dirname(cfg_path) or ".", exist_ok=True)
        tuned_cfg = {
            "tracker_type": "bytetrack",
            "track_high_thresh": 0.45,
            "track_low_thresh": 0.10,
            "new_track_thresh": 0.55,
            "track_buffer": 90,
            "match_thresh": 0.85,
            "fuse_score": True,
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(tuned_cfg, f)
        logger.debug("Generated tuned ByteTrack config at: %s", cfg_path)
    return cfg_path


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class WorkerRecord:
    """A single worker detection in a single frame.

    Attributes:
        id: Persistent tracking ID assigned by ByteTrack.
        bbox: Bounding box as [x1, y1, x2, y2] in pixels.
        center: Center point as [cx, cy] in pixels.
        confidence: Detection confidence score (0.0 - 1.0).
    """
    id: int
    bbox: list
    center: list
    confidence: float

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for JSON serialisation."""
        return {
            "id": self.id,
            "bbox": [round(float(c), 1) for c in self.bbox],
            "center": [round(float(c), 1) for c in self.center],
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class FrameRecord:
    """All worker detections for a single video frame.

    Attributes:
        frame: Frame number (0-indexed).
        timestamp: Time position in seconds.
        workers: List of WorkerRecord instances detected in this frame.
    """
    frame: int
    timestamp: float
    workers: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for JSON serialisation."""
        return {
            "frame": int(self.frame),
            "timestamp": round(float(self.timestamp), 4),
            "workers": [w.to_dict() for w in self.workers],
        }


@dataclass
class TrackingData:
    """Complete tracking output for an entire video or webcam session.

    Attributes:
        video: Source video filename or 'webcam_<id>'.
        fps: Frames per second of the stream.
        total_frames: Total number of frames processed.
        resolution: Video resolution as [width, height].
        frames: List of FrameRecord instances.
    """
    video: str
    fps: float
    total_frames: int
    resolution: list
    frames: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for JSON serialisation."""
        return {
            "video": self.video,
            "fps": round(float(self.fps), 2),
            "total_frames": int(self.total_frames),
            "resolution": [int(self.resolution[0]), int(self.resolution[1])],
            "frames": [f.to_dict() for f in self.frames],
        }

    def save(self, filepath: str) -> None:
        """Save tracking data to a JSON file.

        Args:
            filepath: Destination path for the JSON file.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Tracking data saved to: %s", filepath)


# ---------------------------------------------------------------------------
# Core tracker class
# ---------------------------------------------------------------------------
class WorkerTracker:
    """Detects and tracks workers in factory video footage or live webcam.

    Uses Ultralytics YOLO for person detection and ByteTrack for
    multi-object tracking with persistent ID assignment.

    Args:
        model_path: Path or name of the YOLO model (auto-downloads if needed).
        confidence: Minimum detection confidence threshold.
        tracker_config: Path to tracker YAML config (defaults to tuned ByteTrack).
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        confidence: float = DEFAULT_CONFIDENCE,
        tracker_config: Optional[str] = None,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.tracker_config = tracker_config or ensure_default_tracker_config()
        self.model = None

    def _load_model(self) -> None:
        """Load the YOLO model. Downloads automatically if not present."""
        try:
            logger.info("Loading YOLO model: %s", self.model_path)
            self.model = YOLO(self.model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model '{self.model_path}': {e}\n"
                f"Ensure the model name is correct. Ultralytics models like "
                f"'yolov8s.pt' or 'yolov8n.pt' are downloaded automatically on first use."
            ) from e

    def process_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        tracking_output_path: Optional[str] = None,
    ) -> TrackingData:
        """Process an entire video: detect, track, annotate, and export data.

        Args:
            video_path: Path to the input video file.
            output_video_path: Path for the annotated output video.
                Defaults to 'outputs/tracked_output.mp4'.
            tracking_output_path: Path for the tracking JSON file.
                Defaults to 'data/tracking/tracking_data.json'.

        Returns:
            TrackingData: Structured tracking data for the entire video.

        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If the video cannot be opened.
            RuntimeError: If the model fails to load.
        """
        # --- Validate input ---
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # --- Set default output paths ---
        if output_video_path is None:
            output_video_path = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_VIDEO)
        if tracking_output_path is None:
            tracking_output_path = os.path.join(
                DEFAULT_TRACKING_DIR, DEFAULT_TRACKING_FILE
            )

        # --- Ensure output directories exist ---
        os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(tracking_output_path) or ".", exist_ok=True)

        # --- Load model ---
        if self.model is None:
            self._load_model()

        # --- Open video ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(
                f"Cannot open video file: {video_path}\n"
                f"Ensure the file is a valid video format (mp4, avi, etc.)."
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "Video opened: %s | %dx%d @ %.1f FPS | ~%d frames",
            video_path, width, height, fps, total_frames_estimate,
        )

        video_filename = os.path.basename(video_path)

        # --- Initialise video writer ---
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(
                f"Cannot create output video: {output_video_path}\n"
                f"Check that the output directory is writable."
            )

        # --- Initialise tracking data ---
        tracking_data = TrackingData(
            video=video_filename,
            fps=fps,
            total_frames=0,
            resolution=[width, height],
        )

        # --- Process frames ---
        frame_number = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_number / fps

                # Run YOLO tracking with tuned ByteTrack
                frame_record = self._process_frame(frame, frame_number, timestamp)
                tracking_data.frames.append(frame_record)

                # Annotate frame
                annotated = self._annotate_frame(frame, frame_record)
                writer.write(annotated)

                frame_number += 1

                # Progress logging every 100 frames
                if frame_number % 100 == 0:
                    pct = (frame_number / max(total_frames_estimate, 1)) * 100
                    logger.info(
                        "Processed %d / ~%d frames (%.1f%%)",
                        frame_number, total_frames_estimate, pct,
                    )

        except Exception as e:
            logger.error("Error processing frame %d: %s", frame_number, e)
            raise
        finally:
            cap.release()
            writer.release()

        tracking_data.total_frames = frame_number

        # --- Save tracking data ---
        tracking_data.save(tracking_output_path)

        logger.info(
            "Processing complete. %d frames processed.", frame_number
        )
        logger.info("Output video: %s", output_video_path)
        logger.info("Tracking data: %s", tracking_output_path)

        return tracking_data

    def process_webcam(
        self,
        camera_index: int = 0,
        output_video_path: Optional[str] = None,
        tracking_output_path: Optional[str] = None,
        show_preview: bool = True,
        max_frames: Optional[int] = None,
    ) -> TrackingData:
        """Process live video from a webcam feed with real-time tracking.

        Uses the same YOLO + ByteTrack tracking logic as process_video to ensure
        100% compatibility with downstream Phase 2 contracts.

        Args:
            camera_index: Index of the webcam device (default: 0).
            output_video_path: Optional path to record the annotated webcam session.
            tracking_output_path: Path for tracking JSON output.
                Defaults to 'data/tracking/webcam_tracking_data.json'.
            show_preview: If True, displays live OpenCV preview window (press 'q' to stop).
            max_frames: Optional max frames limit (useful for automated testing).

        Returns:
            TrackingData: Structured tracking data for the webcam session.

        Raises:
            RuntimeError: If the webcam cannot be opened.
        """
        # --- Set default output path ---
        if tracking_output_path is None:
            tracking_output_path = os.path.join(
                DEFAULT_TRACKING_DIR, DEFAULT_WEBCAM_TRACKING_FILE
            )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(tracking_output_path) or ".", exist_ok=True)
        if output_video_path:
            os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)

        # --- Load model ---
        if self.model is None:
            self._load_model()

        # --- Open webcam ---
        logger.info("Opening webcam at camera index: %d", camera_index)
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam (camera index {camera_index}).\n"
                f"Troubleshooting tips:\n"
                f"  1. Ensure your laptop/USB webcam is connected and enabled.\n"
                f"  2. Check if another application (e.g. Zoom, Teams, Camera) is using the webcam.\n"
                f"  3. Try a different camera index using --camera-id (e.g. 1 or 2)."
            )

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = raw_fps if (raw_fps is not None and raw_fps > 0) else 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        logger.info(
            "Webcam active: Camera %d | %dx%d @ ~%.1f FPS",
            camera_index, width, height, fps
        )

        # --- Optional video recorder ---
        writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                logger.warning("Could not initialize video writer for %s. Continuing without recording.", output_video_path)
                writer = None

        # --- Initialise tracking data ---
        tracking_data = TrackingData(
            video=f"webcam_{camera_index}",
            fps=fps,
            total_frames=0,
            resolution=[width, height],
        )

        window_name = f"FactoryGuard AI - Live Webcam (Camera {camera_index}) | Press 'q' to stop"
        session_start_time = time.time()
        frame_number = 0

        print()
        print("=" * 60)
        print("  LIVE WEBCAM STREAMING ACTIVE")
        print(f"  Camera Index : {camera_index}")
        print(f"  Resolution   : {width}x{height}")
        print("  Press 'q' or 'ESC' in the preview window to stop tracking.")
        print("=" * 60)
        print()

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to grab frame from webcam. Ending session.")
                    break

                # Session-relative timestamp
                timestamp = time.time() - session_start_time

                # Run shared YOLO + ByteTrack logic
                frame_record = self._process_frame(frame, frame_number, timestamp)
                tracking_data.frames.append(frame_record)

                # Annotate frame
                annotated = self._annotate_frame(frame, frame_record, is_live=True)

                if writer is not None:
                    writer.write(annotated)

                if show_preview:
                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == ord("Q") or key == 27:
                        logger.info("User requested stop via keyboard.")
                        break

                frame_number += 1

                if max_frames is not None and frame_number >= max_frames:
                    logger.info("Reached max frame limit (%d).", max_frames)
                    break

        except Exception as e:
            logger.error("Error during webcam stream: %s", e)
            raise
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show_preview:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

        tracking_data.total_frames = frame_number

        # --- Save tracking data ---
        tracking_data.save(tracking_output_path)

        logger.info(
            "Webcam session ended. %d frames processed.", frame_number
        )
        if output_video_path and writer is not None:
            logger.info("Webcam recorded output: %s", output_video_path)
        logger.info("Webcam tracking data saved: %s", tracking_output_path)

        return tracking_data

    def _process_frame(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> FrameRecord:
        """Run detection and tracking on a single frame.

        Args:
            frame: The video frame as a NumPy array (BGR).
            frame_number: Current frame index (0-based).
            timestamp: Frame timestamp in seconds.

        Returns:
            FrameRecord with all detected workers in this frame.
        """
        frame_record = FrameRecord(frame=frame_number, timestamp=timestamp)

        try:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_config,
                conf=self.confidence,
                classes=[PERSON_CLASS_ID],
                verbose=False,
            )
        except Exception as e:
            logger.warning(
                "Tracking failed on frame %d: %s. Returning empty frame.",
                frame_number, e,
            )
            return frame_record

        if not results or results[0].boxes is None:
            return frame_record

        boxes = results[0].boxes

        for i in range(len(boxes)):
            # Extract bounding box
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

            # Extract confidence
            conf = float(boxes.conf[i].cpu().numpy())

            # Extract tracking ID
            track_id = -1
            if boxes.id is not None:
                track_id = int(boxes.id[i].cpu().numpy())

            # Compute center
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            worker = WorkerRecord(
                id=track_id,
                bbox=[x1, y1, x2, y2],
                center=[cx, cy],
                confidence=conf,
            )
            frame_record.workers.append(worker)

        return frame_record

    def _annotate_frame(
        self, frame: np.ndarray, frame_record: FrameRecord, is_live: bool = False
    ) -> np.ndarray:
        """Draw bounding boxes and labels on a video frame.

        Args:
            frame: Original video frame (BGR).
            frame_record: FrameRecord containing workers to annotate.
            is_live: Whether this frame is part of a live webcam stream.

        Returns:
            Annotated frame as a NumPy array.
        """
        annotated = frame.copy()

        for worker in frame_record.workers:
            x1, y1, x2, y2 = [int(c) for c in worker.bbox]
            track_id = worker.id
            conf = worker.confidence

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

            # Build label
            if track_id >= 0:
                label = f"Worker {track_id} | Person | {conf:.2f}"
            else:
                label = f"Person | {conf:.2f}"

            # Measure text size for background rectangle
            (text_w, text_h), baseline = cv2.getTextSize(
                label, FONT, FONT_SCALE, FONT_THICKNESS
            )
            label_y = max(y1 - 10, text_h + 5)

            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, label_y - text_h - 5),
                (x1 + text_w + 8, label_y + baseline),
                LABEL_BG_COLOR,
                -1,
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1 + 4, label_y),
                FONT,
                FONT_SCALE,
                TEXT_COLOR,
                FONT_THICKNESS,
            )

        # Draw frame info overlay
        status_prefix = "LIVE WEBCAM" if is_live else "FRAME"
        info_text = f"{status_prefix}: {frame_record.frame} | Workers: {len(frame_record.workers)} | t={frame_record.timestamp:.2f}s"
        cv2.putText(
            annotated, info_text, (10, 30), FONT, 0.7, (0, 255, 255), 2
        )

        if is_live:
            cv2.putText(
                annotated,
                "Press 'q' to stop session",
                (10, annotated.shape[0] - 15),
                FONT,
                0.5,
                (200, 200, 200),
                1,
            )

        return annotated


# ---------------------------------------------------------------------------
# Module-level convenience functions (stable public API)
# ---------------------------------------------------------------------------
def process_video(
    video_path: str,
    model_path: str = DEFAULT_MODEL,
    confidence: float = DEFAULT_CONFIDENCE,
    tracker_config: Optional[str] = None,
    output_video_path: Optional[str] = None,
    tracking_output_path: Optional[str] = None,
) -> TrackingData:
    """Process a video file and produce tracking data + annotated video.

    Primary entry point for video-file processing in Phase 1.

    Args:
        video_path: Path to the input video file.
        model_path: YOLO model name or path (default: yolov8s.pt).
        confidence: Minimum detection confidence (default: 0.45).
        tracker_config: Optional custom ByteTrack YAML config path.
        output_video_path: Path for annotated output video.
        tracking_output_path: Path for tracking JSON output.

    Returns:
        TrackingData: Complete structured tracking data.
    """
    tracker = WorkerTracker(
        model_path=model_path,
        confidence=confidence,
        tracker_config=tracker_config,
    )
    return tracker.process_video(
        video_path,
        output_video_path=output_video_path,
        tracking_output_path=tracking_output_path,
    )


def process_webcam(
    camera_index: int = 0,
    model_path: str = DEFAULT_MODEL,
    confidence: float = DEFAULT_CONFIDENCE,
    tracker_config: Optional[str] = None,
    output_video_path: Optional[str] = None,
    tracking_output_path: Optional[str] = None,
    show_preview: bool = True,
    max_frames: Optional[int] = None,
) -> TrackingData:
    """Process a live webcam stream and produce tracking data (+ optional recorded video).

    Primary entry point for webcam processing in Phase 1. Uses the same underlying
    YOLO + ByteTrack logic as process_video.

    Args:
        camera_index: Camera device index (default: 0).
        model_path: YOLO model name or path (default: yolov8s.pt).
        confidence: Minimum detection confidence (default: 0.45).
        tracker_config: Optional custom ByteTrack YAML config path.
        output_video_path: Path to record annotated output video (optional).
        tracking_output_path: Path for tracking JSON output (default: data/tracking/webcam_tracking_data.json).
        show_preview: Whether to display live preview window (press 'q' to quit).
        max_frames: Optional frame limit.

    Returns:
        TrackingData: Complete structured tracking data.
    """
    tracker = WorkerTracker(
        model_path=model_path,
        confidence=confidence,
        tracker_config=tracker_config,
    )
    return tracker.process_webcam(
        camera_index=camera_index,
        output_video_path=output_video_path,
        tracking_output_path=tracking_output_path,
        show_preview=show_preview,
        max_frames=max_frames,
    )


def get_tracking_data(tracking_json_path: str) -> dict:
    """Load previously saved tracking data from a JSON file.

    This function allows Phase 2+ modules to consume Phase 1 output
    without importing or depending on YOLO/ByteTrack internals.

    Args:
        tracking_json_path: Path to the tracking JSON file.

    Returns:
        dict: Parsed tracking data dictionary.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    if not os.path.isfile(tracking_json_path):
        raise FileNotFoundError(
            f"Tracking data file not found: {tracking_json_path}"
        )

    with open(tracking_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(
        "Loaded tracking data: %s (%d frames)",
        tracking_json_path,
        len(data.get("frames", [])),
    )
    return data
