"""
FactoryGuard AI - Detection Package

Phase 1: Computer Vision Foundation
Provides person detection and tracking using YOLO + ByteTrack.
Supports both pre-recorded video files and live webcam streams.
"""

from .tracker import WorkerTracker, process_video, process_webcam, get_tracking_data

__all__ = ["WorkerTracker", "process_video", "process_webcam", "get_tracking_data"]
