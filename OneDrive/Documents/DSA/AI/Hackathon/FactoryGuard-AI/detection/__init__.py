"""
FactoryGuard AI - Detection Package

Phase 1: Computer Vision Foundation
Provides person detection and tracking using YOLO + ByteTrack.
"""

from .tracker import WorkerTracker, process_video, get_tracking_data

__all__ = ["WorkerTracker", "process_video", "get_tracking_data"]
