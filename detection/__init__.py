"""
FactoryGuard AI - Detection Package

Phase 1 & Phase 2 Integration Layer
Provides person detection, ByteTrack tracking, and worker-level PPE detection.
"""

from .tracker import WorkerTracker, process_video, process_webcam, get_tracking_data
from .ppe_detector import PPEDetector, PPEResult, parse_ppe_predictions, load_env_file

__all__ = [
    "WorkerTracker",
    "process_video",
    "process_webcam",
    "get_tracking_data",
    "PPEDetector",
    "PPEResult",
    "parse_ppe_predictions",
    "load_env_file",
]
