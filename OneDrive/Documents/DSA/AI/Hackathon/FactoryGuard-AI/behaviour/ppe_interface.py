"""
FactoryGuard AI - PPE Interface & Extensibility Layer

Phase 2: Behaviour and Safety Intelligence Layer

Provides structured representation and extensible interface for worker PPE (Personal
Protective Equipment) status.

CRITICAL PRINCIPLE (Prompt Constraint):
    PPE predictions are NEVER faked. If a dedicated model is present, it will update
    helmet, vest, and boots status. Otherwise, all attributes default to 'unknown'.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger("factoryguard.behaviour.ppe_interface")


@dataclass
class WorkerPPEState:
    """Represents the PPE status of a specific worker.

    Attributes:
        worker_id: Persistent tracking ID assigned by ByteTrack.
        helmet: True if wearing helmet, False if missing, None if unknown.
        vest: True if wearing safety vest, False if missing, None if unknown.
        boots: True if wearing safety boots, False if missing, None if unknown.
    """
    worker_id: int
    helmet: Optional[bool] = None  # None = unknown
    vest: Optional[bool] = None    # None = unknown
    boots: Optional[bool] = None   # None = unknown

    def to_dict(self) -> Dict[str, str]:
        """Convert PPE state to plain dictionary with clear 'unknown'/'present'/'missing' labels."""
        def format_attr(val: Optional[bool]) -> str:
            if val is True:
                return "present"
            elif val is False:
                return "missing"
            return "unknown"

        return {
            "helmet": format_attr(self.helmet),
            "vest": format_attr(self.vest),
            "boots": format_attr(self.boots),
        }


class PPEInterface:
    """Interface for querying worker PPE compliance status.

    Can be extended or wrapped with an actual downstream CV classifier (e.g. YOLO-PPE).
    """

    def __init__(self):
        # Cache of known worker states
        self._worker_ppe_cache: Dict[int, WorkerPPEState] = {}
        self._custom_detector = None

    def register_custom_detector(self, detector_func) -> None:
        """Register a custom model inference function to analyze worker crops.

        Args:
            detector_func: Function accepting (frame, worker_bbox) -> WorkerPPEState
        """
        self._custom_detector = detector_func
        logger.info("Custom PPE detector registered with PPEInterface.")

    def get_worker_ppe(self, worker_id: int, frame=None, bbox=None) -> WorkerPPEState:
        """Retrieve current PPE status for a worker.

        Args:
            worker_id: Persistent tracking ID.
            frame: Optional full BGR image frame (used if custom detector is registered).
            bbox: Optional worker bounding box [x1, y1, x2, y2].

        Returns:
            WorkerPPEState instance. Defaults to 'unknown' unless custom detector is active.
        """
        if self._custom_detector is not None and frame is not None and bbox is not None:
            try:
                state = self._custom_detector(frame, bbox)
                state.worker_id = worker_id
                self._worker_ppe_cache[worker_id] = state
                return state
            except Exception as e:
                logger.warning("Custom PPE detector failed for worker %d: %s", worker_id, e)

        # Return cached or default unknown state
        if worker_id not in self._worker_ppe_cache:
            self._worker_ppe_cache[worker_id] = WorkerPPEState(
                worker_id=worker_id,
                helmet=None,
                vest=None,
                boots=None,
            )
        return self._worker_ppe_cache[worker_id]

    def update_worker_ppe(self, worker_id: int, helmet: Optional[bool] = None, vest: Optional[bool] = None, boots: Optional[bool] = None) -> None:
        """Manually update or set worker PPE attributes."""
        current = self.get_worker_ppe(worker_id)
        if helmet is not None:
            current.helmet = helmet
        if vest is not None:
            current.vest = vest
        if boots is not None:
            current.boots = boots
        self._worker_ppe_cache[worker_id] = current
