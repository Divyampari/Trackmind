"""
FactoryGuard AI - Zone Detector

Phase 2: Behaviour and Safety Intelligence Layer

Tracks spatial interactions between tracked workers (from Phase 1) and safety zones.
Detects:
    - Worker Zone Entry
    - Worker Zone Exit
    - Worker Dwell Time (duration spent inside a zone)
    - Prolonged Restricted Presence violations

Includes state tracking, debouncing, and grace periods to prevent duplicate alerts
during temporary ByteTrack tracking gaps or frame jitter.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any

from behaviour.zone_manager import Zone, ZoneType

logger = logging.getLogger("factoryguard.behaviour.zone_detector")


class EventType(str, Enum):
    """Event types generated during worker-zone state transitions."""
    RESTRICTED_ENTRY = "restricted_zone_entry"
    RESTRICTED_DWELL = "prolonged_restricted_zone_presence"
    RESTRICTED_DWELL_ALT = "restricted_zone_dwell"
    RESTRICTED_EXIT = "restricted_zone_exit"
    WARNING_ENTRY = "warning_zone_entry"
    WARNING_DWELL = "prolonged_warning_zone_presence"
    WARNING_DWELL_ALT = "warning_zone_dwell"
    WARNING_EXIT = "warning_zone_exit"


@dataclass
class ZoneEvent:
    """Represents a spatial or state transition event of a worker relative to a safety zone.

    Attributes:
        event_type: One of EventType strings.
        worker_id: Persistent tracking ID assigned by Phase 1 ByteTrack.
        zone_name: Identifier of the zone.
        zone_type: 'restricted' or 'warning'.
        timestamp: Time in seconds relative to video start.
        frame: Frame number (0-indexed).
        dwell_time: Time in seconds worker has spent inside the zone.
        worker_bbox: Bounding box [x1, y1, x2, y2].
        worker_center: Center point [cx, cy].
    """
    event_type: str
    worker_id: int
    zone_name: str
    zone_type: str
    timestamp: float
    frame: int
    dwell_time: float
    worker_bbox: List[float]
    worker_center: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to plain dictionary matching Phase 2 specification."""
        sev = "critical" if ("prolonged" in self.event_type or "dwell" in self.event_type) else (
            "warning" if self.zone_type == "restricted" or "warning" in self.event_type else "info"
        )
        return {
            "worker_id": int(self.worker_id),
            "event": self.event_type,
            "event_type": self.event_type,
            "timestamp": round(float(self.timestamp), 2),
            "zone": self.zone_name,
            "zone_name": self.zone_name,
            "zone_type": self.zone_type,
            "severity": sev,
            "duration": round(float(self.dwell_time), 2),
            "dwell_time": round(float(self.dwell_time), 2),
            "frame": int(self.frame),
            "worker_bbox": [round(float(c), 1) for c in self.worker_bbox],
            "worker_center": [round(float(c), 1) for c in self.worker_center],
        }


@dataclass
class WorkerZoneState:
    """Internal state tracker for a specific (worker_id, zone_name) pair."""
    worker_id: int
    zone_name: str
    zone_type: str
    entry_frame: int
    entry_timestamp: float
    last_seen_frame: int
    last_seen_timestamp: float
    dwell_time: float = 0.0
    entry_alerted: bool = False
    dwell_alerted: bool = False
    worker_bbox: List[float] = field(default_factory=list)
    worker_center: List[float] = field(default_factory=list)


class ZoneDetector:
    """Stateful detector monitoring worker-zone spatial interactions frame-by-frame.

    Args:
        anchor_point: Point on worker bounding box to evaluate inside zone.
            'center': Midpoint [cx, cy] of bounding box (default & prompt preferred).
            'feet': Bottom-center midpoint [cx, y2] (optional alternative).
        grace_period: Time tolerance in seconds for brief tracking gaps before declaring exit.
    """

    def __init__(self, anchor_point: str = "center", grace_period: float = 1.0):
        self.anchor_point = anchor_point.lower()
        self.grace_period = grace_period
        # Key: (worker_id, zone_name) -> WorkerZoneState
        self.active_states: Dict[Tuple[int, str], WorkerZoneState] = {}

    def _get_worker_anchor(self, worker) -> List[float]:
        """Extract evaluation anchor point from WorkerRecord or dict."""
        if hasattr(worker, "bbox"):
            bbox = worker.bbox
            center = worker.center
        else:
            bbox = worker.get("bbox", [0, 0, 0, 0])
            center = worker.get("center", [0, 0])

        if self.anchor_point == "feet":
            cx = center[0]
            cy = bbox[3]  # y2 bottom
            return [cx, cy]
        return [float(center[0]), float(center[1])]

    def update(
        self,
        frame_number: int,
        timestamp: float,
        workers: List[Any],
        zones: List[Zone],
    ) -> List[ZoneEvent]:
        """Process a single frame update with current workers and safety zones.

        Args:
            frame_number: Current frame index (0-based).
            timestamp: Video timestamp in seconds.
            workers: List of WorkerRecord instances or worker dicts from Phase 1.
            zones: List of active Zone instances.

        Returns:
            List of newly generated ZoneEvent instances for this frame.
        """
        events: List[ZoneEvent] = []
        currently_inside_keys = set()

        for worker in workers:
            # Handle both WorkerRecord object and plain dict
            w_id = worker.id if hasattr(worker, "id") else worker["id"]
            if w_id < 0:
                # Skip unassigned or untracked background noise
                continue

            bbox = worker.bbox if hasattr(worker, "bbox") else worker["bbox"]
            center = worker.center if hasattr(worker, "center") else worker["center"]
            anchor = self._get_worker_anchor(worker)

            for zone in zones:
                key = (w_id, zone.name)
                is_inside = zone.contains_point(anchor)

                if is_inside:
                    currently_inside_keys.add(key)

                    if key not in self.active_states:
                        # --- WORKER ENTRY DETECTED ---
                        state = WorkerZoneState(
                            worker_id=w_id,
                            zone_name=zone.name,
                            zone_type=zone.type,
                            entry_frame=frame_number,
                            entry_timestamp=timestamp,
                            last_seen_frame=frame_number,
                            last_seen_timestamp=timestamp,
                            dwell_time=0.0,
                            entry_alerted=True,
                            dwell_alerted=False,
                            worker_bbox=bbox,
                            worker_center=center,
                        )
                        self.active_states[key] = state

                        e_type = (
                            EventType.RESTRICTED_ENTRY.value
                            if zone.type == ZoneType.RESTRICTED.value
                            else EventType.WARNING_ENTRY.value
                        )
                        event = ZoneEvent(
                            event_type=e_type,
                            worker_id=w_id,
                            zone_name=zone.name,
                            zone_type=zone.type,
                            timestamp=timestamp,
                            frame=frame_number,
                            dwell_time=0.0,
                            worker_bbox=bbox,
                            worker_center=center,
                        )
                        events.append(event)
                        logger.info(
                            "Worker %d ENTERED %s zone '%s' at t=%.2fs",
                            w_id, zone.type.upper(), zone.name, timestamp
                        )
                    else:
                        # --- WORKER CONTINUES INSIDE ZONE ---
                        state = self.active_states[key]
                        state.last_seen_frame = frame_number
                        state.last_seen_timestamp = timestamp
                        state.dwell_time = timestamp - state.entry_timestamp
                        state.worker_bbox = bbox
                        state.worker_center = center

                        # Check dwell-time threshold breach
                        if state.dwell_time >= zone.dwell_threshold and not state.dwell_alerted:
                            state.dwell_alerted = True
                            d_type = (
                                EventType.RESTRICTED_DWELL.value
                                if zone.type == ZoneType.RESTRICTED.value
                                else EventType.WARNING_DWELL.value
                            )
                            event = ZoneEvent(
                                event_type=d_type,
                                worker_id=w_id,
                                zone_name=zone.name,
                                zone_type=zone.type,
                                timestamp=timestamp,
                                frame=frame_number,
                                dwell_time=state.dwell_time,
                                worker_bbox=bbox,
                                worker_center=center,
                            )
                            events.append(event)
                            logger.warning(
                                "Worker %d DWELL THRESHOLD BREACH (%.1fs) in %s zone '%s'",
                                w_id, state.dwell_time, zone.type.upper(), zone.name
                            )

        # --- CHECK WORKER EXITS (WITH GRACE PERIOD FOR TEMP TRACKING LOSS) ---
        expired_keys = []
        for key, state in list(self.active_states.items()):
            if key not in currently_inside_keys:
                time_since_last_seen = timestamp - state.last_seen_timestamp
                if time_since_last_seen >= self.grace_period:
                    x_type = (
                        EventType.RESTRICTED_EXIT.value
                        if state.zone_type == ZoneType.RESTRICTED.value
                        else EventType.WARNING_EXIT.value
                    )
                    event = ZoneEvent(
                        event_type=x_type,
                        worker_id=state.worker_id,
                        zone_name=state.zone_name,
                        zone_type=state.zone_type,
                        timestamp=timestamp,
                        frame=frame_number,
                        dwell_time=state.dwell_time,
                        worker_bbox=state.worker_bbox,
                        worker_center=state.worker_center,
                    )
                    events.append(event)
                    logger.info(
                        "Worker %d EXITED %s zone '%s' (total dwell: %.1fs)",
                        state.worker_id, state.zone_type.upper(), state.zone_name, state.dwell_time
                    )
                    expired_keys.append(key)

        for key in expired_keys:
            del self.active_states[key]

        return events
