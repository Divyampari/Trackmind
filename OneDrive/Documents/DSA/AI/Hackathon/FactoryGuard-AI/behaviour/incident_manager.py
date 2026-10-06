"""
FactoryGuard AI - Incident Manager & Incident Data Structure

Phase 2: Behaviour and Safety Intelligence Layer

Manages structured safety incidents, enforces alert deduplication and cooldowns,
and generates JSON incident reports for security audit logs.
"""

import json
import os
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

logger = logging.getLogger("factoryguard.behaviour.incident_manager")


class Severity(str, Enum):
    """Incident severity classification."""
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class SafetyIncident:
    """Represents a structured safety incident produced by Phase 2.

    Attributes:
        incident_id: Unique incident tracking ID (e.g. 'INC-0001').
        worker_id: Persistent worker ID from Phase 1 ByteTrack.
        event: Event identifier (e.g. 'restricted_zone_entry', 'restricted_zone_dwell').
        timestamp: Time position in video (seconds).
        frame: Frame number (0-based).
        zone: Zone name where violation occurred.
        severity: Incident severity level ('WARNING', 'HIGH', 'CRITICAL').
        description: Human-explainable description of safety violation.
        dwell_time: Duration spent in zone prior to incident (seconds).
        ppe_status: PPE status dictionary.
        evidence_frame_path: Filepath to annotated evidence snapshot image.
        worker_bbox: Bounding box [x1, y1, x2, y2].
        worker_center: Center point [cx, cy].
    """
    incident_id: str
    worker_id: int
    event: str
    timestamp: float
    frame: int
    zone: str
    severity: str
    description: str
    dwell_time: float
    ppe_status: Dict[str, str]
    evidence_frame_path: Optional[str] = None
    worker_bbox: List[float] = field(default_factory=list)
    worker_center: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert incident to plain dictionary for JSON reporting."""
        return {
            "incident_id": self.incident_id,
            "worker_id": self.worker_id,
            "event": self.event,
            "timestamp": round(float(self.timestamp), 3),
            "frame": int(self.frame),
            "zone": self.zone,
            "severity": self.severity,
            "description": self.description,
            "dwell_time": round(float(self.dwell_time), 2),
            "ppe_status": self.ppe_status,
            "evidence_frame": self.evidence_frame_path or "",
            "worker_bbox": [round(float(c), 1) for c in self.worker_bbox],
            "worker_center": [round(float(c), 1) for c in self.worker_center],
        }


class IncidentManager:
    """Registers incidents, manages deduplication cooldowns, and exports summary reports.

    Args:
        cooldown_seconds: Default window in seconds to suppress duplicate alerts for the same worker & event.
    """

    def __init__(self, cooldown_seconds: float = 5.0):
        self.incidents: List[SafetyIncident] = []
        self.cooldown_seconds = cooldown_seconds
        self._counter = 1
        # Key: (worker_id, event, zone) -> last_emitted_timestamp
        self._last_alert_time: Dict[Tuple[int, str, str], float] = {}

    def is_duplicate(self, worker_id: int, event: str, zone: str, timestamp: float) -> bool:
        """Check if an identical event was emitted for worker in zone within cooldown window."""
        key = (worker_id, event, zone)
        if key in self._last_alert_time:
            last_t = self._last_alert_time[key]
            if (timestamp - last_t) < self.cooldown_seconds:
                return True
        return False

    def create_incident(
        self,
        worker_id: int,
        event: str,
        timestamp: float,
        frame: int,
        zone: str,
        severity: str,
        description: str,
        dwell_time: float,
        ppe_status: Dict[str, str],
        worker_bbox: List[float],
        worker_center: List[float],
        evidence_frame_path: Optional[str] = None,
    ) -> Optional[SafetyIncident]:
        """Create and register a new safety incident if not suppressed by deduplication.

        Returns:
            SafetyIncident instance if created, or None if suppressed as duplicate.
        """
        if self.is_duplicate(worker_id, event, zone, timestamp):
            logger.debug(
                "Suppressed duplicate incident for Worker %d (%s in '%s') at t=%.2fs",
                worker_id, event, zone, timestamp
            )
            return None

        incident_id = f"INC-{self._counter:04d}"
        self._counter += 1

        incident = SafetyIncident(
            incident_id=incident_id,
            worker_id=worker_id,
            event=event,
            timestamp=timestamp,
            frame=frame,
            zone=zone,
            severity=severity,
            description=description,
            dwell_time=dwell_time,
            ppe_status=ppe_status,
            evidence_frame_path=evidence_frame_path,
            worker_bbox=worker_bbox,
            worker_center=worker_center,
        )

        self.incidents.append(incident)
        self._last_alert_time[(worker_id, event, zone)] = timestamp

        logger.warning(
            "[%s] Incident %s: Worker %d - %s in zone '%s' (Severity: %s)",
            severity, incident_id, worker_id, event, zone, severity
        )
        return incident

    def get_summary(self) -> Dict[str, Any]:
        """Generate high-level statistical summary of incidents."""
        severity_counts = {"WARNING": 0, "HIGH": 0, "CRITICAL": 0}
        event_counts = {}
        worker_counts = {}
        for inc in self.incidents:
            severity_counts[inc.severity] = severity_counts.get(inc.severity, 0) + 1
            event_counts[inc.event] = event_counts.get(inc.event, 0) + 1
            w_str = f"Worker_{inc.worker_id}"
            worker_counts[w_str] = worker_counts.get(w_str, 0) + 1

        return {
            "total_incidents": len(self.incidents),
            "severity_breakdown": severity_counts,
            "event_breakdown": event_counts,
            "worker_breakdown": worker_counts,
        }

    def save_report(self, filepath: str = os.path.join("data", "incidents", "incidents_report.json")) -> str:
        """Save all incidents and summary report to JSON file.

        Args:
            filepath: Destination file path.

        Returns:
            Path to saved JSON file.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        severity_counts = {"WARNING": 0, "HIGH": 0, "CRITICAL": 0}
        event_counts = {}
        worker_counts = {}

        for inc in self.incidents:
            severity_counts[inc.severity] = severity_counts.get(inc.severity, 0) + 1
            event_counts[inc.event] = event_counts.get(inc.event, 0) + 1
            w_str = f"Worker_{inc.worker_id}"
            worker_counts[w_str] = worker_counts.get(w_str, 0) + 1

        summary = {
            "total_incidents": len(self.incidents),
            "severity_breakdown": severity_counts,
            "event_breakdown": event_counts,
            "worker_breakdown": worker_counts,
            "incidents": [inc.to_dict() for inc in self.incidents],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("Incident summary report saved to: %s (%d total incidents)", filepath, len(self.incidents))
        return filepath
