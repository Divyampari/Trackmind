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
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("factoryguard.behaviour.incident_manager")


class Severity(str, Enum):
    """Incident severity classification."""
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class SafetyIncident:
    """Represents a structured safety incident produced by Phase 2/3.

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
        zone_type: Zone classification ('restricted' or 'warning').
        status: Incident lifecycle status ('OPEN', 'CLOSED', etc.).
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
    zone_type: str = ""
    status: str = "OPEN"
    evidence_frame_path: Optional[str] = None
    worker_bbox: List[float] = field(default_factory=list)
    worker_center: List[float] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.dwell_time

    @property
    def evidence(self) -> str:
        return self.evidence_frame_path or ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert incident to plain dictionary for JSON reporting matching Phase 3 schema."""
        ev_path = self.evidence_frame_path or ""
        if ev_path:
            ev_path = ev_path.replace("\\", "/")

        return {
            "incident_id": self.incident_id,
            "worker_id": int(self.worker_id),
            "event": self.event,
            "zone": self.zone,
            "zone_type": self.zone_type,
            "severity": self.severity,
            "timestamp": round(float(self.timestamp), 3),
            "frame": int(self.frame),
            "duration": round(float(self.dwell_time), 2),
            "dwell_time": round(float(self.dwell_time), 2),
            "evidence": ev_path,
            "evidence_frame": ev_path,
            "status": self.status,
            "description": self.description,
            "ppe_status": self.ppe_status,
            "worker_bbox": [round(float(c), 1) for c in self.worker_bbox],
            "worker_center": [round(float(c), 1) for c in self.worker_center],
        }


class IncidentManager:
    """Registers incidents, manages deduplication cooldowns, provides query APIs, and exports summary reports.

    Args:
        cooldown_seconds: Default window in seconds to suppress duplicate alerts for the same worker & event.
    """

    def __init__(self, cooldown_seconds: float = 5.0):
        self.incidents: List[SafetyIncident] = []
        self.cooldown_seconds = cooldown_seconds
        self._counter = 1
        # Key: (worker_id, event, zone) -> last_emitted_timestamp
        self._last_alert_time: Dict[Tuple[int, str, str], float] = {}
        self.source_info: Dict[str, Any] = {
            "source_name": "",
            "source_path": "",
            "resolution": [0, 0],
            "fps": 0.0,
            "total_frames": 0,
            "processed_at": "",
        }

    def set_source_info(
        self,
        source_name: str = "",
        source_path: str = "",
        resolution: Optional[List[int]] = None,
        fps: float = 0.0,
        total_frames: int = 0,
    ) -> None:
        """Set processing source/session metadata."""
        import datetime
        self.source_info = {
            "video_name": source_name,
            "video_path": source_path,
            "resolution": resolution or [0, 0],
            "fps": float(fps),
            "processed_frames": int(total_frames),
            "processed_at": datetime.datetime.now().isoformat(),
        }

    def peek_next_id(self) -> str:
        """Return the next incident ID without incrementing counter."""
        return f"INC-{self._counter:04d}"

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
        zone_type: str = "",
        status: str = "OPEN",
        evidence_frame_path: Optional[str] = None,
        custom_incident_id: Optional[str] = None,
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

        if custom_incident_id:
            incident_id = custom_incident_id
            try:
                num = int(custom_incident_id.replace("INC-", ""))
                if num >= self._counter:
                    self._counter = num + 1
            except ValueError:
                pass
        else:
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
            zone_type=zone_type,
            status=status,
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

    def get_all_incidents(self) -> List[SafetyIncident]:
        """Return all recorded incidents in current session."""
        return list(self.incidents)

    def get_active_incidents(self, active_worker_zones: Optional[List[Tuple[int, str]]] = None) -> List[SafetyIncident]:
        """Return active incidents (matching open worker-zone states or OPEN status)."""
        if active_worker_zones is not None:
            active_set = set(active_worker_zones)
            return [inc for inc in self.incidents if (inc.worker_id, inc.zone) in active_set and inc.status == "OPEN"]
        return [inc for inc in self.incidents if inc.status == "OPEN"]

    def get_incidents_by_severity(self, severity: str) -> List[SafetyIncident]:
        """Filter incidents by severity level (e.g. 'CRITICAL', 'HIGH', 'WARNING')."""
        sev_upper = severity.upper()
        return [inc for inc in self.incidents if inc.severity.upper() == sev_upper]

    def get_incidents_by_worker(self, worker_id: int) -> List[SafetyIncident]:
        """Filter incidents by worker ID."""
        return [inc for inc in self.incidents if inc.worker_id == worker_id]

    def get_incidents_by_zone(self, zone_name: str) -> List[SafetyIncident]:
        """Filter incidents by zone name."""
        z_lower = zone_name.lower()
        return [inc for inc in self.incidents if inc.zone.lower() == z_lower]

    def get_incidents(
        self,
        severity: Optional[str] = None,
        worker_id: Optional[int] = None,
        zone: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[SafetyIncident]:
        """Flexible query API to filter incidents by multiple criteria."""
        results = self.incidents
        if severity:
            s_up = severity.upper()
            results = [inc for inc in results if inc.severity.upper() == s_up]
        if worker_id is not None:
            results = [inc for inc in results if inc.worker_id == worker_id]
        if zone:
            z_low = zone.lower()
            results = [inc for inc in results if inc.zone.lower() == z_low]
        if status:
            st_up = status.upper()
            results = [inc for inc in results if inc.status.upper() == st_up]
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Generate high-level statistical summary of incidents."""
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "WARNING": 0}
        zone_counts = {}
        worker_counts = {}

        for inc in self.incidents:
            sev = inc.severity.upper()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            z_key = inc.zone
            zone_counts[z_key] = zone_counts.get(z_key, 0) + 1
            w_str = f"Worker_{inc.worker_id}"
            worker_counts[w_str] = worker_counts.get(w_str, 0) + 1

        return {
            "source_info": self.source_info,
            "total_incidents": len(self.incidents),
            "severity_summary": severity_counts,
            "severity_breakdown": severity_counts,
            "zone_summary": zone_counts,
            "zone_breakdown": zone_counts,
            "worker_summary": worker_counts,
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

        severity_counts = {"CRITICAL": 0, "HIGH": 0, "WARNING": 0}
        zone_counts = {}
        worker_counts = {}

        for inc in self.incidents:
            sev = inc.severity.upper()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            z_key = inc.zone
            zone_counts[z_key] = zone_counts.get(z_key, 0) + 1
            w_str = f"Worker_{inc.worker_id}"
            worker_counts[w_str] = worker_counts.get(w_str, 0) + 1

        summary = {
            "source_info": self.source_info,
            "total_incidents": len(self.incidents),
            "severity_summary": severity_counts,
            "severity_breakdown": severity_counts,
            "zone_summary": zone_counts,
            "zone_breakdown": zone_counts,
            "worker_summary": worker_counts,
            "worker_breakdown": worker_counts,
            "incidents": [inc.to_dict() for inc in self.incidents],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("Incident summary report saved to: %s (%d total incidents)", filepath, len(self.incidents))
        return filepath
