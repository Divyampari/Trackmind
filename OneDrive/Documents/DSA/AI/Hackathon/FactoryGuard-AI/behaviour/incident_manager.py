"""
FactoryGuard AI - Incident Manager & Incident Data Structure

Phase 3: Evidence & Incident Intelligence Layer

Manages persistent, source-agnostic safety incidents (video or webcam),
enforces alert deduplication and continuous event duration updates,
captures visual evidence paths, and provides rich query APIs.
"""

import json
import os
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union

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
        incident_id: Unique incident tracking ID (e.g. 'INC-001' or 'INC-0001').
        worker_id: Persistent worker ID from Phase 1 ByteTrack.
        event: Event identifier (e.g. 'restricted_zone_entry', 'restricted_zone_dwell').
        timestamp: Time position in video or session timestamp (seconds).
        frame: Frame number (0-based) or None if unavailable.
        zone: Zone name where violation occurred.
        severity: Incident severity level ('WARNING', 'HIGH', 'CRITICAL' or lowercase).
        description: Human-explainable description of safety violation.
        dwell_time: Duration spent in zone / event duration (seconds).
        ppe_status: PPE status dictionary.
        zone_type: Zone classification ('restricted' or 'warning').
        status: Incident lifecycle status ('OPEN', 'CLOSED', etc.).
        evidence_frame_path: Filepath to annotated evidence snapshot image.
        worker_bbox: Bounding box [x1, y1, x2, y2].
        worker_center: Center point [cx, cy].
        source: Input source classification ('video' or 'webcam').
        video_path: Source video filepath if applicable, or None for webcam.
    """
    incident_id: str
    worker_id: int
    event: str
    timestamp: float
    frame: Optional[int]
    zone: str
    severity: str
    description: str = ""
    dwell_time: float = 0.0
    ppe_status: Dict[str, Any] = field(default_factory=dict)
    zone_type: str = ""
    status: str = "OPEN"
    evidence_frame_path: Optional[str] = None
    worker_bbox: List[float] = field(default_factory=list)
    worker_center: List[float] = field(default_factory=list)
    source: str = "video"
    video_path: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.dwell_time

    @duration.setter
    def duration(self, val: float) -> None:
        self.dwell_time = val

    @property
    def evidence(self) -> str:
        return self.evidence_frame_path or ""

    @property
    def event_type(self) -> str:
        return self.event

    def to_dict(self) -> Dict[str, Any]:
        """Convert incident to plain dictionary for JSON reporting matching Phase 3 schema."""
        ev_path = self.evidence_frame_path or None
        if ev_path:
            ev_path = ev_path.replace("\\", "/")

        # Parse PPE info cleanly
        ppe_clean = None
        if isinstance(self.ppe_status, dict) and self.ppe_status:
            ppe_clean = {}
            for k, v in self.ppe_status.items():
                if v is True or v == "present":
                    ppe_clean[k] = True
                elif v is False or v == "missing":
                    ppe_clean[k] = False
                elif v is None or v == "unknown":
                    ppe_clean[k] = None
                else:
                    ppe_clean[k] = v

        frame_val = int(self.frame) if self.frame is not None else None
        sev_str = str(self.severity)

        return {
            "incident_id": self.incident_id,
            "worker_id": int(self.worker_id),
            "event_type": self.event,
            "event": self.event,
            "timestamp": round(float(self.timestamp), 3),
            "frame_number": frame_val,
            "frame": frame_val,
            "zone": self.zone,
            "zone_type": self.zone_type,
            "severity": sev_str.lower(),
            "severity_upper": sev_str.upper(),
            "duration": round(float(self.dwell_time), 2),
            "dwell_time": round(float(self.dwell_time), 2),
            "source": self.source,
            "video_path": self.video_path,
            "evidence_frame": ev_path,
            "evidence": ev_path or "",
            "status": self.status,
            "description": self.description,
            "ppe": ppe_clean,
            "ppe_status": self.ppe_status,
            "worker_bbox": [round(float(c), 1) for c in self.worker_bbox],
            "worker_center": [round(float(c), 1) for c in self.worker_center],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SafetyIncident":
        """Reconstruct SafetyIncident instance from dictionary."""
        ev = d.get("evidence_frame") or d.get("evidence") or None
        frame_val = d.get("frame_number") if d.get("frame_number") is not None else d.get("frame")
        if frame_val is not None:
            try:
                frame_val = int(frame_val)
            except (ValueError, TypeError):
                frame_val = None

        return cls(
            incident_id=str(d.get("incident_id", "INC-0001")),
            worker_id=int(d.get("worker_id", 0)),
            event=str(d.get("event_type") or d.get("event") or "unknown_event"),
            timestamp=float(d.get("timestamp", 0.0)),
            frame=frame_val,
            zone=str(d.get("zone", "")),
            severity=str(d.get("severity_upper") or d.get("severity", "WARNING")).upper(),
            description=str(d.get("description", "")),
            dwell_time=float(d.get("duration") or d.get("dwell_time") or 0.0),
            ppe_status=d.get("ppe_status") or d.get("ppe") or {},
            zone_type=str(d.get("zone_type", "")),
            status=str(d.get("status", "OPEN")),
            evidence_frame_path=ev,
            worker_bbox=d.get("worker_bbox", []),
            worker_center=d.get("worker_center", []),
            source=str(d.get("source", "video")),
            video_path=d.get("video_path"),
        )


class IncidentManager:
    """Source-agnostic Incident Manager for Phase 3.

    Registers incidents, manages alert deduplication, updates continuous event durations,
    persists history to JSON, and provides rich query APIs.

    Args:
        cooldown_seconds: Default window in seconds to suppress duplicate alert creation.
        storage_dir: Base directory for storing incident records JSON.
    """

    def __init__(self, cooldown_seconds: float = 5.0, storage_dir: str = os.path.join("data", "incidents")):
        self.incidents: List[SafetyIncident] = []
        self.cooldown_seconds = cooldown_seconds
        self.storage_dir = storage_dir
        self._counter = 1
        # Key: (worker_id, event, zone) -> last_emitted_timestamp
        self._last_alert_time: Dict[Tuple[int, str, str], float] = {}
        # Key: (worker_id, zone) -> active SafetyIncident
        self._active_incidents: Dict[Tuple[int, str], SafetyIncident] = {}

        self.source_info: Dict[str, Any] = {
            "source_name": "",
            "source_path": "",
            "resolution": [0, 0],
            "fps": 0.0,
            "total_frames": 0,
            "processed_at": "",
        }

        # Auto-create storage directory
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(os.path.join("data", "evidence"), exist_ok=True)

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
        frame: Optional[int],
        zone: str,
        severity: str,
        description: str = "",
        dwell_time: float = 0.0,
        ppe_status: Optional[Dict[str, Any]] = None,
        worker_bbox: Optional[List[float]] = None,
        worker_center: Optional[List[float]] = None,
        zone_type: str = "",
        status: str = "OPEN",
        evidence_frame_path: Optional[str] = None,
        custom_incident_id: Optional[str] = None,
        source: str = "video",
        video_path: Optional[str] = None,
    ) -> Optional[SafetyIncident]:
        """Create and register a new safety incident if not suppressed by deduplication.

        Returns:
            SafetyIncident instance if created, or None if suppressed as duplicate.
        """
        # If continuous duplicate event, update existing active incident duration
        if self.is_duplicate(worker_id, event, zone, timestamp):
            self.update_active_incident_duration(worker_id, zone, dwell_time)
            logger.debug(
                "Suppressed duplicate incident creation for Worker %d (%s in '%s') at t=%.2fs",
                worker_id, event, zone, timestamp
            )
            return None

        if custom_incident_id:
            incident_id = custom_incident_id
            self._update_counter_from_id(custom_incident_id)
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
            ppe_status=ppe_status or {},
            zone_type=zone_type,
            status=status,
            evidence_frame_path=evidence_frame_path,
            worker_bbox=worker_bbox or [],
            worker_center=worker_center or [],
            source=source,
            video_path=video_path,
        )

        self.incidents.append(incident)
        self._last_alert_time[(worker_id, event, zone)] = timestamp
        self._active_incidents[(worker_id, zone)] = incident

        logger.warning(
            "[%s] Incident %s: Worker %d - %s in zone '%s' (Source: %s)",
            severity, incident_id, worker_id, event, zone, source
        )

        # Auto-persist incident record
        self.save_incidents()
        return incident

    def update_active_incident_duration(self, worker_id: int, zone: str, dwell_time: float) -> Optional[SafetyIncident]:
        """Update duration of ongoing continuous incident for worker in zone."""
        key = (worker_id, zone)
        if key in self._active_incidents:
            inc = self._active_incidents[key]
            if dwell_time > inc.dwell_time:
                inc.dwell_time = dwell_time
            return inc
        return None

    def update_incident(
        self,
        incident_id: str,
        duration: Optional[float] = None,
        status: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[SafetyIncident]:
        """Update properties of an existing incident by ID."""
        for inc in self.incidents:
            if inc.incident_id.lower() == incident_id.lower():
                if duration is not None:
                    inc.dwell_time = duration
                if status is not None:
                    inc.status = status
                    if status.upper() == "CLOSED":
                        # remove from active incidents map if present
                        key = (inc.worker_id, inc.zone)
                        self._active_incidents.pop(key, None)
                if description is not None:
                    inc.description = description
                self.save_incidents()
                return inc
        return None

    def close_incident(self, incident_id: str) -> Optional[SafetyIncident]:
        """Close an open incident by ID."""
        return self.update_incident(incident_id, status="CLOSED")

    def _update_counter_from_id(self, inc_id: str) -> None:
        """Update internal ID counter to prevent duplicate IDs."""
        try:
            cleaned = inc_id.upper().replace("INC-", "").lstrip("0")
            num = int(cleaned) if cleaned else 0
            if num >= self._counter:
                self._counter = num + 1
        except ValueError:
            pass

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
        """Filter incidents by severity level (case-insensitive)."""
        sev_upper = severity.upper()
        return [inc for inc in self.incidents if str(inc.severity).upper() == sev_upper]

    def get_incidents_by_worker(self, worker_id: int) -> List[SafetyIncident]:
        """Filter incidents by worker ID."""
        return [inc for inc in self.incidents if inc.worker_id == worker_id]

    def get_incidents_by_zone(self, zone_name: str) -> List[SafetyIncident]:
        """Filter incidents by zone name."""
        z_lower = zone_name.lower()
        return [inc for inc in self.incidents if inc.zone.lower() == z_lower]

    def get_incidents_by_event_type(self, event_type: str) -> List[SafetyIncident]:
        """Filter incidents by event type string (case-insensitive)."""
        et_lower = event_type.lower()
        return [inc for inc in self.incidents if inc.event.lower() == et_lower]

    def get_evidence_path(self, incident_id: str) -> Optional[str]:
        """Retrieve filepath of visual evidence snapshot for an incident."""
        for inc in self.incidents:
            if inc.incident_id.lower() == incident_id.lower():
                return inc.evidence_frame_path
        return None

    def get_incidents(
        self,
        severity: Optional[str] = None,
        worker_id: Optional[int] = None,
        zone: Optional[str] = None,
        status: Optional[str] = None,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[SafetyIncident]:
        """Flexible query API to filter incidents by multiple criteria."""
        results = self.incidents
        if severity:
            s_up = severity.upper()
            results = [inc for inc in results if str(inc.severity).upper() == s_up]
        if worker_id is not None:
            results = [inc for inc in results if inc.worker_id == worker_id]
        if zone:
            z_low = zone.lower()
            results = [inc for inc in results if inc.zone.lower() == z_low]
        if status:
            st_up = status.upper()
            results = [inc for inc in results if inc.status.upper() == st_up]
        if event_type:
            et_low = event_type.lower()
            results = [inc for inc in results if inc.event.lower() == et_low]
        if source:
            src_low = source.lower()
            results = [inc for inc in results if inc.source.lower() == src_low]
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Generate high-level statistical summary of incidents."""
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "WARNING": 0}
        zone_counts = {}
        worker_counts = {}

        for inc in self.incidents:
            sev = str(inc.severity).upper()
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

    def save_incidents(self, filepath: Optional[str] = None) -> str:
        """Save persistent incidents to JSON file.

        Args:
            filepath: Destination file path (default: data/incidents/incidents.json).

        Returns:
            Saved file path.
        """
        dest_path = filepath or os.path.join(self.storage_dir, "incidents.json")
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

        inc_dicts = [inc.to_dict() for inc in self.incidents]
        try:
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(inc_dicts, f, indent=2, ensure_ascii=False)
            logger.info("Saved %d incidents to %s", len(self.incidents), dest_path)
        except Exception as e:
            logger.error("Failed to write incidents file %s: %s", dest_path, e)

        return dest_path

    def save_report(self, filepath: str = os.path.join("data", "incidents", "incidents_report.json")) -> str:
        """Save full incident summary report and list to JSON file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self.save_incidents(os.path.join(self.storage_dir, "incidents.json"))

        summary = self.get_summary()
        summary["incidents"] = [inc.to_dict() for inc in self.incidents]

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info("Saved incident summary report to: %s (%d total incidents)", filepath, len(self.incidents))
        except Exception as e:
            logger.error("Failed to write report file %s: %s", filepath, e)

        return filepath

    def load_incidents(self, filepath: Optional[str] = None) -> List[SafetyIncident]:
        """Load persistent incident history from JSON file.

        Handles missing, corrupted, or legacy report JSON files gracefully.

        Args:
            filepath: Target file path (default: data/incidents/incidents.json).

        Returns:
            List of loaded SafetyIncident objects.
        """
        target = filepath or os.path.join(self.storage_dir, "incidents.json")
        if not os.path.isfile(target):
            # Fallback to report path if incidents.json doesn't exist
            report_alt = os.path.join(self.storage_dir, "incidents_report.json")
            if os.path.isfile(report_alt):
                target = report_alt
            else:
                logger.info("No existing incident file found at %s. Initializing empty history.", target)
                return self.incidents

        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_incidents = []
            if isinstance(data, list):
                raw_incidents = data
            elif isinstance(data, dict):
                raw_incidents = data.get("incidents", [])
                if "source_info" in data:
                    self.source_info = data["source_info"]

            loaded_list: List[SafetyIncident] = []
            for inc_d in raw_incidents:
                if isinstance(inc_d, dict):
                    inc_obj = SafetyIncident.from_dict(inc_d)
                    loaded_list.append(inc_obj)
                    self._update_counter_from_id(inc_obj.incident_id)

            self.incidents = loaded_list
            logger.info("Successfully loaded %d persistent incidents from %s", len(self.incidents), target)
        except Exception as e:
            logger.warning("Failed or corrupted incident JSON file %s: %s. Continuing with existing state.", target, e)

        return self.incidents

