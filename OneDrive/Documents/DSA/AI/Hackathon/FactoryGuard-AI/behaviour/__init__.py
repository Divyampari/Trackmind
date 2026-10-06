"""
FactoryGuard AI - Behaviour and Safety Intelligence Module

Phase 2: Behaviour and Safety Intelligence Layer

This module consumes worker tracking data from Phase 1 (YOLO + ByteTrack)
and evaluates worker safety relative to configurable factory safety zones.

Components:
    - ZoneManager: Interactive creation, loading, saving, and testing of zones.
    - ZoneDetector: State tracking of workers inside zones, entry/exit detection, dwell time tracking.
    - PPEInterface: Extensible interface for worker PPE status (helmet, vest, boots).
    - BehaviourAnalyzer: Safety reasoning engine generating safety violation candidates.
    - IncidentManager: Structured incident recording, deduplication, and JSON reporting.
    - EvidenceManager: Visual evidence capture and frame snapshot annotation.
    - BehaviourEngine: Source-independent pipeline coordinator.
"""

from behaviour.zone_manager import Zone, ZoneManager, ZoneType
from behaviour.zone_detector import ZoneDetector, ZoneEvent, EventType
from behaviour.ppe_interface import PPEInterface, WorkerPPEState
from behaviour.behaviour_analyzer import BehaviourAnalyzer
from behaviour.incident_manager import IncidentManager, SafetyIncident, Severity
from behaviour.evidence import EvidenceManager
from behaviour.engine import BehaviourEngine

__all__ = [
    "Zone",
    "ZoneManager",
    "ZoneType",
    "ZoneDetector",
    "ZoneEvent",
    "EventType",
    "PPEInterface",
    "WorkerPPEState",
    "BehaviourAnalyzer",
    "IncidentManager",
    "SafetyIncident",
    "Severity",
    "EvidenceManager",
    "BehaviourEngine",
]
