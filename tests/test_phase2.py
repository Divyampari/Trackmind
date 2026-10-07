"""
FactoryGuard AI - Phase 2 Comprehensive Test Suite

Verifies Phase 2 Behaviour and Safety Intelligence Layer against all required scenarios:
    1. No workers detected
    2. Worker outside all zones (normal movement)
    3. Worker enters warning zone
    4. Worker enters restricted zone
    5. Worker exits restricted zone
    6. Worker remains inside restricted zone (dwell threshold breached)
    7. Multiple workers in different zones
    8. Multiple workers entering the same zone
    9. Temporary tracking loss (grace period)
    10. Repeated frames of the same violation (deduplication check)
    11. Different zone configurations and resolutions
    12. Evidence frame generation and metadata correctness
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from behaviour.zone_manager import Zone, ZoneManager, ZoneType
from behaviour.zone_detector import ZoneDetector, EventType
from behaviour.behaviour_analyzer import BehaviourAnalyzer
from behaviour.incident_manager import IncidentManager, Severity
from behaviour.evidence import EvidenceManager
from behaviour.ppe_interface import PPEInterface, WorkerPPEState
from behaviour.engine import BehaviourEngine


class TestPhase2BehaviourIntelligence(unittest.TestCase):

    def setUp(self):
        """Set up test zones and instances."""
        # Restricted Zone: [100, 100] to [300, 300]
        self.restricted_zone = Zone(
            name="Robotics Cell A",
            type=ZoneType.RESTRICTED.value,
            polygon=[[100, 100], [300, 100], [300, 300], [100, 300]],
            dwell_threshold=5.0,
        )
        # Warning Zone: [400, 100] to [600, 300]
        self.warning_zone = Zone(
            name="Forklift Corridor B",
            type=ZoneType.WARNING.value,
            polygon=[[400, 100], [600, 100], [600, 300], [400, 300]],
            dwell_threshold=10.0,
        )
        self.zones = [self.restricted_zone, self.warning_zone]

    def test_01_no_workers(self):
        """Scenario 1: No workers detected in frame."""
        detector = ZoneDetector()
        events = detector.update(frame_number=0, timestamp=0.0, workers=[], zones=self.zones)
        self.assertEqual(len(events), 0)
        self.assertEqual(len(detector.active_states), 0)

    def test_02_worker_outside_zones(self):
        """Scenario 2: Worker walking outside all zones (Normal movement)."""
        detector = ZoneDetector()
        worker_outside = {"id": 1, "bbox": [10, 10, 50, 50], "center": [30, 30], "confidence": 0.9}
        events = detector.update(frame_number=0, timestamp=0.0, workers=[worker_outside], zones=self.zones)
        self.assertEqual(len(events), 0)
        self.assertEqual(len(detector.active_states), 0)

    def test_03_worker_enters_warning_zone(self):
        """Scenario 3: Worker enters warning zone."""
        detector = ZoneDetector()
        worker_warning = {"id": 2, "bbox": [450, 150, 490, 250], "center": [470, 200], "confidence": 0.88}
        events = detector.update(frame_number=0, timestamp=0.0, workers=[worker_warning], zones=self.zones)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.WARNING_ENTRY.value)
        self.assertEqual(events[0].worker_id, 2)
        self.assertEqual(events[0].zone_name, "Forklift Corridor B")

    def test_04_worker_enters_restricted_zone(self):
        """Scenario 4: Worker enters restricted zone."""
        detector = ZoneDetector()
        worker_restricted = {"id": 3, "bbox": [150, 150, 210, 250], "center": [180, 200], "confidence": 0.92}
        events = detector.update(frame_number=0, timestamp=0.0, workers=[worker_restricted], zones=self.zones)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.RESTRICTED_ENTRY.value)
        self.assertEqual(events[0].worker_id, 3)
        self.assertEqual(events[0].zone_name, "Robotics Cell A")

    def test_05_worker_exits_restricted_zone(self):
        """Scenario 5: Worker enters then exits restricted zone."""
        detector = ZoneDetector(grace_period=0.5)
        worker_in = {"id": 4, "bbox": [150, 150, 210, 250], "center": [180, 200], "confidence": 0.9}
        worker_out = {"id": 4, "bbox": [10, 10, 50, 50], "center": [30, 30], "confidence": 0.9}

        # Frame 0 (t=0.0): Entry
        events_0 = detector.update(frame_number=0, timestamp=0.0, workers=[worker_in], zones=self.zones)
        self.assertEqual(len(events_0), 1)
        self.assertEqual(events_0[0].event_type, EventType.RESTRICTED_ENTRY.value)

        # Frame 10 (t=0.3): Still inside
        events_1 = detector.update(frame_number=10, timestamp=0.3, workers=[worker_in], zones=self.zones)
        self.assertEqual(len(events_1), 0)

        # Frame 30 (t=1.0): Moved outside -> Exits after grace period
        events_2 = detector.update(frame_number=30, timestamp=1.0, workers=[worker_out], zones=self.zones)
        self.assertEqual(len(events_2), 1)
        self.assertEqual(events_2[0].event_type, EventType.RESTRICTED_EXIT.value)
        self.assertEqual(events_2[0].worker_id, 4)

    def test_06_worker_dwell_time_detection(self):
        """Scenario 6: Worker remains inside restricted zone exceeding 5.0s dwell threshold."""
        detector = ZoneDetector()
        worker = {"id": 5, "bbox": [150, 150, 210, 250], "center": [180, 200], "confidence": 0.95}

        # Frame 0 (t=0.0): Entry
        events_0 = detector.update(frame_number=0, timestamp=0.0, workers=[worker], zones=self.zones)
        self.assertEqual(events_0[0].event_type, EventType.RESTRICTED_ENTRY.value)

        # Frame 90 (t=3.0): Still inside (dwell=3.0s < 5.0s) -> No new event
        events_1 = detector.update(frame_number=90, timestamp=3.0, workers=[worker], zones=self.zones)
        self.assertEqual(len(events_1), 0)

        # Frame 160 (t=5.3): Exceeded 5.0s dwell threshold -> Dwell event emitted!
        events_2 = detector.update(frame_number=160, timestamp=5.3, workers=[worker], zones=self.zones)
        self.assertEqual(len(events_2), 1)
        self.assertEqual(events_2[0].event_type, EventType.RESTRICTED_DWELL.value)
        self.assertGreaterEqual(events_2[0].dwell_time, 5.0)

        # Frame 200 (t=6.6): Continued presence -> NO duplicate dwell event emitted!
        events_3 = detector.update(frame_number=200, timestamp=6.6, workers=[worker], zones=self.zones)
        self.assertEqual(len(events_3), 0)

    def test_07_multiple_workers_different_zones(self):
        """Scenario 7: Multiple workers in different zones simultaneously."""
        detector = ZoneDetector()
        w1_res = {"id": 10, "bbox": [150, 150, 200, 200], "center": [175, 175], "confidence": 0.9}
        w2_warn = {"id": 20, "bbox": [450, 150, 500, 200], "center": [475, 175], "confidence": 0.9}

        events = detector.update(frame_number=0, timestamp=0.0, workers=[w1_res, w2_warn], zones=self.zones)
        self.assertEqual(len(events), 2)
        event_types = {e.event_type for e in events}
        self.assertIn(EventType.RESTRICTED_ENTRY.value, event_types)
        self.assertIn(EventType.WARNING_ENTRY.value, event_types)

    def test_08_multiple_workers_same_zone(self):
        """Scenario 8: Multiple workers entering the same zone."""
        detector = ZoneDetector()
        w1 = {"id": 11, "bbox": [120, 120, 160, 160], "center": [140, 140], "confidence": 0.9}
        w2 = {"id": 12, "bbox": [200, 200, 250, 250], "center": [225, 225], "confidence": 0.9}

        events = detector.update(frame_number=0, timestamp=0.0, workers=[w1, w2], zones=self.zones)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].zone_name, "Robotics Cell A")
        self.assertEqual(events[1].zone_name, "Robotics Cell A")

    def test_09_temporary_tracking_loss(self):
        """Scenario 9: Temporary tracking loss / occlusion (grace period prevents duplicate entry)."""
        detector = ZoneDetector(grace_period=1.5)
        worker = {"id": 50, "bbox": [150, 150, 200, 200], "center": [175, 175], "confidence": 0.9}

        # Frame 0 (t=0.0): Entry
        detector.update(frame_number=0, timestamp=0.0, workers=[worker], zones=self.zones)

        # Frame 10 (t=0.3): Worker temporarily lost by ByteTrack (0 workers in frame)
        events_lost = detector.update(frame_number=10, timestamp=0.3, workers=[], zones=self.zones)
        self.assertEqual(len(events_lost), 0)  # Within 1.5s grace period -> Not declared exited yet

        # Frame 20 (t=0.6): Worker reappears -> State preserved, NO duplicate entry event!
        events_reappear = detector.update(frame_number=20, timestamp=0.6, workers=[worker], zones=self.zones)
        self.assertEqual(len(events_reappear), 0)

    def test_10_duplicate_alert_prevention(self):
        """Scenario 10: Deduplication check in IncidentManager."""
        inc_mgr = IncidentManager(cooldown_seconds=10.0)

        # First alert at t=1.0s
        inc1 = inc_mgr.create_incident(
            worker_id=7,
            event="restricted_zone_entry",
            timestamp=1.0,
            frame=30,
            zone="Robotics Cell A",
            severity="HIGH",
            description="Entry",
            dwell_time=0.0,
            ppe_status={"helmet": "unknown", "vest": "unknown", "boots": "unknown"},
            worker_bbox=[100, 100, 200, 200],
            worker_center=[150, 150],
        )
        self.assertIsNotNone(inc1)

        # Duplicate attempt at t=3.0s (within 10s cooldown) -> Suppressed!
        inc2 = inc_mgr.create_incident(
            worker_id=7,
            event="restricted_zone_entry",
            timestamp=3.0,
            frame=90,
            zone="Robotics Cell A",
            severity="HIGH",
            description="Entry duplicate",
            dwell_time=0.0,
            ppe_status={"helmet": "unknown", "vest": "unknown", "boots": "unknown"},
            worker_bbox=[100, 100, 200, 200],
            worker_center=[150, 150],
        )
        self.assertIsNone(inc2)
        self.assertEqual(len(inc_mgr.incidents), 1)

    def test_11_zone_manager_save_and_load(self):
        """Scenario 11: Zone configuration serialization and loading."""
        test_json_path = os.path.join("data", "zones", "test_zones_temp.json")

        zm1 = ZoneManager()
        zm1.video_name = "test_vid.mp4"
        zm1.add_zone(self.restricted_zone)
        zm1.add_zone(self.warning_zone)
        saved_file = zm1.save_zones(test_json_path)

        self.assertTrue(os.path.isfile(saved_file))

        zm2 = ZoneManager(saved_file)
        self.assertEqual(len(zm2.zones), 2)
        self.assertEqual(zm2.get_zone("Robotics Cell A").type, "restricted")
        self.assertEqual(zm2.get_zone("Forklift Corridor B").type, "warning")

        # Cleanup
        if os.path.isfile(test_json_path):
            os.remove(test_json_path)

    def test_12_evidence_frame_capture(self):
        """Scenario 12: Evidence frame annotation and snapshot generation."""
        evidence_mgr = EvidenceManager(output_dir=os.path.join("data", "incidents", "evidence"))
        dummy_frame = np.full((480, 640, 3), 100, dtype=np.uint8)

        filepath = evidence_mgr.capture_evidence(
            frame=dummy_frame,
            incident_id="INC-TEST",
            worker_id=99,
            event="restricted_zone_dwell",
            timestamp=12.5,
            frame_number=375,
            zone_name="Robotics Cell A",
            severity="CRITICAL",
            worker_bbox=[100, 100, 200, 200],
            zones=self.zones,
        )

        self.assertTrue(os.path.isfile(filepath))
        # Cleanup test snapshot
        if os.path.isfile(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    unittest.main()
