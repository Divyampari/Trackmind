"""
FactoryGuard AI - Phase 3 Incident & Reporting Intelligence Test Suite

Verifies Phase 3 requirements:
    1. No violation -> no incident
    2. Warning entry -> one warning incident (WARNING severity)
    3. Restricted entry -> one high-severity incident (HIGH severity)
    4. Restricted dwell -> one critical incident (CRITICAL severity)
    5. Repeated frames -> no incident spam (deduplication)
    6. Worker exits -> state closes correctly
    7. Worker re-enters -> new valid incident created
    8. Multiple workers -> incidents remain associated with correct worker IDs
    9. Evidence image is created
    10. Evidence contains correct worker/zone/incident metadata
    11. JSON report is valid and match schema
    12. Incident IDs are unique (INC-0001, INC-0002, etc.)
    13. Existing Phase 2 tests still pass
"""

import json
import os
import sys
import unittest
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from behaviour.zone_manager import Zone, ZoneManager, ZoneType
from behaviour.zone_detector import ZoneDetector, EventType
from behaviour.behaviour_analyzer import BehaviourAnalyzer
from behaviour.incident_manager import IncidentManager, SafetyIncident, Severity
from behaviour.evidence import EvidenceManager
from behaviour.engine import BehaviourEngine


class TestPhase3IncidentIntelligence(unittest.TestCase):

    def setUp(self):
        """Initialize test zones and engine components."""
        self.restricted_zone = Zone(
            name="Robotics Cell A",
            type=ZoneType.RESTRICTED.value,
            polygon=[[100, 100], [300, 100], [300, 300], [100, 300]],
            dwell_threshold=5.0,
        )
        self.warning_zone = Zone(
            name="Forklift Corridor B",
            type=ZoneType.WARNING.value,
            polygon=[[400, 100], [600, 100], [600, 300], [400, 300]],
            dwell_threshold=10.0,
        )
        self.zones = [self.restricted_zone, self.warning_zone]
        self.test_evidence_dir = os.path.join("data", "incidents", "evidence_test")
        self.test_report_path = os.path.join("data", "incidents", "test_incidents_report.json")

    def tearDown(self):
        """Clean up temporary test artifacts."""
        if os.path.isfile(self.test_report_path):
            try:
                os.remove(self.test_report_path)
            except OSError:
                pass

    def test_01_no_violation_no_incident(self):
        """1. No violation -> no incident."""
        inc_mgr = IncidentManager()
        analyzer = BehaviourAnalyzer(incident_manager=inc_mgr)
        detector = ZoneDetector()

        worker_outside = {"id": 1, "bbox": [10, 10, 50, 50], "center": [30, 30]}
        events = detector.update(frame_number=0, timestamp=0.0, workers=[worker_outside], zones=self.zones)
        incidents = analyzer.process_frame_events(events=events, frame=None, zones=self.zones)

        self.assertEqual(len(incidents), 0)
        self.assertEqual(len(inc_mgr.incidents), 0)

    def test_02_warning_entry_one_warning_incident(self):
        """2. Warning entry -> one warning incident."""
        inc_mgr = IncidentManager()
        analyzer = BehaviourAnalyzer(incident_manager=inc_mgr)
        detector = ZoneDetector()

        worker_warning = {"id": 2, "bbox": [450, 150, 490, 250], "center": [470, 200]}
        events = detector.update(frame_number=0, timestamp=0.0, workers=[worker_warning], zones=self.zones)
        incidents = analyzer.process_frame_events(events=events, frame=None, zones=self.zones)

        self.assertEqual(len(incidents), 1)
        inc = incidents[0]
        self.assertEqual(inc.event, EventType.WARNING_ENTRY.value)
        self.assertEqual(inc.severity, Severity.WARNING.value)
        self.assertEqual(inc.worker_id, 2)
        self.assertEqual(inc.zone, "Forklift Corridor B")

    def test_03_restricted_entry_one_high_incident(self):
        """3. Restricted entry -> one high-severity incident."""
        inc_mgr = IncidentManager()
        analyzer = BehaviourAnalyzer(incident_manager=inc_mgr)
        detector = ZoneDetector()

        worker_res = {"id": 3, "bbox": [150, 150, 210, 250], "center": [180, 200]}
        events = detector.update(frame_number=0, timestamp=0.0, workers=[worker_res], zones=self.zones)
        incidents = analyzer.process_frame_events(events=events, frame=None, zones=self.zones)

        self.assertEqual(len(incidents), 1)
        inc = incidents[0]
        self.assertEqual(inc.event, EventType.RESTRICTED_ENTRY.value)
        self.assertEqual(inc.severity, Severity.HIGH.value)
        self.assertEqual(inc.worker_id, 3)

    def test_04_restricted_dwell_one_critical_incident(self):
        """4. Restricted dwell -> one critical incident."""
        inc_mgr = IncidentManager()
        analyzer = BehaviourAnalyzer(incident_manager=inc_mgr)
        detector = ZoneDetector()

        worker = {"id": 4, "bbox": [150, 150, 210, 250], "center": [180, 200]}

        # Frame 0 (t=0.0): Entry (HIGH)
        e0 = detector.update(frame_number=0, timestamp=0.0, workers=[worker], zones=self.zones)
        inc0 = analyzer.process_frame_events(events=e0, frame=None, zones=self.zones)
        self.assertEqual(len(inc0), 1)

        # Frame 160 (t=5.5s > 5.0s dwell threshold): Dwell breach (CRITICAL)
        e1 = detector.update(frame_number=160, timestamp=5.5, workers=[worker], zones=self.zones)
        inc1 = analyzer.process_frame_events(events=e1, frame=None, zones=self.zones)
        self.assertEqual(len(inc1), 1)
        self.assertEqual(inc1[0].event, EventType.RESTRICTED_DWELL.value)
        self.assertEqual(inc1[0].severity, Severity.CRITICAL.value)

    def test_05_repeated_frames_no_incident_spam(self):
        """5. Repeated frames -> no incident spam."""
        inc_mgr = IncidentManager(cooldown_seconds=10.0)
        analyzer = BehaviourAnalyzer(incident_manager=inc_mgr)
        detector = ZoneDetector()

        worker = {"id": 5, "bbox": [150, 150, 210, 250], "center": [180, 200]}

        # Process frame 0 twice
        e0 = detector.update(frame_number=0, timestamp=0.0, workers=[worker], zones=self.zones)
        inc0 = analyzer.process_frame_events(events=e0, frame=None, zones=self.zones)

        e0_repeat = detector.update(frame_number=0, timestamp=0.0, workers=[worker], zones=self.zones)
        inc0_repeat = analyzer.process_frame_events(events=e0_repeat, frame=None, zones=self.zones)

        self.assertEqual(len(inc0), 1)
        self.assertEqual(len(inc0_repeat), 0)
        self.assertEqual(len(inc_mgr.incidents), 1)

    def test_06_worker_exits_state_closes(self):
        """6. Worker exits -> state closes correctly."""
        detector = ZoneDetector(grace_period=0.2)
        worker_in = {"id": 6, "bbox": [150, 150, 210, 250], "center": [180, 200]}
        worker_out = {"id": 6, "bbox": [10, 10, 50, 50], "center": [30, 30]}

        # Enter
        detector.update(frame_number=0, timestamp=0.0, workers=[worker_in], zones=self.zones)
        self.assertIn((6, "Robotics Cell A"), detector.active_states)

        # Exit after grace period
        detector.update(frame_number=10, timestamp=0.5, workers=[worker_out], zones=self.zones)
        self.assertNotIn((6, "Robotics Cell A"), detector.active_states)

    def test_07_worker_reenters_new_incident(self):
        """7. Worker re-enters -> new valid incident can be created."""
        inc_mgr = IncidentManager(cooldown_seconds=0.1)
        analyzer = BehaviourAnalyzer(incident_manager=inc_mgr)
        detector = ZoneDetector(grace_period=0.1)

        w_in = {"id": 7, "bbox": [150, 150, 210, 250], "center": [180, 200]}
        w_out = {"id": 7, "bbox": [10, 10, 50, 50], "center": [30, 30]}

        # Entry 1 at t=0.0
        e1 = detector.update(frame_number=0, timestamp=0.0, workers=[w_in], zones=self.zones)
        inc1 = analyzer.process_frame_events(events=e1, frame=None, zones=self.zones)
        self.assertEqual(len(inc1), 1)

        # Exit at t=1.0
        detector.update(frame_number=30, timestamp=1.0, workers=[w_out], zones=self.zones)

        # Entry 2 at t=2.0
        e2 = detector.update(frame_number=60, timestamp=2.0, workers=[w_in], zones=self.zones)
        inc2 = analyzer.process_frame_events(events=e2, frame=None, zones=self.zones)
        self.assertEqual(len(inc2), 1)
        self.assertEqual(len(inc_mgr.incidents), 2)

    def test_08_multiple_workers_correct_ids(self):
        """8. Multiple workers -> incidents remain associated with correct IDs."""
        inc_mgr = IncidentManager()
        analyzer = BehaviourAnalyzer(incident_manager=inc_mgr)
        detector = ZoneDetector()

        w8 = {"id": 8, "bbox": [120, 120, 160, 160], "center": [140, 140]}
        w9 = {"id": 9, "bbox": [200, 200, 250, 250], "center": [225, 225]}

        e = detector.update(frame_number=0, timestamp=0.0, workers=[w8, w9], zones=self.zones)
        incidents = analyzer.process_frame_events(events=e, frame=None, zones=self.zones)

        self.assertEqual(len(incidents), 2)
        worker_ids = [inc.worker_id for inc in incidents]
        self.assertIn(8, worker_ids)
        self.assertIn(9, worker_ids)

    def test_09_10_evidence_image_and_metadata(self):
        """9. Evidence image is created. 10. Evidence contains correct metadata."""
        inc_mgr = IncidentManager()
        ev_mgr = EvidenceManager(output_dir=self.test_evidence_dir)
        analyzer = BehaviourAnalyzer(incident_manager=inc_mgr, evidence_manager=ev_mgr)
        detector = ZoneDetector()

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        worker = {"id": 10, "bbox": [150, 150, 210, 250], "center": [180, 200]}

        e = detector.update(frame_number=0, timestamp=0.0, workers=[worker], zones=self.zones)
        incidents = analyzer.process_frame_events(events=e, frame=dummy_frame, zones=self.zones)

        self.assertEqual(len(incidents), 1)
        inc = incidents[0]
        self.assertTrue(inc.evidence_frame_path)
        self.assertTrue(os.path.isfile(inc.evidence_frame_path))
        self.assertIn("INC-0001", inc.evidence_frame_path)

        # Cleanup evidence snapshot
        if os.path.isfile(inc.evidence_frame_path):
            os.remove(inc.evidence_frame_path)

    def test_11_json_report_validity(self):
        """11. JSON report is valid and matches schema."""
        inc_mgr = IncidentManager()
        inc_mgr.set_source_info(source_name="test.mp4", resolution=[1920, 1080], fps=30.0, total_frames=100)

        inc_mgr.create_incident(
            worker_id=11,
            event="restricted_zone_entry",
            timestamp=5.0,
            frame=150,
            zone="Robotics Cell A",
            severity="HIGH",
            description="Test violation",
            dwell_time=0.0,
            ppe_status={"helmet": "present", "vest": "present"},
            worker_bbox=[100, 100, 200, 200],
            worker_center=[150, 150],
            zone_type="restricted",
            status="OPEN",
            evidence_frame_path="data/incidents/evidence/INC-0001.jpg",
        )

        report_file = inc_mgr.save_report(self.test_report_path)
        self.assertTrue(os.path.isfile(report_file))

        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["total_incidents"], 1)
        self.assertIn("source_info", data)
        self.assertIn("severity_summary", data)
        self.assertEqual(data["severity_summary"]["HIGH"], 1)
        self.assertEqual(len(data["incidents"]), 1)

        inc_json = data["incidents"][0]
        self.assertEqual(inc_json["incident_id"], "INC-0001")
        self.assertEqual(inc_json["worker_id"], 11)
        self.assertEqual(inc_json["zone_type"], "restricted")
        self.assertEqual(inc_json["evidence"], "data/incidents/evidence/INC-0001.jpg")

    def test_12_incident_ids_are_unique(self):
        """12. Incident IDs are unique."""
        inc_mgr = IncidentManager(cooldown_seconds=0.0)
        ids = set()
        for i in range(5):
            inc = inc_mgr.create_incident(
                worker_id=i,
                event="restricted_zone_entry",
                timestamp=float(i * 10),
                frame=i * 300,
                zone="Robotics Cell A",
                severity="HIGH",
                description=f"Inc {i}",
                dwell_time=0.0,
                ppe_status={},
                worker_bbox=[0, 0, 10, 10],
                worker_center=[5, 5],
            )
            self.assertIsNotNone(inc)
            ids.add(inc.incident_id)

        self.assertEqual(len(ids), 5)
        self.assertIn("INC-0001", ids)
        self.assertIn("INC-0005", ids)

    def test_13_query_api_methods(self):
        """Query API methods for Phase 4."""
        inc_mgr = IncidentManager(cooldown_seconds=0.0)
        inc_mgr.create_incident(
            worker_id=1, event="restricted_zone_entry", timestamp=1.0, frame=30,
            zone="Zone A", severity="HIGH", description="", dwell_time=0.0, ppe_status={},
            worker_bbox=[], worker_center=[], zone_type="restricted"
        )
        inc_mgr.create_incident(
            worker_id=2, event="restricted_zone_dwell", timestamp=6.0, frame=180,
            zone="Zone A", severity="CRITICAL", description="", dwell_time=5.0, ppe_status={},
            worker_bbox=[], worker_center=[], zone_type="restricted"
        )
        inc_mgr.create_incident(
            worker_id=1, event="warning_zone_entry", timestamp=2.0, frame=60,
            zone="Zone B", severity="WARNING", description="", dwell_time=0.0, ppe_status={},
            worker_bbox=[], worker_center=[], zone_type="warning"
        )

        self.assertEqual(len(inc_mgr.get_all_incidents()), 3)
        self.assertEqual(len(inc_mgr.get_incidents_by_severity("CRITICAL")), 1)
        self.assertEqual(len(inc_mgr.get_incidents_by_worker(1)), 2)
        self.assertEqual(len(inc_mgr.get_incidents_by_zone("Zone A")), 2)


if __name__ == "__main__":
    unittest.main()
