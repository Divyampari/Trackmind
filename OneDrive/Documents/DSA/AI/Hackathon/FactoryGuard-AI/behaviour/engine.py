"""
FactoryGuard AI - Source-Independent Behaviour Engine

Phase 2: Behaviour and Safety Intelligence Layer

High-level coordinator uniting ZoneManager, ZoneDetector, BehaviourAnalyzer,
IncidentManager, EvidenceManager, and PPEInterface.

SOURCE INDEPENDENCE:
    Processes worker tracking data identically whether sourced from offline recorded
    videos or live webcam camera streams. The core behaviour engine operates on the
    unified Phase 1 tracking representation.

Outputs:
    - Annotated output video with safety zone overlays, worker violation badges, and HUD.
    - Structured incident summary JSON report (data/incidents/incidents_report.json).
    - Visual evidence snapshots for each incident (data/incidents/evidence/...).
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import cv2
import numpy as np

from behaviour.zone_manager import Zone, ZoneManager, ZoneType
from behaviour.zone_detector import ZoneDetector, ZoneEvent
from behaviour.ppe_interface import PPEInterface
from behaviour.behaviour_analyzer import BehaviourAnalyzer
from behaviour.incident_manager import IncidentManager, SafetyIncident
from behaviour.evidence import EvidenceManager

logger = logging.getLogger("factoryguard.behaviour.engine")


class BehaviourEngine:
    """Source-independent safety intelligence engine for FactoryGuard AI.

    Args:
        zone_manager: Optional ZoneManager instance.
        zone_config_path: Path to JSON zone configuration file.
        interactive_zones: If True, prompt operator to draw zones interactively on first frame.
        cooldown_seconds: Deduplication alert window in seconds.
    """

    def __init__(
        self,
        zone_manager: Optional[ZoneManager] = None,
        zone_config_path: Optional[str] = None,
        interactive_zones: bool = False,
        cooldown_seconds: float = 5.0,
    ):
        self.zone_manager = zone_manager or ZoneManager(zone_config_path)
        self.interactive_zones = interactive_zones

        self.zone_detector = ZoneDetector(anchor_point="center", grace_period=1.0)
        self.incident_manager = IncidentManager(cooldown_seconds=cooldown_seconds)
        self.evidence_manager = EvidenceManager()
        self.ppe_interface = PPEInterface()

        self.analyzer = BehaviourAnalyzer(
            incident_manager=self.incident_manager,
            evidence_manager=self.evidence_manager,
            ppe_interface=self.ppe_interface,
        )

    def process_frame(
        self,
        frame: Optional[np.ndarray],
        frame_number: int,
        timestamp: float,
        workers: List[Any],
    ) -> Tuple[Optional[np.ndarray], List[SafetyIncident]]:
        """Process a single frame (used for both video playback and live webcam streaming).

        Args:
            frame: BGR NumPy frame image (can be None if analyzing JSON without video).
            frame_number: 0-indexed frame index.
            timestamp: Frame timestamp in seconds.
            workers: List of WorkerRecord objects or worker dicts from Phase 1.

        Returns:
            Tuple of (annotated_frame, list_of_new_incidents_in_this_frame).
        """
        # Ensure default preset zones exist if none were configured
        if not self.zone_manager.zones and frame is not None:
            h, w = frame.shape[:2]
            self.zone_manager.generate_default_preset_zones((w, h))

        # 1. Update ZoneDetector state
        events = self.zone_detector.update(
            frame_number=frame_number,
            timestamp=timestamp,
            workers=workers,
            zones=self.zone_manager.zones,
        )

        # 2. Analyze events and generate incidents
        new_incidents = self.analyzer.process_frame_events(
            events=events,
            frame=frame,
            zones=self.zone_manager.zones,
        )

        # 3. Annotate frame if provided
        annotated = None
        if frame is not None:
            annotated = self._annotate_frame(frame, frame_number, timestamp, workers)

        return annotated, new_incidents

    def process_tracking_data(
        self,
        tracking_data: Union[dict, Any],
        video_path: Optional[str] = None,
        output_video_path: Optional[str] = None,
        report_output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process an entire video or saved Phase 1 TrackingData object.

        Args:
            tracking_data: TrackingData dataclass or parsed dict from Phase 1.
            video_path: Path to raw input video (required for annotated video render & evidence capture).
            output_video_path: Destination path for Phase 2 annotated output video.
            report_output_path: Destination path for Phase 2 JSON incident report.

        Returns:
            Dictionary summary of Phase 2 behaviour analysis results.
        """
        # Normalize tracking data
        if hasattr(tracking_data, "to_dict"):
            data_dict = tracking_data.to_dict()
        else:
            data_dict = tracking_data

        video_name = data_dict.get("video", "video")
        fps = data_dict.get("fps", 30.0)
        total_frames = data_dict.get("total_frames", len(data_dict.get("frames", [])))
        resolution = data_dict.get("resolution", [1920, 1080])

        self.zone_manager.video_name = video_name
        self.zone_manager.resolution = resolution

        # Open video source if provided
        cap = None
        writer = None
        if video_path and os.path.isfile(video_path):
            cap = cv2.VideoCapture(video_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or fps

            # Check interactive zone setup request
            if self.interactive_zones:
                ret, first_frame = cap.read()
                if ret:
                    self.zone_manager.interactive_define_zones(first_frame, video_name=video_name)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to start

            if output_video_path:
                os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

        # Ensure preset zones if empty
        if not self.zone_manager.zones:
            self.zone_manager.generate_default_preset_zones((resolution[0], resolution[1]), video_name=video_name)

        logger.info(
            "Starting Phase 2 Behaviour Analysis: '%s' (%d zones configured)",
            video_name, len(self.zone_manager.zones)
        )

        frames = data_dict.get("frames", [])
        for f_idx, frame_record in enumerate(frames):
            frame_num = frame_record.get("frame", f_idx)
            ts = frame_record.get("timestamp", frame_num / fps)
            workers = frame_record.get("workers", [])

            raw_frame = None
            if cap and cap.isOpened():
                ret, raw_frame = cap.read()
                if not ret:
                    raw_frame = None

            annotated_frame, _ = self.process_frame(
                frame=raw_frame,
                frame_number=frame_num,
                timestamp=ts,
                workers=workers,
            )

            if writer and annotated_frame is not None:
                writer.write(annotated_frame)

        if cap:
            cap.release()
        if writer:
            writer.release()

        # Save incident report
        dest_report = report_output_path or os.path.join("data", "incidents", "incidents_report.json")
        saved_report_path = self.incident_manager.save_report(dest_report)

        summary = {
            "video": video_name,
            "total_frames": total_frames,
            "zones_monitored": len(self.zone_manager.zones),
            "total_incidents": len(self.incident_manager.incidents),
            "incidents_report_path": saved_report_path,
            "output_video_path": output_video_path or "",
        }
        return summary

    def _annotate_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        workers: List[Any],
    ) -> np.ndarray:
        """Render safety zones, worker status badges, active incident highlights, and HUD."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Render Safety Zones
        for zone in self.zone_manager.zones:
            pts = np.array(zone.polygon, dtype=np.int32)
            color = (0, 0, 255) if zone.type == ZoneType.RESTRICTED.value else (0, 255, 255)

            # Semi-transparent overlay
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.20, annotated, 0.80, 0, annotated)
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

            # Zone Label
            cx = int(np.mean([p[0] for p in zone.polygon]))
            cy = int(np.mean([p[1] for p in zone.polygon]))
            cv2.putText(
                annotated,
                f"[{zone.type.upper()}] {zone.name}",
                (cx - 50, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        # 2. Render Workers with Safety Badges
        active_violations_count = 0
        for worker in workers:
            w_id = worker.id if hasattr(worker, "id") else worker["id"]
            if w_id < 0:
                continue

            bbox = worker.bbox if hasattr(worker, "bbox") else worker["bbox"]
            x1, y1, x2, y2 = [int(c) for c in bbox]

            # Determine worker safety status
            status_text = f"Worker {w_id}"
            box_color = (0, 255, 128)  # Green for normal

            # Check active zone states for this worker
            in_restricted = False
            in_warning = False
            dwell_alert = False

            for (w_key, z_name), state in self.zone_detector.active_states.items():
                if w_key == w_id:
                    if state.zone_type == ZoneType.RESTRICTED.value:
                        in_restricted = True
                        if state.dwell_alerted:
                            dwell_alert = True
                    elif state.zone_type == ZoneType.WARNING.value:
                        in_warning = True

            if in_restricted:
                active_violations_count += 1
                if dwell_alert:
                    box_color = (0, 0, 255)
                    status_text = f"Worker {w_id} | CRITICAL (Dwell Breach)"
                else:
                    box_color = (0, 0, 220)
                    status_text = f"Worker {w_id} | RESTRICTED AREA"
            elif in_warning:
                box_color = (0, 255, 255)
                status_text = f"Worker {w_id} | WARNING ZONE"

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Label box
            (tw, th), bl = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            lbl_y = max(y1 - 8, th + 5)
            cv2.rectangle(annotated, (x1, lbl_y - th - 4), (x1 + tw + 6, lbl_y + bl), (40, 40, 40), -1)
            cv2.putText(annotated, status_text, (x1 + 3, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 3. Render Top HUD Bar
        hud_h = 35
        cv2.rectangle(annotated, (0, 0), (w, hud_h), (25, 25, 25), -1)
        hud_text = (
            f"Frame: {frame_number} | t={timestamp:.2f}s | "
            f"Workers: {len(workers)} | Active Violations: {active_violations_count} | "
            f"Total Incidents: {len(self.incident_manager.incidents)}"
        )
        cv2.putText(annotated, hud_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        return annotated
