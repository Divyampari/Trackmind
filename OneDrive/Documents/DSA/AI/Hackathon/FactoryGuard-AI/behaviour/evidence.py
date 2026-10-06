"""
FactoryGuard AI - Evidence Capture Manager

Phase 2: Behaviour and Safety Intelligence Layer

Captures, annotates, and saves visual evidence frame snapshots whenever a safety
incident occurs. Evidence images include highlighted safety zone boundaries,
worker bounding box, incident badge, timestamp, and severity header banner.

This ensures full explainability for security operators and safety auditors.
"""

import os
import logging
from typing import List, Optional, Any
import cv2
import numpy as np

from behaviour.zone_manager import Zone, ZoneType

logger = logging.getLogger("factoryguard.behaviour.evidence")


class EvidenceManager:
    """Manages visual evidence frame capture and image file output.

    Args:
        output_dir: Directory where evidence frame images are saved.
    """

    def __init__(self, output_dir: str = os.path.join("data", "incidents", "evidence")):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def capture_evidence(
        self,
        frame: np.ndarray,
        incident_id: str,
        worker_id: int,
        event: str,
        timestamp: float,
        frame_number: int,
        zone_name: str,
        severity: str,
        worker_bbox: List[float],
        zones: List[Zone],
        description: str = "",
    ) -> str:
        """Annotate and save an evidence image snapshot for a safety incident.

        Args:
            frame: Original BGR video frame image.
            incident_id: Unique incident ID (e.g. 'INC-0001').
            worker_id: Persistent tracking ID.
            event: Event type string.
            timestamp: Video timestamp in seconds.
            frame_number: Frame index.
            zone_name: Violating zone name.
            severity: 'WARNING', 'HIGH', or 'CRITICAL'.
            worker_bbox: Bounding box [x1, y1, x2, y2].
            zones: List of active Zone instances to render.
            description: Human readable text description.

        Returns:
            Filepath to the saved evidence image.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame passed for evidence capture (%s). Skipping snapshot.", incident_id)
            return ""

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Render Zone Polygons
        for zone in zones:
            pts = np.array(zone.polygon, dtype=np.int32)
            is_target_zone = (zone.name.lower() == zone_name.lower())

            if is_target_zone:
                color = (0, 0, 255) if severity in ("HIGH", "CRITICAL") else (0, 255, 255)
                thickness = 4
                alpha = 0.35
            else:
                color = (0, 165, 255) if zone.type == ZoneType.WARNING.value else (0, 0, 200)
                thickness = 2
                alpha = 0.15

            # Semi-transparent fill
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=thickness)

            # Zone label
            cx = int(np.mean([p[0] for p in zone.polygon]))
            cy = int(np.mean([p[1] for p in zone.polygon]))
            cv2.putText(
                annotated,
                f"ZONE: {zone.name}",
                (cx - 40, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        # 2. Draw Violating Worker Bounding Box
        if len(worker_bbox) == 4:
            x1, y1, x2, y2 = [int(c) for c in worker_bbox]
            box_color = (0, 0, 255) if severity in ("HIGH", "CRITICAL") else (0, 255, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)

            # Alert Badge above bbox
            badge_text = f"VIOLATION: Worker {worker_id} | {event}"
            (tw, th), bl = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            lbl_y = max(y1 - 10, th + 10)
            cv2.rectangle(annotated, (x1, lbl_y - th - 6), (x1 + tw + 10, lbl_y + bl), box_color, -1)
            cv2.putText(annotated, badge_text, (x1 + 5, lbl_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 3. Top Banner Overlay
        banner_h = 55
        cv2.rectangle(annotated, (0, 0), (w, banner_h), (20, 20, 20), -1)

        severity_color = (0, 0, 255) if severity in ("HIGH", "CRITICAL") else (0, 255, 255)
        # Left severity pill
        cv2.rectangle(annotated, (10, 10), (140, 45), severity_color, -1)
        cv2.putText(annotated, severity, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Right metadata text
        header_text = f"[{incident_id}] Worker {worker_id} - {event} | Zone: {zone_name} | t={timestamp:.2f}s"
        cv2.putText(annotated, header_text, (155, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 4. Save Evidence Snapshot
        filename = f"{incident_id}_w{worker_id}_frame{frame_number}.jpg"
        filepath = os.path.join(self.output_dir, filename)

        success = cv2.imwrite(filepath, annotated)
        if success:
            logger.info("Saved evidence snapshot: %s", filepath)
            return filepath
        else:
            logger.error("Failed to write evidence frame image to: %s", filepath)
            return ""
