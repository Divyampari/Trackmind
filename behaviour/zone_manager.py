"""
FactoryGuard AI - Zone Manager

Phase 2: Behaviour and Safety Intelligence Layer

Provides configurable, camera/video-specific safety zone management.
Zones are NOT hardcoded; each video or camera session can have its own
custom polygon layout, zone types, and dwell-time thresholds.

Zone Types:
    - RESTRICTED ("restricted"): High-danger area (e.g., automated machinery, chemical storage).
    - WARNING ("warning"): Caution area (e.g., forklift aisle, loading dock).

Zone Persistence:
    Zone definitions are saved as JSON files (e.g., data/zones/my_video_zones.json)
    and can be reloaded for subsequent analysis runs without redrawing.
"""

import json
import os
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np

logger = logging.getLogger("factoryguard.behaviour.zone_manager")


class ZoneType(str, Enum):
    """Supported safety zone types."""
    RESTRICTED = "restricted"
    WARNING = "warning"


@dataclass
class Zone:
    """Represents a factory safety zone defined by a polygon.

    Attributes:
        name: Human-readable zone identifier (e.g. 'Machine Zone A').
        type: Zone classification ('restricted' or 'warning').
        polygon: List of [x, y] vertex coordinates in frame pixels.
        dwell_threshold: Duration in seconds before generating dwell violation (default 5.0s for restricted, 10.0s for warning).
        description: Optional textual notes about the zone.
    """
    name: str
    type: str  # "restricted" or "warning"
    polygon: List[List[float]]
    dwell_threshold: float = 5.0
    description: str = ""

    def __post_init__(self):
        # Normalize type
        self.type = self.type.lower()
        if self.type not in (ZoneType.RESTRICTED.value, ZoneType.WARNING.value):
            logger.warning(
                "Zone '%s' has non-standard type '%s'. Defaulting to 'warning'.",
                self.name, self.type
            )

    def contains_point(self, point: List[float]) -> bool:
        """Check whether a point [x, y] is inside or on the boundary of the zone polygon.

        Args:
            point: [cx, cy] coordinate pair in frame pixels.

        Returns:
            True if the point lies inside or on the polygon, False otherwise.
        """
        if len(self.polygon) < 3:
            return False

        pts = np.array(self.polygon, dtype=np.int32)
        px, py = float(point[0]), float(point[1])

        # cv2.pointPolygonTest returns > 0 for inside, 0 for on edge, < 0 for outside
        result = cv2.pointPolygonTest(pts, (px, py), measureDist=False)
        return result >= 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert zone to plain dictionary for JSON serialization."""
        return {
            "name": self.name,
            "type": self.type,
            "polygon": [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in self.polygon],
            "dwell_threshold": round(float(self.dwell_threshold), 2),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Zone":
        """Instantiate Zone from dictionary."""
        default_dwell = 5.0 if data.get("type", "restricted") == "restricted" else 10.0
        return cls(
            name=data["name"],
            type=data.get("type", "restricted"),
            polygon=data["polygon"],
            dwell_threshold=data.get("dwell_threshold", default_dwell),
            description=data.get("description", ""),
        )


class ZoneManager:
    """Manages creation, storage, loading, and interactive configuration of zones.

    Args:
        zones_filepath: Optional path to load/save zone configuration JSON.
    """

    def __init__(self, zones_filepath: Optional[str] = None):
        self.zones: List[Zone] = []
        self.video_name: str = "default"
        self.resolution: List[int] = [1920, 1080]
        self.filepath: Optional[str] = zones_filepath

        if zones_filepath and os.path.isfile(zones_filepath):
            self.load_zones(zones_filepath)

    def add_zone(self, zone: Zone) -> None:
        """Add a safety zone to the manager, replacing any existing zone with same name."""
        self.zones = [z for z in self.zones if z.name.lower() != zone.name.lower()]
        self.zones.append(zone)
        logger.info("Added zone '%s' (%s, threshold=%.1fs)", zone.name, zone.type, zone.dwell_threshold)

    def remove_zone(self, name: str) -> bool:
        """Remove a zone by name."""
        initial_len = len(self.zones)
        self.zones = [z for z in self.zones if z.name.lower() != name.lower()]
        removed = len(self.zones) < initial_len
        if removed:
            logger.info("Removed zone '%s'", name)
        return removed

    def get_zone(self, name: str) -> Optional[Zone]:
        """Find zone by name (case-insensitive)."""
        for z in self.zones:
            if z.name.lower() == name.lower():
                return z
        return None

    def save_zones(self, filepath: Optional[str] = None) -> str:
        """Save current zone configuration to JSON.

        Args:
            filepath: Destination file path. Uses self.filepath if None.

        Returns:
            Path to saved JSON file.
        """
        dest_path = filepath or self.filepath or os.path.join("data", "zones", f"{self.video_name}_zones.json")
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

        payload = {
            "video_name": self.video_name,
            "resolution": self.resolution,
            "zones": [z.to_dict() for z in self.zones],
        }

        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        self.filepath = dest_path
        logger.info("Saved %d zone(s) to: %s", len(self.zones), dest_path)
        return dest_path

    def load_zones(self, filepath: str) -> bool:
        """Load zone configuration from JSON file.

        Args:
            filepath: Path to JSON configuration file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not os.path.isfile(filepath):
            logger.error("Zone config file not found: %s", filepath)
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.video_name = data.get("video_name", "unknown")
            self.resolution = data.get("resolution", [1920, 1080])
            self.zones = [Zone.from_dict(z) for z in data.get("zones", [])]
            self.filepath = filepath
            logger.info("Loaded %d zone(s) from '%s'", len(self.zones), filepath)
            return True
        except Exception as e:
            logger.error("Failed to parse zone configuration file '%s': %s", filepath, e)
            return False

    def generate_default_preset_zones(
        self, video_resolution: Tuple[int, int] = (1920, 1080), video_name: str = "default"
    ) -> None:
        """Generate default preset zones tailored to standard factory frame resolutions.

        This provides immediate out-of-the-box test zones for any video without requiring
        manual drawing if the operator bypasses interactive configuration.

        Args:
            video_resolution: (width, height) tuple in pixels.
            video_name: Name of video or stream.
        """
        w, h = video_resolution
        self.resolution = [w, h]
        self.video_name = video_name

        # Restricted Zone A (e.g. Left/Middle Automated Machine Area)
        r_poly = [
            [int(w * 0.15), int(h * 0.25)],
            [int(w * 0.45), int(h * 0.25)],
            [int(w * 0.45), int(h * 0.70)],
            [int(w * 0.15), int(h * 0.70)],
        ]
        restricted_zone = Zone(
            name="Machine Zone A",
            type=ZoneType.RESTRICTED.value,
            polygon=r_poly,
            dwell_threshold=5.0,
            description="Hazardous robotics assembly cell",
        )

        # Warning Zone B (e.g. Right Side Material Handling / Loading Corridor)
        w_poly = [
            [int(w * 0.55), int(h * 0.30)],
            [int(w * 0.85), int(h * 0.30)],
            [int(w * 0.85), int(h * 0.80)],
            [int(w * 0.55), int(h * 0.80)],
        ]
        warning_zone = Zone(
            name="Chemical Storage & Corridor",
            type=ZoneType.WARNING.value,
            polygon=w_poly,
            dwell_threshold=10.0,
            description="Forklift and material transit warning zone",
        )

        self.zones = [restricted_zone, warning_zone]
        logger.info("Generated default preset zones for resolution %dx%d", w, h)

    def interactive_define_zones(
        self,
        frame: np.ndarray,
        video_name: str = "camera",
        output_filepath: Optional[str] = None,
    ) -> List[Zone]:
        """Launch OpenCV interactive GUI for operator polygon zone definition.

        Allows operator to click points to outline polygon safety zones on the frame,
        assign zone names and types ('restricted' vs 'warning'), and save configuration.

        Key Controls:
            - Left Click : Add vertex point
            - Key 'r'    : Mark zone as RESTRICTED ('Machine Zone X') & complete polygon
            - Key 'w'    : Mark zone as WARNING ('Warning Corridor Y') & complete polygon
            - Key 'c'    : Clear current drawing points
            - Key 's'    : Save all defined zones & exit editor
            - Key 'q'/ESC: Finish editing session

        Args:
            frame: Reference image frame (BGR NumPy array).
            video_name: Name of current video file or stream.
            output_filepath: Target JSON path to save result.

        Returns:
            List of defined Zone instances.
        """
        h, w = frame.shape[:2]
        self.resolution = [w, h]
        self.video_name = video_name

        current_points: List[List[int]] = []
        window_name = "FactoryGuard AI - Interactive Safety Zone Designer"
        temp_zones: List[Zone] = list(self.zones)

        def mouse_callback(event, x, y, flags, param):
            nonlocal current_points
            if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                current_points.append([x, y])

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)

        print("\n" + "=" * 60)
        print("  INTERACTIVE ZONE CONFIGURATION MODE")
        print("=" * 60)
        print("  - Click LEFT or RIGHT MOUSE BUTTON on desired vertex to add polygon point")
        print("  - Press 'r' : Finalize current polygon as RESTRICTED zone")
        print("  - Press 'w' : Finalize current polygon as WARNING zone")
        print("  - Press 'c' : Clear current polygon points")
        print("  - Press 's' / 'q' / ESC : Save & Exit Zone Designer")
        print("=" * 60 + "\n")

        zone_counter = len(temp_zones) + 1

        while True:
            display = frame.copy()

            # Render existing saved zones
            for z in temp_zones:
                pts = np.array(z.polygon, dtype=np.int32)
                color = (0, 0, 255) if z.type == ZoneType.RESTRICTED.value else (0, 255, 255)
                # Fill semi-transparent polygon
                overlay = display.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
                cv2.polylines(display, [pts], isClosed=True, color=color, thickness=2)

                # Label
                cx = int(np.mean([p[0] for p in z.polygon]))
                cy = int(np.mean([p[1] for p in z.polygon]))
                cv2.putText(
                    display,
                    f"[{z.type.upper()}] {z.name}",
                    (cx - 50, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

            # Render current points in progress
            if len(current_points) > 0:
                pts = np.array(current_points, dtype=np.int32)
                for pt in current_points:
                    cv2.circle(display, (pt[0], pt[1]), 5, (0, 255, 0), -1)
                if len(current_points) > 1:
                    cv2.polylines(display, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

            # Render instructions banner on top
            cv2.rectangle(display, (0, 0), (w, 45), (30, 30, 30), -1)
            inst = (
                f"Points: {len(current_points)} | Press 'r': RESTRICTED | "
                f"Press 'w': WARNING | Press 'c': Clear | Press 's': Save & Exit"
            )
            cv2.putText(display, inst, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF

            # Key 'c' -> Clear current points
            if key == ord('c'):
                current_points = []
                logger.info("Cleared current polygon points.")

            # Key 'r' -> Save current polygon as RESTRICTED zone
            elif key == ord('r'):
                if len(current_points) >= 3:
                    z_name = f"Restricted Zone {zone_counter}"
                    new_z = Zone(
                        name=z_name,
                        type=ZoneType.RESTRICTED.value,
                        polygon=[[p[0], p[1]] for p in current_points],
                        dwell_threshold=5.0,
                        description="User-defined restricted area",
                    )
                    temp_zones.append(new_z)
                    logger.info("Created interactive zone: %s", z_name)
                    zone_counter += 1
                    current_points = []
                else:
                    logger.warning("Need at least 3 points to complete a polygon zone.")

            # Key 'w' -> Save current polygon as WARNING zone
            elif key == ord('w'):
                if len(current_points) >= 3:
                    z_name = f"Warning Zone {zone_counter}"
                    new_z = Zone(
                        name=z_name,
                        type=ZoneType.WARNING.value,
                        polygon=[[p[0], p[1]] for p in current_points],
                        dwell_threshold=10.0,
                        description="User-defined warning area",
                    )
                    temp_zones.append(new_z)
                    logger.info("Created interactive zone: %s", z_name)
                    zone_counter += 1
                    current_points = []
                else:
                    logger.warning("Need at least 3 points to complete a polygon zone.")

            # Key 's', 'q', or ESC -> Save and exit editor
            elif key in (ord('s'), ord('q'), 27):
                break

        cv2.destroyWindow(window_name)

        if temp_zones:
            self.zones = temp_zones
            dest_file = output_filepath or os.path.join("data", "zones", f"{self.video_name}_zones.json")
            self.save_zones(dest_file)
        else:
            logger.info("No zones defined interactively. Falling back to default preset zones.")
            self.generate_default_preset_zones(video_resolution=(w, h), video_name=video_name)

        return self.zones
