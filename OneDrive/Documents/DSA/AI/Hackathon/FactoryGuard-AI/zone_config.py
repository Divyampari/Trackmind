"""
FactoryGuard AI - Safety Zone Configuration Utility

Phase 2 Extension: User-Configurable Safety Zones

Provides a simple, interactive, and reusable CLI utility to define, edit, save,
and inspect camera/video-specific safety zones.

User Flow:
    1. Displays the first frame of the input video.
    2. Allows operator to click 4 corner points (or polygon vertices) around the area.
    3. Asks for Zone Name (e.g. 'Machine Zone A') and Zone Type ('restricted' or 'warning').
    4. Saves configuration to external JSON file (default: data/zones/factory_zone_config.json).
    5. Configuration is loaded by Phase 2 behaviour engine during video analysis.

Usage:
    python zone_config.py                                      # Interactive setup on default video
    python zone_config.py --video videos/my_test.mp4           # Specify input video
    python zone_config.py --output data/zones/factory_zone_config.json
    python zone_config.py --preset                             # Generate default preset zones
    python zone_config.py --list                               # List saved zones in JSON
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from typing import List, Optional, Dict, Any, Tuple

import cv2
import numpy as np

# Import existing Zone data model and ZoneManager
from behaviour.zone_manager import Zone, ZoneManager, ZoneType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("factoryguard.zone_config")


def find_default_video() -> Optional[str]:
    """Find a valid video file in videos/ directory."""
    videos_dir = "videos"
    if not os.path.isdir(videos_dir):
        return None

    preferred_files = ["my_test.mp4", "factory_workers_test.mp4", "test_factory.mp4"]
    for pref in preferred_files:
        path = os.path.join(videos_dir, pref)
        if os.path.isfile(path):
            return path

    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"}
    for fname in sorted(os.listdir(videos_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in video_extensions:
            return os.path.join(videos_dir, fname)

    return None


def get_video_first_frame(video_path: Optional[str], resolution=(1080, 720)) -> Tuple[np.ndarray, str, List[int]]:
    """Retrieve first frame from video file, or generate canvas if video not available."""
    if video_path and os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                vname = os.path.basename(video_path)
                return frame, vname, [w, h]

    # Fallback blank factory canvas
    w, h = resolution
    canvas = np.full((h, w, 3), 50, dtype=np.uint8)
    # Draw simple factory floor grid lines
    for x in range(0, w, 60):
        cv2.line(canvas, (x, 0), (x, h), (70, 70, 70), 1)
    for y in range(0, h, 60):
        cv2.line(canvas, (0, y), (w, y), (70, 70, 70), 1)
    cv2.putText(canvas, "FACTORY FLOOR CANVAS", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    return canvas, "factory_canvas", [w, h]


def interactive_zone_designer(
    video_path: Optional[str] = None,
    output_path: str = os.path.join("data", "zones", "factory_zone_config.json"),
    clear_existing: bool = False,
) -> str:
    """Launch interactive OpenCV window for operator point selection."""
    frame, video_name, resolution = get_video_first_frame(video_path)
    w, h = resolution[0], resolution[1]

    zone_mgr = ZoneManager()
    zone_mgr.video_name = video_name
    zone_mgr.resolution = resolution

    # If output file already exists and not starting fresh, load existing zones
    if os.path.isfile(output_path) and not clear_existing:
        zone_mgr.load_zones(output_path)

    current_points: List[List[int]] = []
    temp_zones: List[Zone] = [] if clear_existing else list(zone_mgr.zones)
    window_name = "FactoryGuard AI - User Safety Zone Configurator"

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append([x, y])

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n" + "=" * 65)
    print("  USER-CONFIGURABLE SAFETY ZONE UTILITY")
    print("=" * 65)
    print("  1. Click 4 points (or corner vertices) around area to monitor")
    print("  2. Press 'r' : Save polygon as RESTRICTED zone ('Machine Zone X')")
    print("  3. Press 'w' : Save polygon as WARNING zone ('Walkway Zone Y')")
    print("  4. Press 'c' : Clear current selected points")
    print("  5. Press 'd' : Delete/Clear ALL existing zones")
    print("  6. Press 's' / ESC / 'q' : Save zone config to JSON and Exit")
    print("=" * 65 + "\n")

    zone_counter = len(temp_zones) + 1

    while True:
        display = frame.copy()

        # Render already defined zones
        for z in temp_zones:
            pts = np.array(z.polygon, dtype=np.int32)
            color = (0, 0, 255) if z.type == ZoneType.RESTRICTED.value else (0, 255, 255)

            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
            cv2.polylines(display, [pts], isClosed=True, color=color, thickness=2)

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
            for idx, pt in enumerate(current_points):
                cv2.circle(display, (pt[0], pt[1]), 6, (0, 255, 0), -1)
                cv2.putText(
                    display,
                    f"P{idx+1}",
                    (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            if len(current_points) > 1:
                pts = np.array(current_points, dtype=np.int32)
                cv2.polylines(display, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

        # Render top instruction HUD
        cv2.rectangle(display, (0, 0), (w, 45), (30, 30, 30), -1)
        inst_text = (
            f"Pts: {len(current_points)} | 'r': RESTRICTED | 'w': WARNING | "
            f"'c': Clear Pts | 'd': Delete All | 's': Save"
        )
        cv2.putText(display, inst_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('c'):
            current_points = []
            logger.info("Cleared selected points.")

        elif key == ord('d'):
            temp_zones = []
            current_points = []
            zone_counter = 1
            logger.info("Deleted all defined safety zones.")

        elif key == ord('r'):
            if len(current_points) >= 3:
                z_name = f"Machine Zone {chr(64 + zone_counter)}"
                new_zone = Zone(
                    name=z_name,
                    type=ZoneType.RESTRICTED.value,
                    polygon=[[p[0], p[1]] for p in current_points],
                    dwell_threshold=5.0,
                    description="Restricted machinery cell",
                )
                temp_zones.append(new_zone)
                logger.info("Added RESTRICTED zone '%s' with %d points.", z_name, len(current_points))
                zone_counter += 1
                current_points = []
            else:
                logger.warning("Please click at least 3 or 4 points to define a zone polygon.")

        elif key == ord('w'):
            if len(current_points) >= 3:
                z_name = f"Warning Zone {chr(64 + zone_counter)}"
                new_zone = Zone(
                    name=z_name,
                    type=ZoneType.WARNING.value,
                    polygon=[[p[0], p[1]] for p in current_points],
                    dwell_threshold=10.0,
                    description="Forklift / transit warning corridor",
                )
                temp_zones.append(new_zone)
                logger.info("Added WARNING zone '%s' with %d points.", z_name, len(current_points))
                zone_counter += 1
                current_points = []
            else:
                logger.warning("Please click at least 3 or 4 points to define a zone polygon.")

        elif key in (ord('s'), ord('q'), 27):
            break

    cv2.destroyWindow(window_name)

    zone_mgr.zones = temp_zones
    saved_file = zone_mgr.save_zones(output_path)

    print("\n" + "=" * 65)
    print("  ZONE CONFIGURATION SAVED SUCCESSFULLY!")
    print("=" * 65)
    print(f"  Saved Path : {saved_file}")
    print(f"  Total Zones: {len(zone_mgr.zones)}")
    for z in zone_mgr.zones:
        print(f"    - [{z.type.upper()}] '{z.name}' ({len(z.polygon)} vertices, dwell={z.dwell_threshold}s)")
    print("=" * 65 + "\n")

    return saved_file


def generate_preset_config(output_path: str = os.path.join("data", "zones", "factory_zone_config.json")) -> str:
    """Generate clean preset zone config for 1080p / 720p frames without opening GUI."""
    zm = ZoneManager()
    zm.generate_default_preset_zones((1080, 720), video_name="factory_workers_test.mp4")
    saved_path = zm.save_zones(output_path)
    print(f"Generated preset zone configuration: {saved_path}")
    return saved_path


def main():
    parser = argparse.ArgumentParser(
        prog="FactoryGuard AI - Safety Zone Configurator",
        description="Define, save, load, and inspect user-configurable safety zones.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Input video file to draw zones on (e.g. videos/my_test.mp4).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "zones", "factory_zone_config.json"),
        help="Target JSON file path (default: data/zones/factory_zone_config.json).",
    )
    parser.add_argument(
        "--preset",
        action="store_true",
        help="Generate default preset zone configuration non-interactively.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List and display existing saved zone configuration JSON.",
    )
    parser.add_argument(
        "--fresh", "--clear",
        action="store_true",
        dest="clear_existing",
        help="Start with a fresh clean frame (wipe existing saved zones).",
    )

    args = parser.parse_args()

    if args.list:
        if os.path.isfile(args.output):
            with open(args.output, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"\n--- Zone Configuration File ({args.output}) ---")
            print(content)
        else:
            print(f"Zone config file '{args.output}' does not exist.")
        return

    if args.preset:
        generate_preset_config(args.output)
        return

    video_path = args.video or find_default_video()
    interactive_zone_designer(video_path=video_path, output_path=args.output, clear_existing=args.clear_existing)


if __name__ == "__main__":
    main()
