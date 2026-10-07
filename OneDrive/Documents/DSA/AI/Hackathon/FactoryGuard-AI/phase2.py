"""
FactoryGuard AI - Phase 2: Behaviour Intelligence Pipeline

Workflow:
    1. Video Input: Accepts video path (e.g., python phase2.py videos/test_factory.mp4)
    2. Zone Configuration: Loads existing JSON zone config or opens interactive UI on first frame
    3. Frame-by-Frame Processing: Streams video frames without loading full video into memory
    4. Phase 1 Tracking: Person detection via YOLOv8 + persistent worker IDs via ByteTrack
    5. Phase 2 Spatial & Temporal Analysis:
        - Evaluates worker midpoint [cx, cy] against polygons via cv2.pointPolygonTest
        - Detects entry, exit, and prolonged presence dwell violations (> threshold)
    6. Outputs:
        - Annotated video: outputs/phase2_output.mp4
        - Structured events JSON: data/behaviour/behaviour_events.json

Usage:
    python phase2.py videos/test_factory.mp4
    python phase2.py videos/test_factory.mp4 --draw-zones
    python phase2.py videos/test_factory.mp4 --zones data/zones/factory_zone_config.json
    python phase2.py videos/test_factory.mp4 --output-video outputs/phase2_output.mp4 --dwell-threshold 5.0
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from behaviour.engine import BehaviourEngine
from behaviour.zone_manager import Zone, ZoneManager, ZoneType
from detection.tracker import WorkerTracker, DEFAULT_MODEL, DEFAULT_CONFIDENCE
from zone_config import interactive_zone_designer


def setup_logging(verbose: bool = False) -> None:
    """Configure application logging format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="FactoryGuard AI - Phase 2",
        description="Phase 2: Behaviour Intelligence & Safety Zone Monitoring on Video Footage.",
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Path to input video file (e.g., videos/test_factory.mp4).",
    )
    parser.add_argument(
        "--video", "-v-path",
        dest="video_opt",
        default=None,
        help="Path to input video file (alternative flag).",
    )
    parser.add_argument(
        "--zones", "-z",
        default=os.path.join("data", "zones", "factory_zone_config.json"),
        help="Path to zone configuration JSON (default: data/zones/factory_zone_config.json).",
    )
    parser.add_argument(
        "--draw-zones", "-d", "--interactive", "-i",
        action="store_true",
        dest="draw_zones",
        help="Open interactive OpenCV zone drawer on the first frame of the video.",
    )
    parser.add_argument(
        "--output-video", "-o",
        default=os.path.join("outputs", "phase2_output.mp4"),
        help="Path for annotated output video (default: outputs/phase2_output.mp4).",
    )
    parser.add_argument(
        "--events", "-e",
        default=os.path.join("data", "behaviour", "behaviour_events.json"),
        help="Path for structured behaviour events JSON (default: data/behaviour/behaviour_events.json).",
    )
    parser.add_argument(
        "--report", "-r",
        default=os.path.join("data", "incidents", "incidents_report.json"),
        help="Path for incident report JSON (default: data/incidents/incidents_report.json).",
    )
    parser.add_argument(
        "--dwell-threshold",
        type=float,
        default=5.0,
        help="Dwell violation threshold in seconds (default: 5.0s).",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="Alert deduplication cooldown window in seconds (default: 5.0s).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO detection model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Minimum detection confidence (default: {DEFAULT_CONFIDENCE}).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose / debug logs.",
    )
    return parser.parse_args()


def find_default_video() -> str:
    """Locate available video file in videos/ directory."""
    videos_dir = "videos"
    if not os.path.isdir(videos_dir):
        raise FileNotFoundError(f"'{videos_dir}/' directory not found. Please specify video path.")

    preferred_files = ["test_factory.mp4", "my_test.mp4", "factory_workers_test.mp4"]
    for pref in preferred_files:
        path = os.path.join(videos_dir, pref)
        if os.path.isfile(path):
            return path

    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"}
    for filename in sorted(os.listdir(videos_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in video_extensions:
            return os.path.join(videos_dir, filename)

    raise FileNotFoundError(f"No video files found in '{videos_dir}/'. Please provide video path.")


def run_phase2(
    video_path: str,
    zones_path: str = os.path.join("data", "zones", "factory_zone_config.json"),
    draw_zones: bool = False,
    output_video_path: str = os.path.join("outputs", "phase2_output.mp4"),
    events_output_path: str = os.path.join("data", "behaviour", "behaviour_events.json"),
    report_output_path: str = os.path.join("data", "incidents", "incidents_report.json"),
    dwell_threshold: float = 5.0,
    cooldown_seconds: float = 5.0,
    model_path: str = DEFAULT_MODEL,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict:
    """Execute complete Phase 2 Behaviour Intelligence pipeline on a video file."""
    logger = logging.getLogger("factoryguard.phase2")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")

    # Initialize Zone Manager
    zone_mgr = ZoneManager()
    zone_mgr.video_name = os.path.basename(video_path)

    # Check if user requested interactive drawing OR no zone file exists
    should_draw = draw_zones or not os.path.isfile(zones_path)

    if should_draw:
        logger.info("Launching interactive zone designer on: %s", video_path)
        interactive_zone_designer(video_path=video_path, output_path=zones_path, clear_existing=draw_zones)

    # Load zone configuration
    if os.path.isfile(zones_path):
        zone_mgr.load_zones(zones_path)
        logger.info("Loaded %d zone(s) from %s", len(zone_mgr.zones), zones_path)
    else:
        logger.warning("Zone config %s not found. Using default factory preset zones.", zones_path)
        cap_temp = cv2.VideoCapture(video_path)
        w = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        h = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        cap_temp.release()
        zone_mgr.generate_default_preset_zones((w, h), video_name=os.path.basename(video_path))
        zone_mgr.save_zones(zones_path)

    # Override dwell thresholds if specified
    if dwell_threshold is not None:
        for z in zone_mgr.zones:
            if z.type == ZoneType.RESTRICTED.value:
                z.dwell_threshold = dwell_threshold

    # Initialize Behaviour Engine
    engine = BehaviourEngine(
        zone_manager=zone_mgr,
        interactive_zones=False,  # Already handled above
        cooldown_seconds=cooldown_seconds,
    )

    # Process video stream frame-by-frame
    start_time = time.time()
    summary = engine.process_video(
        video_path=video_path,
        output_video_path=output_video_path,
        behaviour_output_path=events_output_path,
        report_output_path=report_output_path,
        model_path=model_path,
        confidence=confidence,
    )
    elapsed = time.time() - start_time
    summary["elapsed_time"] = elapsed

    return summary


def main() -> None:
    """Main CLI entry point."""
    args = parse_args()
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger("factoryguard.phase2")

    print()
    print("=" * 68)
    print("  FactoryGuard AI - Phase 2: Behaviour Intelligence Pipeline")
    print("  Worker Safety Zone Monitoring, Dwell Tracking & Behavior Events")
    print("=" * 68)
    print()

    # Resolve video path
    video_path = args.video or args.video_opt
    if video_path is None:
        try:
            video_path = find_default_video()
            logger.info("Auto-detected video: %s", video_path)
        except FileNotFoundError as e:
            logger.error(str(e))
            print("\nUsage: python phase2.py <video_path>")
            print("Example: python phase2.py videos/test_factory.mp4\n")
            sys.exit(1)

    print(f"  Input Video Source   : {video_path}")
    print(f"  Zone Configuration   : {args.zones}")
    print(f"  Interactive Drawing  : {args.draw_zones}")
    print(f"  Dwell Threshold      : {args.dwell_threshold}s")
    print(f"  Alert Cooldown       : {args.cooldown}s")
    print(f"  Annotated Video Out  : {args.output_video}")
    print(f"  Behaviour Events Out : {args.events}")
    print(f"  Incident Report Out  : {args.report}")
    print("=" * 68)
    print()

    try:
        summary = run_phase2(
            video_path=video_path,
            zones_path=args.zones,
            draw_zones=args.draw_zones,
            output_video_path=args.output_video,
            events_output_path=args.events,
            report_output_path=args.report,
            dwell_threshold=args.dwell_threshold,
            cooldown_seconds=args.cooldown,
            model_path=args.model,
            confidence=args.confidence,
        )
    except Exception as e:
        logger.error("Phase 2 processing failed: %s", e, exc_info=args.verbose)
        sys.exit(1)

    # Print summary
    print()
    print("=" * 68)
    print("  Phase 2 Processing Completed Successfully!")
    print("=" * 68)
    print(f"  Video Processed      : {summary['video']}")
    print(f"  Total Frames         : {summary['total_frames']}")
    print(f"  Resolution           : {summary['resolution'][0]}x{summary['resolution'][1]} @ {summary['fps']:.1f} FPS")
    print(f"  Zones Monitored      : {summary['zones_monitored']}")
    print(f"  Total Incidents/Evts : {summary['total_incidents']}")
    print(f"  Processing Time      : {summary['elapsed_time']:.2f}s")
    if summary['total_frames'] > 0 and summary['elapsed_time'] > 0:
        print(f"  Processing Speed     : {summary['total_frames'] / summary['elapsed_time']:.1f} FPS")
    print()
    print(f"  [+] Annotated Video  : {summary['output_video_path']}")
    print(f"  [+] Behaviour Events : {summary['behaviour_events_path']}")
    print(f"  [+] Incident Report  : {summary['incidents_report_path']}")
    print("=" * 68)
    print()


if __name__ == "__main__":
    main()
