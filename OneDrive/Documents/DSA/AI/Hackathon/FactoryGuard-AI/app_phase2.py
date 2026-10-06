"""
FactoryGuard AI - Phase 2 Main Application Entry Point

Phase 2: Behaviour and Safety Intelligence Layer

Processes worker tracking data (from Phase 1 YOLO + ByteTrack) against
configurable safety zones to detect:
    - Restricted zone entries & prolonged dwell presence
    - Warning zone entries & sustained presence
    - Structured safety incident logs & visual evidence snapshots
    - PPE integration compliance status

Usage:
    python app_phase2.py                                       # Default video & tracking data
    python app_phase2.py --interactive                          # Launch interactive zone drawer
    python app_phase2.py data/tracking/tracking_data.json      # Specify tracking data file
    python app_phase2.py --zones data/zones/my_zones.json      # Load specific zone layout
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from behaviour.engine import BehaviourEngine
from behaviour.zone_manager import ZoneManager
from detection.tracker import get_tracking_data, process_video, DEFAULT_MODEL, DEFAULT_CONFIDENCE


def setup_logging(verbose: bool = False) -> None:
    """Configure application logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 2."""
    parser = argparse.ArgumentParser(
        prog="FactoryGuard AI - Phase 2",
        description=(
            "Phase 2: Behaviour & Safety Intelligence Layer. "
            "Evaluates tracked worker movements against safety zones, detects restricted "
            "zone presence & dwell time, generates incidents and visual evidence."
        ),
    )
    parser.add_argument(
        "tracking_json",
        nargs="?",
        default=None,
        help="Path to Phase 1 tracking JSON (default: data/tracking/tracking_data.json).",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Path to input video file (e.g., videos/my_test.mp4).",
    )
    parser.add_argument(
        "--zones",
        default=None,
        help="Path to zone configuration JSON (e.g., data/zones/my_zones.json).",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch OpenCV interactive zone designer to draw zones on first video frame.",
    )
    parser.add_argument(
        "--output-video",
        default="outputs/behaviour_output.mp4",
        help="Path for annotated output video (default: outputs/behaviour_output.mp4).",
    )
    parser.add_argument(
        "--output-report",
        default="data/incidents/incidents_report.json",
        help="Path for JSON incident report output (default: data/incidents/incidents_report.json).",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="Deduplication cooldown window in seconds (default: 5.0s).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging.",
    )
    return parser.parse_args()


def find_default_video() -> str:
    """Find a video file in the videos/ directory."""
    videos_dir = "videos"
    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"}

    if not os.path.isdir(videos_dir):
        raise FileNotFoundError(f"'{videos_dir}/' directory not found.")

    preferred = os.path.join(videos_dir, "my_test.mp4")
    if os.path.isfile(preferred):
        return preferred

    for filename in sorted(os.listdir(videos_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in video_extensions:
            return os.path.join(videos_dir, filename)

    raise FileNotFoundError(f"No video files found in '{videos_dir}/'.")


def main() -> None:
    """Main application entry point for Phase 2."""
    args = parse_args()
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger("factoryguard.app_phase2")

    print()
    print("=" * 65)
    print("  FactoryGuard AI - Phase 2: Behaviour & Safety Intelligence")
    print("  Worker Safety Zone Monitoring, Dwell Analysis & Incidents")
    print("=" * 65)
    print()

    # Resolve tracking JSON and Video Path
    tracking_path = args.tracking_json or os.path.join("data", "tracking", "tracking_data.json")
    video_path = args.video

    if video_path is None:
        try:
            video_path = find_default_video()
            logger.info("Auto-detected input video: %s", video_path)
        except FileNotFoundError:
            logger.warning("No video file found in videos/. Operating in JSON-only mode.")

    # Run Phase 1 tracker if tracking data JSON does not exist but video exists
    if not os.path.isfile(tracking_path) and video_path and os.path.isfile(video_path):
        logger.info("Tracking data '%s' not found. Executing Phase 1 pipeline first...", tracking_path)
        process_video(video_path=video_path, tracking_output_path=tracking_path)

    if not os.path.isfile(tracking_path):
        logger.error(
            "Tracking data file not found: %s\n"
            "Run Phase 1 first or specify a valid video file.", tracking_path
        )
        sys.exit(1)

    # Load Phase 1 tracking data
    logger.info("Loading Phase 1 tracking data from: %s", tracking_path)
    tracking_data = get_tracking_data(tracking_path)

    # Initialize Zone Manager
    zones_file = args.zones
    if zones_file is None:
        default_cfg = os.path.join("data", "zones", "factory_zone_config.json")
        if os.path.isfile(default_cfg):
            zones_file = default_cfg

    zone_mgr = ZoneManager(zones_file)

    # Initialize Phase 2 Behaviour Engine
    engine = BehaviourEngine(
        zone_manager=zone_mgr,
        interactive_zones=args.interactive,
        cooldown_seconds=args.cooldown,
    )

    print(f"  Input Tracking Data : {tracking_path}")
    print(f"  Input Video Source  : {video_path or 'N/A (JSON only)'}")
    print(f"  Interactive Zones   : {args.interactive}")
    print(f"  Zone Config Path    : {args.zones or 'Auto-detected / Preset'}")
    print(f"  Alert Cooldown      : {args.cooldown}s")
    print(f"  Output Video Path   : {args.output_video}")
    print(f"  Incident Report Path: {args.output_report}")
    print()

    # Execute Phase 2 pipeline
    start_time = time.time()

    summary = engine.process_tracking_data(
        tracking_data=tracking_data,
        video_path=video_path,
        output_video_path=args.output_video,
        report_output_path=args.output_report,
    )

    elapsed = time.time() - start_time

    # Display Phase 2 Results Summary
    incidents = engine.incident_manager.incidents

    print()
    print("=" * 65)
    print("  Phase 2 Processing Complete!")
    print("=" * 65)
    print(f"  Video Processed    : {summary['video']}")
    print(f"  Total Frames       : {summary['total_frames']}")
    print(f"  Zones Monitored    : {summary['zones_monitored']}")
    print(f"  Total Incidents    : {len(incidents)}")
    print(f"  Processing Time    : {elapsed:.2f}s")
    print()

    if incidents:
        print("  --- Incident Log Summary ---")
        for inc in incidents:
            ev_str = f" [Evidence: {inc.evidence_frame_path}]" if inc.evidence_frame_path else ""
            print(
                f"    * [{inc.severity:^8}] {inc.incident_id}: Worker {inc.worker_id} "
                f"| {inc.event} in '{inc.zone}' at t={inc.timestamp:.2f}s{ev_str}"
            )
        print()

    print(f"  Output Video       : {args.output_video}")
    print(f"  Incident Report    : {args.output_report}")
    print(f"  Evidence Folder    : data/incidents/evidence/")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
