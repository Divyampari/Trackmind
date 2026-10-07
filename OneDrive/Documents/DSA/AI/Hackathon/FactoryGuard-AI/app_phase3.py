"""
FactoryGuard AI - Phase 3 Main Application Entry Point

PHASE 3: Incident, Evidence & Reporting Intelligence Layer

Transforms Phase 2 worker-zone events into structured, evidence-backed
safety incidents with lifecycle tracking, visual evidence capture, machine-readable
JSON reports, and query APIs for Phase 4 dashboards.

Usage:
    python app_phase3.py
    python app_phase3.py --video videos/my_test.mp4
    python app_phase3.py data/tracking/tracking_data.json --output-report data/incidents/incidents_report.json
"""

import argparse
import json
import logging
import os
import sys
import time

from behaviour.engine import BehaviourEngine
from behaviour.zone_manager import ZoneManager
from detection.tracker import get_tracking_data, process_video


def setup_logging(verbose: bool = False) -> None:
    """Configure application logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 3."""
    parser = argparse.ArgumentParser(
        prog="FactoryGuard AI - Phase 3",
        description="Phase 3: Incident Intelligence, Visual Evidence & Reporting Pipeline.",
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
        "--output-video",
        default="outputs/behaviour_output.mp4",
        help="Path for annotated output video (default: outputs/behaviour_output.mp4).",
    )
    parser.add_argument(
        "--output-report",
        default=os.path.join("data", "incidents", "incidents_report.json"),
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


def run_phase3(
    tracking_json_path: str = None,
    video_path: str = None,
    zone_config_path: str = None,
    output_video_path: str = "outputs/behaviour_output.mp4",
    output_report_path: str = "data/incidents/incidents_report.json",
    cooldown_seconds: float = 5.0,
) -> dict:
    """Programmatic entry point for Phase 3 pipeline execution."""
    tracking_path = tracking_json_path or os.path.join("data", "tracking", "tracking_data.json")

    if video_path is None:
        try:
            video_path = find_default_video()
        except FileNotFoundError:
            pass

    if not os.path.isfile(tracking_path) and video_path and os.path.isfile(video_path):
        process_video(video_path=video_path, tracking_output_path=tracking_path)

    if not os.path.isfile(tracking_path):
        raise FileNotFoundError(f"Tracking data file not found: {tracking_path}")

    tracking_data = get_tracking_data(tracking_path)
    zone_mgr = ZoneManager(zone_config_path)

    engine = BehaviourEngine(
        zone_manager=zone_mgr,
        cooldown_seconds=cooldown_seconds,
    )

    summary = engine.process_tracking_data(
        tracking_data=tracking_data,
        video_path=video_path,
        output_video_path=output_video_path,
        report_output_path=output_report_path,
    )

    return summary


def main() -> None:
    """Main CLI entry point for Phase 3."""
    args = parse_args()
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger("factoryguard.app_phase3")

    print()
    print("=" * 65)
    print("  FactoryGuard AI - Phase 3: Incident Intelligence & Reporting")
    print("=" * 65)
    print()

    summary = run_phase3(
        tracking_json_path=args.tracking_json,
        video_path=args.video,
        zone_config_path=args.zones,
        output_video_path=args.output_video,
        output_report_path=args.output_report,
        cooldown_seconds=args.cooldown,
    )

    print()
    print("=" * 65)
    print("  Phase 3 Processing Complete!")
    print("=" * 65)
    print(f"  Source Video       : {summary['video']}")
    print(f"  Total Frames       : {summary['total_frames']}")
    print(f"  Zones Monitored    : {summary['zones_monitored']}")
    print(f"  Total Incidents    : {summary['total_incidents']}")
    print(f"  Incident Report    : {summary['incidents_report_path']}")
    print(f"  Evidence Folder    : data/incidents/evidence/")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
