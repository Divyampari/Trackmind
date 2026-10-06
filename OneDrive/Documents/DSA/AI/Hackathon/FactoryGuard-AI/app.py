"""
FactoryGuard AI - Main Application Entry Point

Phase 1: Computer Vision Foundation

Processes factory video footage to detect and track workers using
YOLOv8 + tuned ByteTrack. Produces an annotated output video and structured
tracking data (JSON) for downstream phases.

Usage:
    python app.py                              # Uses default/detected video
    python app.py videos/my_test.mp4           # Specify video path
    python app.py videos/my_test.mp4 --model yolov8s.pt --confidence 0.45
"""

import argparse
import logging
import sys
import os
import time

from detection.tracker import (
    process_video,
    DEFAULT_MODEL,
    DEFAULT_CONFIDENCE,
    DEFAULT_TRACKER_CFG,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure application logging.

    Args:
        verbose: If True, set DEBUG level. Otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="FactoryGuard AI",
        description=(
            "Phase 1: Detect and track workers in factory video footage. "
            "Produces annotated video and structured tracking data."
        ),
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Path to the input video file (e.g., videos/my_test.mp4).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO model name or path (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Minimum detection confidence (default: {DEFAULT_CONFIDENCE}).",
    )
    parser.add_argument(
        "--tracker-config",
        default=DEFAULT_TRACKER_CFG,
        help=f"Custom ByteTrack tracker YAML configuration (default: {DEFAULT_TRACKER_CFG}).",
    )
    parser.add_argument(
        "--output-video",
        default=None,
        help="Path for the annotated output video (default: outputs/tracked_output.mp4).",
    )
    parser.add_argument(
        "--output-tracking",
        default=None,
        help="Path for tracking JSON output (default: data/tracking/tracking_data.json).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging.",
    )
    return parser.parse_args()


def find_default_video() -> str:
    """Attempt to find a video file in the videos/ directory.

    Returns:
        Path to the first video file found.

    Raises:
        FileNotFoundError: If no video files are found.
    """
    videos_dir = "videos"
    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"}

    if not os.path.isdir(videos_dir):
        raise FileNotFoundError(
            f"'{videos_dir}/' directory not found. "
            f"Create it and place a video file inside."
        )

    # Prioritize my_test.mp4 if present
    preferred = os.path.join(videos_dir, "my_test.mp4")
    if os.path.isfile(preferred):
        return preferred

    for filename in sorted(os.listdir(videos_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in video_extensions:
            return os.path.join(videos_dir, filename)

    raise FileNotFoundError(
        f"No video files found in '{videos_dir}/'. "
        f"Place a video file in the '{videos_dir}/' directory."
    )


def main() -> None:
    """Main application entry point."""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    logger = logging.getLogger("factoryguard.app")

    # --- Banner ---
    print()
    print("=" * 60)
    print("  FactoryGuard AI - Phase 1: Computer Vision Foundation")
    print("  Worker Detection & Tracking Pipeline (Refined)")
    print("=" * 60)
    print()

    # --- Resolve video path ---
    video_path = args.video
    if video_path is None:
        try:
            video_path = find_default_video()
            logger.info("Auto-detected video: %s", video_path)
        except FileNotFoundError as e:
            logger.error(str(e))
            print("\nUsage: python app.py <video_path>")
            print("Example: python app.py videos/my_test.mp4")
            sys.exit(1)
    elif not os.path.isfile(video_path):
        logger.error("Video file not found: %s", video_path)
        sys.exit(1)

    # --- Display configuration ---
    print(f"  Input Video    : {video_path}")
    print(f"  YOLO Model     : {args.model}")
    print(f"  Confidence     : {args.confidence}")
    print(f"  Tracker Config : {args.tracker_config}")
    print(f"  Output Video   : {args.output_video or 'outputs/tracked_output.mp4'}")
    print(f"  Tracking Data  : {args.output_tracking or 'data/tracking/tracking_data.json'}")
    print()

    # --- Process video ---
    start_time = time.time()

    try:
        tracking_data = process_video(
            video_path=video_path,
            model_path=args.model,
            confidence=args.confidence,
            tracker_config=args.tracker_config,
            output_video_path=args.output_video,
            tracking_output_path=args.output_tracking,
        )
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Invalid input: %s", e)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("Runtime error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        sys.exit(1)

    elapsed = time.time() - start_time

    # --- Summary ---
    total_workers_detected = sum(
        len(f.workers) for f in tracking_data.frames
    )
    unique_ids = set()
    for frame in tracking_data.frames:
        for worker in frame.workers:
            if worker.id >= 0:
                unique_ids.add(worker.id)

    print()
    print("=" * 60)
    print("  Processing Complete!")
    print("=" * 60)
    print(f"  Total Frames     : {tracking_data.total_frames}")
    print(f"  Video FPS        : {tracking_data.fps}")
    print(f"  Resolution       : {tracking_data.resolution[0]}x{tracking_data.resolution[1]}")
    print(f"  Total Detections : {total_workers_detected}")
    print(f"  Unique Workers   : {len(unique_ids)}")
    print(f"  Processing Time  : {elapsed:.1f}s")
    if tracking_data.total_frames > 0 and elapsed > 0:
        print(f"  Avg FPS          : {tracking_data.total_frames / elapsed:.1f}")
    print()
    print(f"  Output Video     : {args.output_video or 'outputs/tracked_output.mp4'}")
    print(f"  Tracking Data    : {args.output_tracking or 'data/tracking/tracking_data.json'}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
