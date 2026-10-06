"""
FactoryGuard AI - Main Application Entry Point

Phase 1: Computer Vision Foundation

Processes factory video footage or live webcam streams to detect and track
workers using YOLOv8 + tuned ByteTrack. Produces an annotated output video
and structured tracking data (JSON) for downstream phases.

Usage:
    # 1. Video File Input:
    python app.py                                        # Uses default/detected video
    python app.py videos/my_test.mp4                     # Specify video path
    python app.py --source videos/my_test.mp4            # Using --source flag

    # 2. Live Webcam Input:
    python app.py --source webcam                        # Default laptop/USB webcam
    python app.py --webcam                               # Shortcut flag for webcam
    python app.py --source webcam --camera-id 0          # Explicit camera index
    python app.py --source webcam --record-webcam        # Record webcam output to mp4
"""

import argparse
import logging
import sys
import os
import time

from detection.tracker import (
    process_video,
    process_webcam,
    DEFAULT_MODEL,
    DEFAULT_CONFIDENCE,
    DEFAULT_TRACKER_CFG,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRACKING_DIR,
    DEFAULT_OUTPUT_VIDEO,
    DEFAULT_TRACKING_FILE,
    DEFAULT_WEBCAM_TRACKING_FILE,
    DEFAULT_WEBCAM_OUTPUT_VIDEO,
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
            "Phase 1: Detect and track workers in factory video footage or live webcam. "
            "Produces annotated video and structured tracking data."
        ),
    )
    # Positional or named source
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Path to the input video file (e.g., videos/my_test.mp4).",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Input source: 'webcam' (or camera index '0') or path to video file.",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Shortcut to run in live webcam tracking mode.",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device index for webcam mode (default: 0).",
    )
    parser.add_argument(
        "--record-webcam",
        action="store_true",
        help="Record the live annotated webcam session to an MP4 video.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the live OpenCV preview window (useful for headless / background execution).",
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
        help="Path for the annotated output video.",
    )
    parser.add_argument(
        "--output-tracking",
        default=None,
        help="Path for tracking JSON output.",
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
        f"Place a video file in the '{videos_dir}/' directory or use --source webcam."
    )


def main() -> None:
    """Main application entry point."""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    logger = logging.getLogger("factoryguard.app")

    # --- Determine mode: Webcam vs Video File ---
    is_webcam = False
    camera_id = args.camera_id

    if args.webcam:
        is_webcam = True
    elif args.source is not None:
        source_str = args.source.strip().lower()
        if source_str in ("webcam", "camera", "cam", "live"):
            is_webcam = True
        elif source_str.isdigit():
            is_webcam = True
            camera_id = int(source_str)
        else:
            # Source is a video file path
            video_path = args.source
    elif args.video is not None:
        if args.video.strip().lower() in ("webcam", "camera", "cam"):
            is_webcam = True
        else:
            video_path = args.video
    else:
        # Default fallback: check for video file
        try:
            video_path = find_default_video()
        except FileNotFoundError:
            # If no video is found, prompt help
            print("\nNo video file found in 'videos/' and no --source specified.")
            print("To run webcam mode:   python app.py --source webcam")
            print("To run video file:    python app.py videos/my_test.mp4\n")
            sys.exit(1)

    # --- Banner ---
    print()
    print("=" * 60)
    print("  FactoryGuard AI - Phase 1: Computer Vision Foundation")
    mode_label = "Live Webcam Stream" if is_webcam else "Pre-recorded Video"
    print(f"  Worker Detection & Tracking Pipeline [{mode_label}]")
    print("=" * 60)
    print()

    # --- Execute according to mode ---
    start_time = time.time()

    if is_webcam:
        # --- Webcam Mode ---
        tracking_out = args.output_tracking or os.path.join(
            DEFAULT_TRACKING_DIR, DEFAULT_WEBCAM_TRACKING_FILE
        )
        video_out = None
        if args.record_webcam or args.output_video:
            video_out = args.output_video or os.path.join(
                DEFAULT_OUTPUT_DIR, DEFAULT_WEBCAM_OUTPUT_VIDEO
            )

        print(f"  Source Mode    : Live Webcam (Camera Index: {camera_id})")
        print(f"  YOLO Model     : {args.model}")
        print(f"  Confidence     : {args.confidence}")
        print(f"  Tracker Config : {args.tracker_config}")
        print(f"  Recording      : {video_out or 'Disabled (use --record-webcam to enable)'}")
        print(f"  Tracking Data  : {tracking_out}")
        print(f"  Live Preview   : {'Disabled' if args.no_preview else 'Enabled (Press q to stop)'}")
        print()

        try:
            tracking_data = process_webcam(
                camera_index=camera_id,
                model_path=args.model,
                confidence=args.confidence,
                tracker_config=args.tracker_config,
                output_video_path=video_out,
                tracking_output_path=tracking_out,
                show_preview=not args.no_preview,
            )
        except RuntimeError as e:
            logger.error("Webcam runtime error: %s", e)
            sys.exit(1)
        except Exception as e:
            logger.error("Unexpected error during webcam tracking: %s", e, exc_info=True)
            sys.exit(1)

    else:
        # --- Video File Mode ---
        if not os.path.isfile(video_path):
            logger.error("Video file not found: %s", video_path)
            sys.exit(1)

        tracking_out = args.output_tracking or os.path.join(
            DEFAULT_TRACKING_DIR, DEFAULT_TRACKING_FILE
        )
        video_out = args.output_video or os.path.join(
            DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_VIDEO
        )

        print(f"  Source Video   : {video_path}")
        print(f"  YOLO Model     : {args.model}")
        print(f"  Confidence     : {args.confidence}")
        print(f"  Tracker Config : {args.tracker_config}")
        print(f"  Output Video   : {video_out}")
        print(f"  Tracking Data  : {tracking_out}")
        print()

        try:
            tracking_data = process_video(
                video_path=video_path,
                model_path=args.model,
                confidence=args.confidence,
                tracker_config=args.tracker_config,
                output_video_path=video_out,
                tracking_output_path=tracking_out,
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
    print(f"  Stream FPS       : {tracking_data.fps}")
    print(f"  Resolution       : {tracking_data.resolution[0]}x{tracking_data.resolution[1]}")
    print(f"  Total Detections : {total_workers_detected}")
    print(f"  Unique Workers   : {len(unique_ids)}")
    print(f"  Processing Time  : {elapsed:.1f}s")
    if tracking_data.total_frames > 0 and elapsed > 0:
        print(f"  Avg FPS          : {tracking_data.total_frames / elapsed:.1f}")
    print()
    if not is_webcam or (is_webcam and video_out):
        print(f"  Output Video     : {video_out}")
    print(f"  Tracking Data    : {tracking_out}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
