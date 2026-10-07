"""
FactoryGuard AI - Integration Test for Phase 1 & Phase 2 Output Contract

This test verifies that tracking and PPE data can be consumed by
future phases WITHOUT importing or depending on YOLO/ByteTrack/Roboflow.

Purpose:
    Demonstrates that another Python module (e.g., Phase 2 Behaviour
    Intelligence) can read and use tracking_data.json independently.

Usage:
    python tests/test_integration.py
    python tests/test_integration.py data/tracking/tracking_data.json
"""

import json
import os
import sys


def load_tracking_data(filepath: str) -> dict:
    """Load tracking data from JSON - exactly how Phase 2 would do it.

    This function does NOT import anything from the detection module.
    It reads the JSON file directly, demonstrating the decoupled interface.
    """
    if not os.path.isfile(filepath):
        print(f"ERROR: Tracking data file not found: {filepath}")
        print("Run Phase 1 first: python app.py videos/your_video.mp4")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_schema(data: dict) -> list:
    """Validate the tracking data schema and return any issues found."""
    issues = []

    # Top-level fields
    required_top = ["video", "fps", "total_frames", "resolution", "frames"]
    for field in required_top:
        if field not in data:
            issues.append(f"Missing top-level field: '{field}'")

    if "fps" in data and not isinstance(data["fps"], (int, float)):
        issues.append(f"'fps' should be numeric, got {type(data['fps']).__name__}")

    if "total_frames" in data and not isinstance(data["total_frames"], int):
        issues.append(f"'total_frames' should be int, got {type(data['total_frames']).__name__}")

    if "resolution" in data:
        if not isinstance(data["resolution"], list) or len(data["resolution"]) != 2:
            issues.append("'resolution' should be [width, height]")

    if "frames" not in data:
        return issues

    # Frame-level validation
    for i, frame in enumerate(data["frames"][:20]):  # Check first 20 frames
        if "frame" not in frame:
            issues.append(f"Frame {i}: missing 'frame' field")
        if "timestamp" not in frame:
            issues.append(f"Frame {i}: missing 'timestamp' field")
        if "workers" not in frame:
            issues.append(f"Frame {i}: missing 'workers' field")
            continue

        for j, worker in enumerate(frame["workers"]):
            prefix = f"Frame {frame.get('frame', i)}, Worker {j}"
            if "id" not in worker:
                issues.append(f"{prefix}: missing 'id'")
            if "bbox" not in worker:
                issues.append(f"{prefix}: missing 'bbox'")
            elif not isinstance(worker["bbox"], list) or len(worker["bbox"]) != 4:
                issues.append(f"{prefix}: 'bbox' should be [x1, y1, x2, y2]")
            if "center" not in worker:
                issues.append(f"{prefix}: missing 'center'")
            elif not isinstance(worker["center"], list) or len(worker["center"]) != 2:
                issues.append(f"{prefix}: 'center' should be [cx, cy]")
            if "confidence" not in worker:
                issues.append(f"{prefix}: missing 'confidence'")
            # Optional PPE validation
            if "ppe" in worker and isinstance(worker["ppe"], dict):
                for key in ("helmet", "vest", "boots"):
                    if key in worker["ppe"]:
                        val = worker["ppe"][key]
                        if val not in (True, False, "unknown"):
                            issues.append(f"{prefix}: ppe.{key} should be True/False/'unknown', got {val}")

    return issues


def demonstrate_phase2_consumption(data: dict) -> None:
    """Show how Phase 2 would consume tracking and PPE data."""
    print("\n" + "=" * 60)
    print("  Phase 2 Integration Demonstration")
    print("  (How Phase 2 consumes tracking + PPE data)")
    print("=" * 60)

    print(f"\n  Video source : {data['video']}")
    print(f"  FPS          : {data['fps']}")
    print(f"  Total frames : {data['total_frames']}")
    print(f"  Resolution   : {data['resolution'][0]}x{data['resolution'][1]}")

    # Collect all unique worker IDs
    all_worker_ids = set()
    frames_with_workers = 0
    total_detections = 0
    ppe_samples = 0

    for frame in data["frames"]:
        if frame["workers"]:
            frames_with_workers += 1
        for worker in frame["workers"]:
            all_worker_ids.add(worker["id"])
            total_detections += 1
            if "ppe" in worker:
                ppe_samples += 1

    print(f"\n  Unique worker IDs  : {sorted(all_worker_ids)}")
    print(f"  Total detections   : {total_detections}")
    print(f"  Frames with workers: {frames_with_workers} / {data['total_frames']}")
    print(f"  Records with PPE   : {ppe_samples}")

    # Show example: accessing a specific worker's trajectory and PPE status
    if all_worker_ids:
        example_id = min(id for id in all_worker_ids if id >= 0) if any(id >= 0 for id in all_worker_ids) else list(all_worker_ids)[0]
        print(f"\n  --- Example: Trajectory & PPE of Worker {example_id} ---")
        count = 0
        for frame in data["frames"]:
            for worker in frame["workers"]:
                if worker["id"] == example_id and count < 5:
                    ppe_str = f" | ppe={worker['ppe']}" if "ppe" in worker else ""
                    print(
                        f"    Frame {frame['frame']:>5} | "
                        f"t={frame['timestamp']:>7.3f}s | "
                        f"bbox={worker['bbox']} | "
                        f"center={worker['center']} | "
                        f"conf={worker['confidence']:.4f}"
                        f"{ppe_str}"
                    )
                    count += 1
        if count == 5:
            print(f"    ... (showing first 5 occurrences)")

    # Demonstrate: Phase 2 movement computation
    print("\n  --- Example: Computing movement between frames ---")
    if all_worker_ids:
        example_id = min(id for id in all_worker_ids if id >= 0) if any(id >= 0 for id in all_worker_ids) else list(all_worker_ids)[0]
        prev_center = None
        prev_timestamp = None
        shown = 0
        for frame in data["frames"]:
            for worker in frame["workers"]:
                if worker["id"] == example_id:
                    if prev_center is not None and shown < 3:
                        dx = worker["center"][0] - prev_center[0]
                        dy = worker["center"][1] - prev_center[1]
                        dt = frame["timestamp"] - prev_timestamp
                        dist = (dx**2 + dy**2) ** 0.5
                        speed = dist / dt if dt > 0 else 0
                        print(
                            f"    Frame {frame['frame']:>5}: "
                            f"moved {dist:.1f}px in {dt:.3f}s "
                            f"(~{speed:.1f} px/s)"
                        )
                        shown += 1
                    prev_center = worker["center"]
                    prev_timestamp = frame["timestamp"]

    print("\n  Phase 2 can use this data for:")
    print("    - PPE Compliance Alerts: Worker X missing helmet in restricted zone")
    print("    - Zone Entry Detection: compare worker['center'] to zone polygons")
    print("    - Dwell Time Analysis: track duration worker spent in hazard regions")
    print("    - Movement Patterns: compute speed/acceleration from trajectory")


def main():
    tracking_path = os.path.join("data", "tracking", "tracking_data.json")

    if len(sys.argv) > 1:
        tracking_path = sys.argv[1]

    print()
    print("=" * 60)
    print("  FactoryGuard AI - Phase 1 Integration Test")
    print("=" * 60)
    print(f"\n  Testing file: {tracking_path}")

    # Step 1: Load data (no YOLO/ByteTrack/Roboflow imports needed)
    data = load_tracking_data(tracking_path)
    print("  [PASS] Tracking data loaded successfully")

    # Step 2: Validate schema
    issues = validate_schema(data)
    if issues:
        print(f"\n  [FAIL] Schema validation found {len(issues)} issue(s):")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  [PASS] Schema validation passed")

    # Step 3: Verify data fields accessible
    try:
        _ = data["video"]
        _ = data["fps"]
        _ = data["total_frames"]
        _ = data["resolution"]
        _ = data["frames"]
        if data["frames"]:
            frame = data["frames"][0]
            _ = frame["frame"]
            _ = frame["timestamp"]
            _ = frame["workers"]
            if frame["workers"]:
                worker = frame["workers"][0]
                _ = worker["id"]
                _ = worker["bbox"]
                _ = worker["center"]
                _ = worker["confidence"]
        print("  [PASS] All data fields accessible")
    except KeyError as e:
        print(f"  [FAIL] Missing field: {e}")

    # Step 4: Verify coordinate conventions
    if data["frames"]:
        for frame in data["frames"]:
            for worker in frame["workers"]:
                x1, y1, x2, y2 = worker["bbox"]
                cx, cy = worker["center"]
                if x2 <= x1 or y2 <= y1:
                    print(f"  [WARN] Invalid bbox in frame {frame['frame']}: {worker['bbox']}")
                    break
                expected_cx = (x1 + x2) / 2.0
                expected_cy = (y1 + y2) / 2.0
                if abs(cx - expected_cx) > 1.0 or abs(cy - expected_cy) > 1.0:
                    print(f"  [WARN] Center mismatch in frame {frame['frame']}")
                    break
            else:
                continue
            break
        else:
            print("  [PASS] Coordinate conventions verified")

    # Step 5: Verify timestamps
    if len(data["frames"]) >= 2:
        f0 = data["frames"][0]
        f1 = data["frames"][1]
        expected_dt = 1.0 / data["fps"]
        actual_dt = f1["timestamp"] - f0["timestamp"]
        if abs(actual_dt - expected_dt) < 0.01:
            print("  [PASS] Timestamps are FPS-derived (not wall-clock)")
        else:
            print(f"  [WARN] Timestamp delta ({actual_dt:.4f}) differs from expected ({expected_dt:.4f})")

    # Step 6: Show Phase 2 consumption demo
    demonstrate_phase2_consumption(data)

    print("\n" + "=" * 60)
    print("  Integration Test Complete")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
