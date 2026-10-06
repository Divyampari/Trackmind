"""
FactoryGuard AI - Webcam Integration and Schema Tests

Verifies:
1. Graceful error handling for invalid camera indices.
2. Live/simulated webcam stream processing using the shared YOLO + ByteTrack tracker.
3. Schema conformance of webcam_tracking_data.json against Phase 1 & Phase 2 contract.
"""

import json
import os
import sys
import unittest
import cv2

from detection.tracker import (
    WorkerTracker,
    process_webcam,
    get_tracking_data,
    DEFAULT_MODEL,
    DEFAULT_CONFIDENCE,
)
from tests.test_integration import validate_schema


class TestWebcamSupport(unittest.TestCase):
    """Test suite for Phase 1 webcam tracking support."""

    def test_invalid_camera_index_raises_graceful_error(self):
        """Webcam with invalid index (e.g. 999) must raise RuntimeError with helpful message."""
        tracker = WorkerTracker(model_path=DEFAULT_MODEL, confidence=DEFAULT_CONFIDENCE)
        with self.assertRaises(RuntimeError) as ctx:
            tracker.process_webcam(camera_index=999, show_preview=False, max_frames=2)
        
        err_msg = str(ctx.exception)
        self.assertIn("Could not open webcam", err_msg)
        self.assertIn("camera index 999", err_msg)
        print("  [PASS] Invalid camera index raises clear, descriptive RuntimeError.")

    def test_webcam_stream_and_schema(self):
        """Test webcam execution (if camera 0 is openable) and validate output schema."""
        cap = cv2.VideoCapture(0)
        camera_available = cap.isOpened()
        cap.release()

        if not camera_available:
            print("  [SKIP] Physical webcam 0 not accessible in this environment. Skipping live capture test.")
            return

        test_tracking_path = os.path.join("data", "tracking", "test_webcam_tracking.json")
        test_video_path = os.path.join("outputs", "test_webcam_recording.mp4")

        # Process a brief 5-frame live webcam batch in headless mode
        tracking_data = process_webcam(
            camera_index=0,
            output_video_path=test_video_path,
            tracking_output_path=test_tracking_path,
            show_preview=False,
            max_frames=5,
        )

        self.assertIsNotNone(tracking_data)
        self.assertEqual(tracking_data.total_frames, 5)
        self.assertTrue(os.path.isfile(test_tracking_path))

        # Validate schema
        with open(test_tracking_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        issues = validate_schema(data)
        self.assertEqual(len(issues), 0, f"Schema issues found: {issues}")
        self.assertEqual(data["total_frames"], 5)
        self.assertIn("webcam", data["video"])
        self.assertTrue(len(data["frames"]) == 5)

        # Verify timestamp monotonically increases starting from >= 0
        timestamps = [f["timestamp"] for f in data["frames"]]
        self.assertTrue(all(t >= 0 for t in timestamps))
        for i in range(1, len(timestamps)):
            self.assertGreaterEqual(timestamps[i], timestamps[i-1])

        # Verify get_tracking_data loads the file
        loaded = get_tracking_data(test_tracking_path)
        self.assertEqual(loaded["total_frames"], 5)

        # Cleanup test artifacts
        if os.path.isfile(test_tracking_path):
            os.remove(test_tracking_path)
        if os.path.isfile(test_video_path):
            os.remove(test_video_path)

        print("  [PASS] Live webcam capture and schema validation passed successfully.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  FactoryGuard AI - Running Webcam Tests")
    print("=" * 60 + "\n")
    unittest.main()
