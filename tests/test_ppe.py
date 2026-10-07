"""
FactoryGuard AI - PPE Detection Unit & Integration Tests

Tests:
1. PPE response parsing (positive, negative, unknown, confidence filtering).
2. Worker bounding box cropping and boundary clamping.
3. Worker-level caching, cache hits, and refresh intervals.
4. Multi-worker independent PPE tracking.
5. Graceful handling of network/API failures and missing API keys.
6. Data contract and JSON serialisation.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from detection.ppe_detector import (
    PPEResult,
    PPEDetector,
    parse_ppe_predictions,
    load_env_file,
    DEFAULT_PPE_INTERVAL_SECONDS,
)
from detection.tracker import WorkerRecord, FrameRecord, TrackingData


class TestPPEDetection(unittest.TestCase):
    """Test suite for Roboflow PPE detection and caching."""

    def test_parse_all_positive_ppe(self):
        """Verify helmet, vest, and boots are mapped to True when detected."""
        predictions = [
            {"class": "helmet", "confidence": 0.92},
            {"class": "vest", "confidence": 0.88},
            {"class": "boots", "confidence": 0.75},
        ]
        result = parse_ppe_predictions(predictions, confidence_threshold=0.35)
        self.assertTrue(result.helmet)
        self.assertTrue(result.vest)
        self.assertTrue(result.boots)
        self.assertTrue(result.is_fully_compliant())
        self.assertEqual(len(result.get_violations()), 0)

    def test_parse_all_negative_ppe(self):
        """Verify 'no helmet', 'no vest', 'no boots' are mapped to False."""
        predictions = [
            {"class": "no helmet", "confidence": 0.85},
            {"class": "no vest", "confidence": 0.90},
            {"class": "no boots", "confidence": 0.70},
        ]
        result = parse_ppe_predictions(predictions, confidence_threshold=0.35)
        self.assertFalse(result.helmet)
        self.assertFalse(result.vest)
        self.assertFalse(result.boots)
        self.assertFalse(result.is_fully_compliant())
        self.assertEqual(set(result.get_violations()), {"helmet", "vest", "boots"})

    def test_parse_mixed_and_unknown_ppe(self):
        """Verify partially detected PPE items with 'unknown' for absent items."""
        predictions = [
            {"class": "safety helmet", "confidence": 0.91},
            {"class": "no vest", "confidence": 0.83},
            # Boots not in predictions
        ]
        result = parse_ppe_predictions(predictions, confidence_threshold=0.35)
        self.assertTrue(result.helmet)
        self.assertFalse(result.vest)
        self.assertEqual(result.boots, "unknown")
        self.assertEqual(result.get_violations(), ["vest"])

    def test_parse_confidence_threshold_filtering(self):
        """Predictions below confidence threshold must be filtered out to 'unknown'."""
        predictions = [
            {"class": "helmet", "confidence": 0.20},  # Below 0.35 threshold
            {"class": "vest", "confidence": 0.80},
        ]
        result = parse_ppe_predictions(predictions, confidence_threshold=0.35)
        self.assertEqual(result.helmet, "unknown")
        self.assertTrue(result.vest)

    def test_parse_conflicting_predictions_resolves_to_highest_confidence(self):
        """If model returns both 'helmet' (0.85) and 'no helmet' (0.40), pick highest confidence."""
        predictions = [
            {"class": "helmet", "confidence": 0.85},
            {"class": "no helmet", "confidence": 0.40},
        ]
        result = parse_ppe_predictions(predictions, confidence_threshold=0.35)
        self.assertTrue(result.helmet)

    def test_worker_crop_clamping_and_validation(self):
        """Verify worker cropping safely clamps to image boundaries and rejects tiny crops."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Valid crop
        crop = PPEDetector.crop_worker(frame, [100, 100, 200, 300])
        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape, (200, 100, 3))

        # Crop exceeding frame bounds (should clamp safely)
        crop_clamped = PPEDetector.crop_worker(frame, [-50, -50, 700, 600])
        self.assertIsNotNone(crop_clamped)
        self.assertEqual(crop_clamped.shape, (480, 640, 3))

        # Tiny / invalid crop (e.g. 5x5 px)
        crop_tiny = PPEDetector.crop_worker(frame, [100, 100, 105, 105])
        self.assertIsNone(crop_tiny)

    def test_caching_and_refresh_intervals(self):
        """Verify worker PPE status is cached and only refreshed after interval elapsed."""
        detector = PPEDetector(inference_interval_seconds=1.0)
        worker_id = 7

        # 1. Initial frame at t=0.0: should refresh
        self.assertTrue(detector.should_refresh(worker_id, current_timestamp=0.0))

        # Simulate caching a result at t=0.0
        detector._cache[worker_id] = PPEResult(helmet=True, vest=True, boots="unknown", last_updated=0.0)

        # 2. Subsequent frame at t=0.5s: cache hit (should NOT refresh)
        self.assertFalse(detector.should_refresh(worker_id, current_timestamp=0.5))
        cached = detector.get_worker_ppe(worker_id)
        self.assertTrue(cached.helmet)
        self.assertTrue(cached.vest)

        # 3. Subsequent frame at t=1.1s: interval elapsed (should refresh)
        self.assertTrue(detector.should_refresh(worker_id, current_timestamp=1.1))

    def test_multi_worker_independent_caching(self):
        """Verify multiple worker IDs maintain independent PPE states and update timestamps."""
        detector = PPEDetector(inference_interval_seconds=1.0)

        detector._cache[1] = PPEResult(helmet=True, vest=True, boots=True, last_updated=1.0)
        detector._cache[2] = PPEResult(helmet=False, vest=True, boots="unknown", last_updated=1.5)

        # At t=1.6s, Worker 1 should refresh (>1.0s elapsed), Worker 2 should NOT (<1.0s elapsed)
        self.assertTrue(detector.should_refresh(1, current_timestamp=2.1))
        self.assertFalse(detector.should_refresh(2, current_timestamp=2.1))

        self.assertTrue(detector.get_worker_ppe(1).helmet)
        self.assertFalse(detector.get_worker_ppe(2).helmet)

    def test_api_failure_graceful_fallback(self):
        """Verify network/API errors do not crash and cleanly return 'unknown'."""
        detector = PPEDetector(api_key="mock_key", inference_interval_seconds=1.0)

        # Mock client that raises an exception
        mock_client = MagicMock()
        mock_client.infer.side_effect = RuntimeError("Network timeout or connection refused")
        detector._client = mock_client
        detector._client_initialized = True

        dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.infer_worker_crop(dummy_crop)

        # Must return unknown without throwing
        self.assertEqual(result.helmet, "unknown")
        self.assertEqual(result.vest, "unknown")
        self.assertEqual(result.boots, "unknown")

    def test_worker_record_serialization_with_ppe(self):
        """Verify WorkerRecord serialises ppe dict correctly into tracking JSON contract."""
        worker = WorkerRecord(
            id=3,
            bbox=[100.5, 120.2, 250.8, 410.6],
            center=[175.6, 265.4],
            confidence=0.9123,
            ppe={"helmet": True, "vest": False, "boots": "unknown"},
        )
        d = worker.to_dict()

        self.assertEqual(d["id"], 3)
        self.assertEqual(d["bbox"], [100.5, 120.2, 250.8, 410.6])
        self.assertEqual(d["center"], [175.6, 265.4])
        self.assertEqual(d["confidence"], 0.9123)
        self.assertEqual(d["ppe"], {"helmet": True, "vest": False, "boots": "unknown"})


if __name__ == "__main__":
    unittest.main()
