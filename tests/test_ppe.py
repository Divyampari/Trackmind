"""
Unit tests for PPE detection module (PPEDetector, WorkerPPEHistory, parse_ppe_predictions, cropping).
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.ppe_detector import (
    PPEDetector,
    PPEResult,
    WorkerPPEHistory,
    parse_ppe_predictions,
    crop_worker_image,
)


class TestPPEDetector(unittest.TestCase):

    def test_parse_ppe_positive_and_negative_predictions(self):
        # Positive case
        preds_pos = [
            {"class": "helmet", "confidence": 0.85, "x": 10, "y": 10, "width": 20, "height": 20},
            {"class": "vest", "confidence": 0.90, "x": 10, "y": 40, "width": 30, "height": 40},
            {"class": "boots", "confidence": 0.78, "x": 10, "y": 80, "width": 20, "height": 20},
        ]
        res_pos = parse_ppe_predictions(preds_pos, confidence_threshold=0.50)
        self.assertEqual(res_pos["helmet"][0], True)
        self.assertEqual(res_pos["vest"][0], True)
        self.assertEqual(res_pos["boots"][0], True)

        # Negative case
        preds_neg = [
            {"class": "no helmet", "confidence": 0.80, "x": 10, "y": 10, "width": 20, "height": 20},
            {"class": "no vest", "confidence": 0.75, "x": 10, "y": 40, "width": 30, "height": 40},
        ]
        res_neg = parse_ppe_predictions(preds_neg, confidence_threshold=0.50)
        self.assertEqual(res_neg["helmet"][0], False)
        self.assertEqual(res_neg["vest"][0], False)
        self.assertIsNone(res_neg["boots"][0])

    def test_parse_ppe_below_threshold(self):
        preds = [
            {"class": "helmet", "confidence": 0.20},
            {"class": "vest", "confidence": 0.15},
        ]
        res = parse_ppe_predictions(preds, confidence_threshold=0.50)
        self.assertIsNone(res["helmet"][0])
        self.assertIsNone(res["vest"][0])
        self.assertIsNone(res["boots"][0])

    def test_temporal_smoothing_hysteresis(self):
        history = WorkerPPEHistory(window_size=5)
        # Initially unknown
        self.assertEqual(history.current_helmet, "unknown")

        # 1st positive observation
        h, v, b = history.update(True, None, None)
        self.assertEqual(h, True)

        # Neutral observation (no detection) should retain True
        h, v, b = history.update(None, None, None)
        self.assertEqual(h, True)

        # A single weak negative observation does NOT immediately flip to False
        h, v, b = history.update(False, None, None)
        self.assertEqual(h, True)

        # Second negative observation flips to False
        h, v, b = history.update(False, None, None)
        self.assertEqual(h, False)

    def test_padded_crop_clamping(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Bbox near top edge
        bbox = [10, 5, 50, 60]
        crop = crop_worker_image(frame, bbox, padding_w=0.2, padding_top=0.3, padding_bot=0.1)
        self.assertIsNotNone(crop)
        self.assertGreater(crop.shape[0], 0)
        self.assertGreater(crop.shape[1], 0)

    def test_unconfigured_fallback_is_unknown(self):
        detector = PPEDetector(api_key="")
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        res = detector.update_worker(frame, [10, 10, 50, 100], worker_id=1, timestamp=1.0)
        self.assertEqual(res.helmet, "unknown")
        self.assertEqual(res.vest, "unknown")
        self.assertEqual(res.boots, "unknown")

    def test_contract_dict_format(self):
        res = PPEResult(worker_id=1, helmet=True, vest=False, boots="unknown")
        d = res.to_dict()
        self.assertEqual(d, {"helmet": True, "vest": False, "boots": "unknown"})


if __name__ == "__main__":
    unittest.main()
