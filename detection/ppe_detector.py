"""
FactoryGuard AI - PPE Detection Module

Phase 1 & Phase 2 Integration Layer: PPE Detection using Roboflow

This module provides worker-level PPE (Personal Protective Equipment)
detection for factory safety monitoring. It detects:
    1. Helmet (Hard hat)
    2. Safety Vest
    3. Safety Boots

Model:
    Roboflow Hosted Inference Model: "helmet-vest-and-boots-detection/8"

Architecture:
    Video Frame -> YOLO + ByteTrack -> Tracked Worker
        -> Crop Worker BBox -> Roboflow PPE Inference
        -> Worker PPE Cache (refreshed periodically, default: 1.0s)
        -> Normalized PPE Status -> WorkerRecord / Tracking JSON

Safety & Security:
    - Never hardcodes or logs API keys.
    - Loads ROBOFLOW_API_KEY from environment or local .env file.
    - Gracefully falls back to "unknown" on network/API failure without
      crashing the computer vision or behaviour pipeline.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union, Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("factoryguard.ppe")

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_ROBOFLOW_MODEL_ID = "helmet-vest-and-boots-detection/8"
DEFAULT_ROBOFLOW_API_URL = "https://serverless.roboflow.com"
DEFAULT_PPE_INTERVAL_SECONDS = 1.0       # Run PPE inference once per worker per second
DEFAULT_PPE_CONFIDENCE_THRESHOLD = 0.35  # Confidence threshold for PPE item detections
DEFAULT_MAX_CACHE_AGE_SECONDS = 30.0    # Evict worker PPE cache if inactive for >30s
MIN_CROP_DIMENSION = 20                 # Ignore worker crops smaller than 20x20 pixels


# ---------------------------------------------------------------------------
# Lightweight .env loader (zero external dependency)
# ---------------------------------------------------------------------------
def load_env_file(env_path: Optional[str] = None) -> None:
    """Load key-value pairs from a .env file into os.environ if not already set.

    Args:
        env_path: Optional explicit path to .env. Defaults to looking in cwd or project root.
    """
    if env_path is None:
        candidates = [
            ".env",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                env_path = candidate
                break

    if env_path and os.path.isfile(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
            logger.debug("Loaded environment variables from: %s", env_path)
        except Exception as e:
            logger.warning("Could not parse .env file (%s): %s", env_path, e)


# Load .env upon module import if available
load_env_file()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class PPEResult:
    """Normalized PPE detection result for a single worker.

    Attributes:
        helmet: True (detected), False (not worn / 'no helmet' detected), or 'unknown'.
        vest: True (detected), False (not worn / 'no vest' detected), or 'unknown'.
        boots: True (detected), False (not worn / 'no boots' detected), or 'unknown'.
        last_updated: Video/session timestamp in seconds when inference was performed.
        raw_detections: Optional list of raw detected item labels & confidences.
    """
    helmet: Union[bool, str] = "unknown"
    vest: Union[bool, str] = "unknown"
    boots: Union[bool, str] = "unknown"
    last_updated: float = 0.0
    raw_detections: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary representation for JSON export."""
        return {
            "helmet": self.helmet,
            "vest": self.vest,
            "boots": self.boots,
        }

    def is_fully_compliant(self) -> bool:
        """Check if all required PPE items are confirmed worn (True)."""
        return self.helmet is True and self.vest is True and self.boots is True

    def get_violations(self) -> list:
        """Return list of confirmed missing PPE items (where status is explicitly False)."""
        violations = []
        if self.helmet is False:
            violations.append("helmet")
        if self.vest is False:
            violations.append("vest")
        if self.boots is False:
            violations.append("boots")
        return violations


# ---------------------------------------------------------------------------
# PPE Class Categorisation & Parsing
# ---------------------------------------------------------------------------
# Normalised mapping rules for Roboflow model classes
HELMET_POSITIVE_CLASSES = {
    "helmet", "hard-hat", "hardhat", "safety-helmet", "safety helmet", "hard hat"
}
HELMET_NEGATIVE_CLASSES = {
    "no helmet", "no-helmet", "no_helmet", "without helmet", "without-helmet",
    "no hardhat", "no-hardhat", "no hard hat"
}

VEST_POSITIVE_CLASSES = {
    "vest", "safety-vest", "safety vest", "reflective-vest", "reflective vest",
    "jacket", "safety jacket", "hi-viz vest", "hi-viz"
}
VEST_NEGATIVE_CLASSES = {
    "no vest", "no-vest", "no_vest", "without vest", "without-vest", "no jacket"
}

BOOTS_POSITIVE_CLASSES = {
    "boots", "boot", "safety-boots", "safety boots", "safety shoes", "safety-shoes",
    "shoes", "shoe"
}
BOOTS_NEGATIVE_CLASSES = {
    "no boots", "no-boots", "no_boots", "without boots", "without-boots",
    "no shoes", "no-shoes", "no shoe"
}


def parse_ppe_predictions(
    predictions: list,
    confidence_threshold: float = DEFAULT_PPE_CONFIDENCE_THRESHOLD,
) -> PPEResult:
    """Parse raw Roboflow predictions into a structured, normalised PPEResult.

    Args:
        predictions: List of prediction dictionaries from Roboflow response.
        confidence_threshold: Minimum confidence score to accept a prediction.

    Returns:
        PPEResult with helmet, vest, boots set to True, False, or 'unknown'.
    """
    # Track highest confidence prediction for each category: (is_positive, confidence)
    best_helmet: Optional[Tuple[bool, float]] = None
    best_vest: Optional[Tuple[bool, float]] = None
    best_boots: Optional[Tuple[bool, float]] = None
    raw_list = []

    for pred in predictions:
        if not isinstance(pred, dict):
            continue

        raw_class = str(pred.get("class", "")).strip().lower()
        conf = float(pred.get("confidence", 0.0))

        if conf < confidence_threshold or not raw_class:
            continue

        raw_list.append({"class": raw_class, "confidence": round(conf, 3)})

        # 1. Helmet check
        if raw_class in HELMET_POSITIVE_CLASSES:
            if best_helmet is None or conf > best_helmet[1]:
                best_helmet = (True, conf)
        elif raw_class in HELMET_NEGATIVE_CLASSES:
            if best_helmet is None or conf > best_helmet[1]:
                best_helmet = (False, conf)

        # 2. Vest check
        if raw_class in VEST_POSITIVE_CLASSES:
            if best_vest is None or conf > best_vest[1]:
                best_vest = (True, conf)
        elif raw_class in VEST_NEGATIVE_CLASSES:
            if best_vest is None or conf > best_vest[1]:
                best_vest = (False, conf)

        # 3. Boots check
        if raw_class in BOOTS_POSITIVE_CLASSES:
            if best_boots is None or conf > best_boots[1]:
                best_boots = (True, conf)
        elif raw_class in BOOTS_NEGATIVE_CLASSES:
            if best_boots is None or conf > best_boots[1]:
                best_boots = (False, conf)

    # Convert best matches to final statuses
    helmet_status = best_helmet[0] if best_helmet is not None else "unknown"
    vest_status = best_vest[0] if best_vest is not None else "unknown"
    boots_status = best_boots[0] if best_boots is not None else "unknown"

    return PPEResult(
        helmet=helmet_status,
        vest=vest_status,
        boots=boots_status,
        raw_detections=raw_list,
    )


# ---------------------------------------------------------------------------
# Core PPEDetector Class
# ---------------------------------------------------------------------------
class PPEDetector:
    """Manages PPE detection for tracked workers using Roboflow and local caching.

    Key Features:
        - Bounding Box Cropping: Crops only the tracked worker region to save bandwidth.
        - Worker-Level Caching: Caches PPE status per persistent worker ID.
        - Periodic Refresh: Only calls inference once every `inference_interval_seconds`
          per worker (default 1.0s), preventing excessive API calls.
        - Graceful Fallback: Returns 'unknown' on any network, authentication, or API error.

    Args:
        api_key: Roboflow API key. If None, reads from ROBOFLOW_API_KEY environment variable.
        model_id: Roboflow model identifier (default: 'helmet-vest-and-boots-detection/8').
        api_url: Roboflow inference API URL (default: 'https://serverless.roboflow.com').
        inference_interval_seconds: Minimum time between inference calls per worker.
        confidence_threshold: Minimum confidence score to register a PPE class.
        enabled: Set to False to disable PPE calls and return 'unknown' for all items.
        max_cache_age_seconds: Time after which inactive worker cache entries are purged.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = DEFAULT_ROBOFLOW_MODEL_ID,
        api_url: str = DEFAULT_ROBOFLOW_API_URL,
        inference_interval_seconds: float = DEFAULT_PPE_INTERVAL_SECONDS,
        confidence_threshold: float = DEFAULT_PPE_CONFIDENCE_THRESHOLD,
        enabled: bool = True,
        max_cache_age_seconds: float = DEFAULT_MAX_CACHE_AGE_SECONDS,
    ):
        self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY", "").strip()
        self.model_id = model_id
        self.api_url = api_url
        self.inference_interval_seconds = max(0.1, float(inference_interval_seconds))
        self.confidence_threshold = float(confidence_threshold)
        self.enabled = bool(enabled)
        self.max_cache_age_seconds = float(max_cache_age_seconds)

        # Internal worker cache: worker_id -> PPEResult
        self._cache: Dict[int, PPEResult] = {}
        # Client instance
        self._client = None
        self._client_initialized = False
        self._has_logged_api_key_warning = False
        self._has_logged_network_warning = False

    def _get_client(self) -> Optional[Any]:
        """Initialise or return the InferenceHTTPClient instance."""
        if not self.enabled:
            return None

        if not self.api_key:
            if not self._has_logged_api_key_warning:
                logger.info(
                    "ROBOFLOW_API_KEY not configured. PPE detection will output 'unknown' for all items.\n"
                    "To enable Roboflow PPE detection, set ROBOFLOW_API_KEY in your .env or environment."
                )
                self._has_logged_api_key_warning = True
            return None

        if not self._client_initialized:
            try:
                from inference_sdk import InferenceHTTPClient
                self._client = InferenceHTTPClient(
                    api_url=self.api_url,
                    api_key=self.api_key,
                )
                self._client_initialized = True
                logger.info("Roboflow PPE Inference client initialised for model '%s'.", self.model_id)
            except Exception as e:
                logger.warning("Could not initialise InferenceHTTPClient: %s. PPE will default to 'unknown'.", e)
                self._client = None
                self._client_initialized = True

        return self._client

    @staticmethod
    def crop_worker(frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
        """Safely crop a worker region from a video frame.

        Args:
            frame: Full video frame as a NumPy array (BGR).
            bbox: Bounding box as [x1, y1, x2, y2].

        Returns:
            Cropped image as a NumPy array, or None if crop is invalid or out-of-bounds.
        """
        if frame is None or len(bbox) < 4:
            return None

        h, w = frame.shape[:2]
        x1 = max(0, min(int(bbox[0]), w - 1))
        y1 = max(0, min(int(bbox[1]), h - 1))
        x2 = max(0, min(int(bbox[2]), w))
        y2 = max(0, min(int(bbox[3]), h))

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w < MIN_CROP_DIMENSION or crop_h < MIN_CROP_DIMENSION:
            return None

        return frame[y1:y2, x1:x2].copy()

    def should_refresh(self, worker_id: int, current_timestamp: float) -> bool:
        """Check if PPE inference should be refreshed for a given worker ID.

        Args:
            worker_id: Persistent track ID of the worker.
            current_timestamp: Current video/session timestamp in seconds.

        Returns:
            True if worker is not yet in cache or if inference interval has elapsed.
        """
        if worker_id not in self._cache:
            return True

        cached_entry = self._cache[worker_id]
        elapsed = current_timestamp - cached_entry.last_updated
        return elapsed >= self.inference_interval_seconds

    def infer_worker_crop(self, crop: np.ndarray) -> PPEResult:
        """Send a cropped worker image to Roboflow for PPE inference.

        Args:
            crop: Cropped worker image (BGR NumPy array).

        Returns:
            PPEResult with parsed helmet, vest, boots status.
        """
        client = self._get_client()
        if client is None or crop is None:
            return PPEResult()

        try:
            # Roboflow inference SDK accepts numpy arrays directly
            response = client.infer(crop, model_id=self.model_id)
            predictions = []

            if isinstance(response, dict):
                predictions = response.get("predictions", [])
            elif isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict):
                predictions = response[0].get("predictions", response)

            result = parse_ppe_predictions(
                predictions=predictions,
                confidence_threshold=self.confidence_threshold,
            )
            return result

        except Exception as e:
            if not self._has_logged_network_warning:
                logger.warning(
                    "Roboflow PPE inference request failed (%s). Defaulting to 'unknown'.\n"
                    "Pipeline will continue running smoothly.", e
                )
                self._has_logged_network_warning = True
            else:
                logger.debug("Roboflow PPE inference failed: %s", e)
            return PPEResult()

    def update_worker(
        self,
        frame: np.ndarray,
        worker_id: int,
        bbox: list,
        timestamp: float,
    ) -> PPEResult:
        """Update and retrieve the current PPE status for a worker.

        If the worker's cached PPE status is fresh, returns the cached result.
        If it is time to refresh, crops the worker region, executes Roboflow
        inference, updates the cache, and returns the new result.

        Args:
            frame: Current video frame (BGR).
            worker_id: Persistent worker track ID.
            bbox: Worker bounding box [x1, y1, x2, y2].
            timestamp: Current frame timestamp in seconds.

        Returns:
            PPEResult representing current PPE compliance status.
        """
        if worker_id < 0:
            return PPEResult(last_updated=timestamp)

        # Check if we should refresh inference
        if self.should_refresh(worker_id, timestamp):
            crop = self.crop_worker(frame, bbox)
            if crop is not None and self._get_client() is not None:
                ppe_result = self.infer_worker_crop(crop)
                ppe_result.last_updated = timestamp
                self._cache[worker_id] = ppe_result
            else:
                # If no client or invalid crop, ensure cache has an entry with current timestamp
                if worker_id not in self._cache:
                    self._cache[worker_id] = PPEResult(last_updated=timestamp)
                else:
                    self._cache[worker_id].last_updated = timestamp

        # Periodic cache cleanup for workers not seen recently
        self._prune_stale_cache(timestamp)

        return self._cache.get(worker_id, PPEResult(last_updated=timestamp))

    def get_worker_ppe(self, worker_id: int) -> PPEResult:
        """Retrieve the cached PPE result for a worker ID.

        Args:
            worker_id: Worker track ID.

        Returns:
            PPEResult (defaults to all 'unknown' if worker not in cache).
        """
        return self._cache.get(worker_id, PPEResult())

    def _prune_stale_cache(self, current_timestamp: float) -> None:
        """Remove cache entries for workers that haven't been updated in max_cache_age_seconds."""
        if len(self._cache) <= 50:
            return  # Small cache, no pruning needed

        stale_ids = [
            wid for wid, res in self._cache.items()
            if (current_timestamp - res.last_updated) > self.max_cache_age_seconds
        ]
        for wid in stale_ids:
            del self._cache[wid]

    def clear_cache(self) -> None:
        """Clear the worker PPE cache."""
        self._cache.clear()
