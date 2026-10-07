"""
FactoryGuard AI - PPE Detection Module (Roboflow Integration)
=============================================================
Inspects tracked worker bounding boxes for required Personal Protective
Equipment (PPE):
  - Helmet / Hardhat (True / False / "unknown")
  - Safety Vest (True / False / "unknown")
  - Safety Boots (True / False / "unknown")

Features:
  - Padded worker cropping (15% W, 20% Top, 10% Bot) to prevent cutting off PPE
  - Support for both positive ('helmet', 'vest', 'boots') and negative ('no helmet', 'no vest') labels
  - Worker-level temporal smoothing & hysteresis to prevent single-frame flickering
  - Periodic inference interval (default: 1.0s per tracked worker)
  - Zero-crash fallback when API is unconfigured or unreachable
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import numpy as np

logger = logging.getLogger("factoryguard.ppe")


def load_env_file(env_path: Optional[str] = None) -> None:
    """
    Safely load key-value pairs from a .env file into os.environ.
    Does not overwrite existing environment variables.
    """
    if env_path is None:
        possible_paths = [
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        ]
        for p in possible_paths:
            if os.path.isfile(p):
                env_path = p
                break

    if not env_path or not os.path.isfile(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = value
    except Exception as exc:
        logger.warning(f"Could not parse .env file at {env_path}: {exc}")


class PPEResult:
    """Structured container for PPE detection output."""

    __slots__ = (
        "worker_id",
        "helmet",
        "vest",
        "boots",
        "helmet_conf",
        "vest_conf",
        "boots_conf",
        "timestamp",
        "raw_detections",
    )

    def __init__(
        self,
        worker_id: Any,
        helmet: Any = "unknown",
        vest: Any = "unknown",
        boots: Any = "unknown",
        helmet_conf: float = 0.0,
        vest_conf: float = 0.0,
        boots_conf: float = 0.0,
        timestamp: float = 0.0,
        raw_detections: Optional[List[Dict[str, Any]]] = None,
    ):
        self.worker_id = worker_id
        self.helmet = helmet
        self.vest = vest
        self.boots = boots
        self.helmet_conf = helmet_conf
        self.vest_conf = vest_conf
        self.boots_conf = boots_conf
        self.timestamp = timestamp or time.time()
        self.raw_detections = raw_detections or []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to standard Phase 1 / Phase 2 PPE dictionary contract."""
        return {
            "helmet": self.helmet,
            "vest": self.vest,
            "boots": self.boots,
        }

    def __repr__(self) -> str:
        return (
            f"PPEResult(worker={self.worker_id}, "
            f"H={self.helmet}, V={self.vest}, B={self.boots})"
        )


class WorkerPPEHistory:
    """
    Sliding window temporal filter with hysteresis for a single tracked worker.
    Prevents single-frame flicker between True, False, and 'unknown'.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.helmet_votes: deque = deque(maxlen=window_size)
        self.vest_votes: deque = deque(maxlen=window_size)
        self.boots_votes: deque = deque(maxlen=window_size)

        self.current_helmet: Any = "unknown"
        self.current_vest: Any = "unknown"
        self.current_boots: Any = "unknown"

    def update(
        self,
        helmet_obs: Optional[bool],
        vest_obs: Optional[bool],
        boots_obs: Optional[bool],
    ) -> Tuple[Any, Any, Any]:
        """
        Add a fresh observation and calculate the smoothed, hysteresis-stable state.
        None indicates no detection (neutral/insufficient evidence in current frame).
        """
        if helmet_obs is not None:
            self.helmet_votes.append(helmet_obs)
        if vest_obs is not None:
            self.vest_votes.append(vest_obs)
        if boots_obs is not None:
            self.boots_votes.append(boots_obs)

        self.current_helmet = self._evaluate_votes(self.helmet_votes, self.current_helmet)
        self.current_vest = self._evaluate_votes(self.vest_votes, self.current_vest)
        self.current_boots = self._evaluate_votes(self.boots_votes, self.current_boots)

        return self.current_helmet, self.current_vest, self.current_boots

    def _evaluate_votes(self, votes: deque, current_state: Any) -> Any:
        if not votes:
            return current_state

        trues = sum(1 for v in votes if v is True)
        falses = sum(1 for v in votes if v is False)
        total = len(votes)

        # Hysteresis rules:
        # 1. To transition into True, need at least 2 True votes (or 1 if total==1) and trues > falses
        if trues >= 2 and trues > falses:
            return True
        elif trues == 1 and total == 1 and falses == 0:
            return True

        # 2. To transition into False, need at least 2 False votes (or 1 if total==1) and falses > trues
        if falses >= 2 and falses > trues:
            return False
        elif falses == 1 and total == 1 and trues == 0:
            return False

        # 3. If votes are conflicting or insufficient, retain established state
        return current_state


def crop_worker_image(
    frame: np.ndarray,
    bbox: List[float],
    padding_w: float = 0.15,
    padding_top: float = 0.20,
    padding_bot: float = 0.10,
) -> Optional[np.ndarray]:
    """
    Safely crop worker bounding box with context padding clamped to image boundaries.
    Top padding is enlarged (20%) to capture helmets; width is padded (15%) for vests.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    if len(bbox) < 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    # Apply directional padding
    pad_x = bw * padding_w
    pad_y1 = bh * padding_top
    pad_y2 = bh * padding_bot

    px1 = int(max(0, round(x1 - pad_x)))
    py1 = int(max(0, round(y1 - pad_y1)))
    px2 = int(min(w, round(x2 + pad_x)))
    py2 = int(min(h, round(y2 + pad_y2)))

    if px2 <= px1 or py2 <= py1:
        return None

    crop = frame[py1:py2, px1:px2]
    return crop if crop.size > 0 else None


def parse_ppe_predictions(
    predictions: List[Dict[str, Any]],
    confidence_threshold: float = 0.35,
) -> Dict[str, Tuple[Optional[bool], float]]:
    """
    Parse raw Roboflow predictions.
    Supports both positive labels ('helmet', 'vest', 'boots')
    and negative labels ('no helmet', 'no vest', 'no boots').
    Returns {item: (observation_bool_or_none, max_confidence)}.
    """
    helmet_obs: Optional[bool] = None
    vest_obs: Optional[bool] = None
    boots_obs: Optional[bool] = None

    best_h_conf = 0.0
    best_v_conf = 0.0
    best_b_conf = 0.0

    for pred in predictions:
        cls_name = str(pred.get("class", "")).lower().strip()
        conf = float(pred.get("confidence", 0.0))

        if conf < confidence_threshold:
            continue

        # Helmet
        if any(term in cls_name for term in ["hard hat", "hardhat", "safety helmet", "yellow helmet", "white helmet"]) or cls_name == "helmet":
            if conf > best_h_conf:
                helmet_obs = True
                best_h_conf = conf
        elif any(term in cls_name for term in ["no helmet", "no-helmet", "no_helmet", "head"]):
            if conf > best_h_conf:
                helmet_obs = False
                best_h_conf = conf

        # Vest
        if any(term in cls_name for term in ["safety vest", "reflective vest", "hi-vis", "jacket"]) or cls_name == "vest":
            if conf > best_v_conf:
                vest_obs = True
                best_v_conf = conf
        elif any(term in cls_name for term in ["no vest", "no-vest", "no_vest"]):
            if conf > best_v_conf:
                vest_obs = False
                best_v_conf = conf

        # Boots
        if any(term in cls_name for term in ["safety boots", "steel toe", "shoes"]) or cls_name == "boots":
            if conf > best_b_conf:
                boots_obs = True
                best_b_conf = conf
        elif any(term in cls_name for term in ["no boots", "no-boots", "no_boots"]):
            if conf > best_b_conf:
                boots_obs = False
                best_b_conf = conf

    return {
        "helmet": (helmet_obs, best_h_conf),
        "vest": (vest_obs, best_v_conf),
        "boots": (boots_obs, best_b_conf),
    }


class PPEDetector:
    """
    Manages Roboflow PPE inference, worker bounding box cropping,
    caching, interval gating, and temporal smoothing per worker.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "helmet-vest-and-boots-detection/8",
        api_url: str = "https://detect.roboflow.com",
        confidence_threshold: float = 0.35,
        inference_interval_seconds: float = 1.0,
        max_cache_age_seconds: float = 30.0,
        temporal_window: int = 5,
        enabled: bool = True,
    ):
        load_env_file()
        self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY", "")
        self.model_id = model_id
        self.api_url = api_url
        self.confidence_threshold = confidence_threshold
        self.inference_interval_seconds = inference_interval_seconds
        self.max_cache_age_seconds = max_cache_age_seconds
        self.temporal_window = temporal_window
        self.enabled = enabled

        self._cache: Dict[Any, PPEResult] = {}
        self._histories: Dict[Any, WorkerPPEHistory] = {}
        self._client: Optional[Any] = None
        self._client_initialized: bool = False
        self._has_logged_api_key_warning: bool = False
        self._has_logged_network_warning: bool = False

    @property
    def is_available(self) -> bool:
        """Returns True if API key is configured and detector is enabled."""
        return bool(self.enabled and self.api_key)

    def _get_client(self) -> Optional[Any]:
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

        if self._client_initialized:
            return self._client

        try:
            from inference_sdk import InferenceHTTPClient
            self._client = InferenceHTTPClient(
                api_url=self.api_url,
                api_key=self.api_key,
            )
            self._client_initialized = True
            logger.info(
                f"Roboflow PPE Inference client initialised for model '{self.model_id}'."
            )
            return self._client
        except Exception as exc:
            logger.warning(f"Failed to initialize Roboflow InferenceHTTPClient: {exc}")
            self._client_initialized = True
            self._client = None
            return None

    def should_refresh(self, worker_id: Any, current_timestamp: float) -> bool:
        """Check if interval has passed since last Roboflow inference for worker."""
        if worker_id not in self._cache:
            return True
        last_result = self._cache[worker_id]
        return (current_timestamp - last_result.timestamp) >= self.inference_interval_seconds

    def infer_worker_crop(
        self,
        crop: np.ndarray,
        worker_id: Any,
        timestamp: float,
    ) -> PPEResult:
        """Execute Roboflow inference on a single padded worker crop."""
        client = self._get_client()
        if client is None or crop is None or crop.size == 0:
            return PPEResult(
                worker_id=worker_id,
                helmet="unknown",
                vest="unknown",
                boots="unknown",
                timestamp=timestamp,
            )

        try:
            raw_response = client.infer(crop, model_id=self.model_id)
            preds = raw_response.get("predictions", []) if isinstance(raw_response, dict) else []

            obs = parse_ppe_predictions(preds, self.confidence_threshold)
            h_obs, h_conf = obs["helmet"]
            v_obs, v_conf = obs["vest"]
            b_obs, b_conf = obs["boots"]

            # Update temporal sliding window
            if worker_id not in self._histories:
                self._histories[worker_id] = WorkerPPEHistory(window_size=self.temporal_window)

            s_helmet, s_vest, s_boots = self._histories[worker_id].update(h_obs, v_obs, b_obs)

            return PPEResult(
                worker_id=worker_id,
                helmet=s_helmet,
                vest=s_vest,
                boots=s_boots,
                helmet_conf=h_conf,
                vest_conf=v_conf,
                boots_conf=b_conf,
                timestamp=timestamp,
                raw_detections=preds,
            )

        except Exception as exc:
            if not self._has_logged_network_warning:
                logger.warning(
                    f"Roboflow PPE inference request failed ({exc}). Defaulting to 'unknown'.\n"
                    "Pipeline will continue running smoothly."
                )
                self._has_logged_network_warning = True

            # Return existing history/cache if available, otherwise unknown
            cached = self._cache.get(worker_id)
            if cached:
                return PPEResult(
                    worker_id=worker_id,
                    helmet=cached.helmet,
                    vest=cached.vest,
                    boots=cached.boots,
                    timestamp=timestamp,
                )
            return PPEResult(
                worker_id=worker_id,
                helmet="unknown",
                vest="unknown",
                boots="unknown",
                timestamp=timestamp,
            )

    def update_worker(
        self,
        frame: np.ndarray,
        bbox: List[float],
        worker_id: Any,
        timestamp: float,
        force_refresh: bool = False,
    ) -> PPEResult:
        """
        Get or refresh PPE detection for a worker bounding box.
        Respects refresh interval and applies temporal smoothing.
        """
        if not self.enabled:
            return PPEResult(
                worker_id=worker_id,
                helmet="unknown",
                vest="unknown",
                boots="unknown",
                timestamp=timestamp,
            )

        if not force_refresh and not self.should_refresh(worker_id, timestamp):
            if worker_id in self._cache:
                return self._cache[worker_id]

        crop = crop_worker_image(frame, bbox)
        if crop is None:
            if worker_id in self._cache:
                return self._cache[worker_id]
            return PPEResult(
                worker_id=worker_id,
                helmet="unknown",
                vest="unknown",
                boots="unknown",
                timestamp=timestamp,
            )

        result = self.infer_worker_crop(crop, worker_id, timestamp)
        self._cache[worker_id] = result
        self._prune_stale_cache(timestamp)
        return result

    def get_worker_ppe(self, worker_id: Any) -> Dict[str, Any]:
        """Fetch current cached PPE dictionary for a worker ID."""
        if worker_id in self._cache:
            return self._cache[worker_id].to_dict()
        return {
            "helmet": "unknown",
            "vest": "unknown",
            "boots": "unknown",
        }

    def _prune_stale_cache(self, current_timestamp: float) -> None:
        """Remove cached entries for workers not seen for > max_cache_age_seconds."""
        stale_keys = [
            wid
            for wid, res in self._cache.items()
            if (current_timestamp - res.timestamp) > self.max_cache_age_seconds
        ]
        for wid in stale_keys:
            self._cache.pop(wid, None)
            self._histories.pop(wid, None)

    def clear_cache(self) -> None:
        """Reset all cached worker predictions and history."""
        self._cache.clearDistributed = {}
        self._cache.clear()
        self._histories.clear()
