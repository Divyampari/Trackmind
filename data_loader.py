import json
import os
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

CANDIDATE_DATA_PATHS = [
    "data/incidents/incidents.json",
    "data/incidents.json",
    "data/sample_incidents.json",
    "data/tracking/incidents.json",
    "incidents.json"
]

CANDIDATE_VIDEO_PATHS = [
    "outputs/tracked_output.mp4",
    "videos/tracked_output.mp4",
    "outputs/webcam_output.mp4"
]

REQUIRED_FIELDS = [
    "incident_id", "worker_id", "event_type", 
    "zone", "timestamp", "duration", "severity", "evidence"
]

def load_incidents(json_path: Optional[str] = None) -> pd.DataFrame:
    """
    Safely loads incident data from JSON file into a pandas DataFrame.
    Searches multiple candidate paths if no specific path is given.
    Gracefully handles missing, empty, or corrupted JSON files.
    """
    target_paths = [json_path] if json_path else CANDIDATE_DATA_PATHS

    resolved_path = None
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for path in target_paths:
        if not path:
            continue
        if os.path.exists(path):
            resolved_path = path
            break
        alt_path = os.path.join(base_dir, path)
        if os.path.exists(alt_path):
            resolved_path = alt_path
            break

    if not resolved_path:
        return pd.DataFrame(columns=REQUIRED_FIELDS + ["source", "ppe"])

    try:
        with open(resolved_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame(columns=REQUIRED_FIELDS + ["source", "ppe"])

        df = pd.DataFrame(data)

        # Fallback field mappings for Phase 3 contract
        if "evidence" not in df.columns or df["evidence"].isnull().all():
            if "evidence_frame" in df.columns:
                df["evidence"] = df["evidence_frame"]

        if "ppe" not in df.columns or df["ppe"].isnull().all():
            if "ppe_status" in df.columns:
                df["ppe"] = df["ppe_status"]

        # Ensure all required fields exist
        for col in REQUIRED_FIELDS:
            if col not in df.columns:
                df[col] = None

        if "source" not in df.columns:
            df["source"] = "Uploaded Video"
        else:
            df["source"] = df["source"].fillna("Uploaded Video")

        if "ppe" not in df.columns:
            df["ppe"] = None

        # Clean & Normalize types
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0.0)
        df['worker_id'] = pd.to_numeric(df['worker_id'], errors='coerce').fillna(0).astype(int)

        # Normalize severity: HIGH -> CRITICAL, MEDIUM -> WARNING, LOW -> INFO
        def norm_sev(val: Any) -> str:
            s = str(val).strip().upper()
            if s in ("HIGH", "CRITICAL"):
                return "CRITICAL"
            elif s in ("MEDIUM", "WARNING", "MODERATE"):
                return "WARNING"
            elif s in ("LOW", "INFO"):
                return "INFO"
            return s if s else "WARNING"

        df['severity'] = df['severity'].apply(norm_sev)

        # Format event_type (replace underscores with spaces and titlecase)
        def norm_evt(val: Any) -> str:
            s = str(val).strip()
            if "_" in s:
                return s.replace("_", " ").title()
            return s if s else "Safety Violation"

        df['event_type'] = df['event_type'].apply(norm_evt)
        df['zone'] = df['zone'].astype(str).str.strip().replace("None", "General Floor")
        df['source'] = df['source'].astype(str).str.strip().replace({"video": "Uploaded Video", "webcam": "Webcam"})

        return df

    except (json.JSONDecodeError, OSError, TypeError) as e:
        print(f"[FactoryGuard Data Error] Failed to read {resolved_path}: {e}")
        return pd.DataFrame(columns=REQUIRED_FIELDS + ["source", "ppe"])


def load_video(video_path: Optional[str] = None) -> Optional[str]:
    """
    Checks if output video file exists and returns path, or None if missing.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if video_path is not None:
        if os.path.exists(video_path):
            return video_path
        alt_path = os.path.join(base_dir, video_path)
        if os.path.exists(alt_path):
            return alt_path
        return None

    # Default candidate paths when no specific path is given
    for path in CANDIDATE_VIDEO_PATHS:
        if os.path.exists(path):
            return path
        alt_path = os.path.join(base_dir, path)
        if os.path.exists(alt_path):
            return alt_path

    outputs_dir = os.path.join(base_dir, "outputs")
    if os.path.exists(outputs_dir):
        for f in os.listdir(outputs_dir):
            if f.endswith(".mp4"):
                return os.path.join(outputs_dir, f)

    return None


def load_evidence(evidence_relative_path: str) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Safely loads an evidence image file given a relative or full file path.
    Returns (exists: bool, status_message: str, image_obj: Optional[Image]).
    """
    if not evidence_relative_path or pd.isna(evidence_relative_path):
        return False, "No evidence image specified for this incident record.", None

    normalized_path = os.path.normpath(str(evidence_relative_path))
    
    if not os.path.exists(normalized_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.normpath(os.path.join(base_dir, str(evidence_relative_path)))
        if os.path.exists(alt_path):
            normalized_path = alt_path
        else:
            # Check inside data/evidence/ or evidence/
            basename = os.path.basename(normalized_path)
            for subfolder in ["data/evidence", "evidence"]:
                check_path = os.path.join(base_dir, subfolder, basename)
                if os.path.exists(check_path):
                    normalized_path = check_path
                    break
            else:
                return False, f"Evidence image missing: '{normalized_path}' was not found.", None

    try:
        img = Image.open(normalized_path)
        return True, "Evidence loaded successfully", img
    except Exception as e:
        return False, f"Failed to read image '{normalized_path}': {str(e)}", None


def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates summary KPIs from the loaded incident dataset.
    """
    if df.empty:
        return {
            "total_workers": 0,
            "total_incidents": 0,
            "critical_incidents": 0,
            "warning_incidents": 0,
            "active_alerts": 0,
            "restricted_zones": 0
        }

    total_workers = int(df['worker_id'].nunique()) if 'worker_id' in df.columns else 0
    total_incidents = len(df)
    critical_incidents = int((df['severity'] == 'CRITICAL').sum())
    warning_incidents = int((df['severity'] == 'WARNING').sum())
    active_alerts = critical_incidents + warning_incidents
    restricted_zones = int(df['zone'].nunique()) if 'zone' in df.columns else 0

    return {
        "total_workers": total_workers,
        "total_incidents": total_incidents,
        "critical_incidents": critical_incidents,
        "warning_incidents": warning_incidents,
        "active_alerts": active_alerts,
        "restricted_zones": restricted_zones
    }


def calculate_system_status(df: pd.DataFrame) -> str:
    """
    Calculates overall system safety status: 'SAFE', 'WARNING', or 'CRITICAL'.
    """
    if df.empty:
        return "SAFE"

    metrics = calculate_metrics(df)
    if metrics["critical_incidents"] > 0:
        return "CRITICAL"
    elif metrics["warning_incidents"] > 0:
        return "WARNING"
    return "SAFE"


def filter_incidents(
    df: pd.DataFrame, 
    severity: Optional[str] = None, 
    event_type: Optional[str] = None, 
    worker_id: Optional[Any] = None, 
    zone: Optional[str] = None,
    source: Optional[str] = None
) -> pd.DataFrame:
    """
    Filters incidents DataFrame dynamically based on user selection.
    Safely matches 'ALL' (case-insensitive) to retain all records without data loss.
    """
    if df.empty:
        return df

    filtered_df = df.copy()

    def is_all(val: Any) -> bool:
        if val is None:
            return True
        s = str(val).strip().upper()
        return s in ("", "ALL", "NONE")

    # 1. Severity Filter
    if not is_all(severity):
        target_sev = str(severity).strip().upper()
        filtered_df = filtered_df[filtered_df['severity'].astype(str).str.upper() == target_sev]

    # 2. Event Type Filter
    if not is_all(event_type):
        target_event = str(event_type).strip()
        filtered_df = filtered_df[filtered_df['event_type'].astype(str).str.strip() == target_event]

    # 3. Worker ID Filter
    if not is_all(worker_id):
        target_worker = str(worker_id).strip()
        filtered_df = filtered_df[filtered_df['worker_id'].astype(str).str.strip() == target_worker]

    # 4. Zone Filter
    if not is_all(zone):
        target_zone = str(zone).strip()
        filtered_df = filtered_df[filtered_df['zone'].astype(str).str.strip() == target_zone]

    # 5. Source Filter
    if not is_all(source) and 'source' in filtered_df.columns:
        target_source = str(source).strip()
        filtered_df = filtered_df[filtered_df['source'].astype(str).str.strip().str.upper() == target_source.upper()]

    return filtered_df


def load_zones() -> List[Dict[str, Any]]:
    """
    Safely loads Phase 2 zone definitions if present.
    """
    candidate_zone_paths = [
        "data/zones/zones.json",
        "data/zones.json",
        "zones.json"
    ]
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for path in candidate_zone_paths:
        full_path = path if os.path.exists(path) else os.path.join(base_dir, path)
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "zones" in data and isinstance(data["zones"], list):
                    return data["zones"]
            except Exception:
                pass
    return []


def parse_ppe_status(ppe_data: Any) -> Dict[str, str]:
    """
    Parses PPE status from incident record.
    Returns dict with keys: 'helmet', 'vest', 'boots'.
    Strict rule: Never convert UNKNOWN into NO. Never infer status.
    """
    default_status = {"helmet": "UNKNOWN", "vest": "UNKNOWN", "boots": "UNKNOWN"}

    if not ppe_data or pd.isna(ppe_data):
        return default_status

    if isinstance(ppe_data, dict):
        return {
            "helmet": str(ppe_data.get("helmet", "UNKNOWN")).upper(),
            "vest": str(ppe_data.get("vest", "UNKNOWN")).upper(),
            "boots": str(ppe_data.get("boots", "UNKNOWN")).upper()
        }

    if isinstance(ppe_data, str):
        # Format like "Helmet: YES, Vest: NO"
        s = ppe_data.upper()
        for k in default_status.keys():
            if f"{k.upper()}: YES" in s or f"{k.upper()}=YES" in s:
                default_status[k] = "YES"
            elif f"{k.upper()}: NO" in s or f"{k.upper()}=NO" in s:
                default_status[k] = "NO"
            elif f"{k.upper()}: UNKNOWN" in s:
                default_status[k] = "UNKNOWN"

    return default_status
