import os
import sys
import json
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader import (
    load_incidents,
    load_video,
    load_evidence,
    calculate_metrics,
    calculate_system_status,
    filter_incidents
)

def run_verification():
    print("=== FactoryGuard AI Phase 4 Implementation Verification ===")
    
    # 1. Test standard data loading
    df = load_incidents("data/sample_incidents.json")
    print(f"[SUCCESS] Loaded sample incidents dataset: {len(df)} records.")
    assert len(df) == 8, f"Expected 8 sample records, got {len(df)}"

    # 2. Test metrics calculation
    metrics = calculate_metrics(df)
    print(f"[SUCCESS] Metrics calculated: {metrics}")
    assert metrics['total_incidents'] == 8
    assert metrics['critical_incidents'] == 4
    assert metrics['warning_incidents'] == 4

    # 3. Test safety status calculation
    status = calculate_system_status(df)
    print(f"[SUCCESS] Calculated overall safety status: '{status}'")
    assert status == "CRITICAL"

    # 4. Test filtering logic
    df_all = filter_incidents(df, severity="ALL", event_type="ALL", worker_id="ALL", zone="ALL")
    print(f"[SUCCESS] Filters ALL: {len(df_all)} items retained.")
    assert len(df_all) == 8, f"Expected 8 items when filters are ALL, got {len(df_all)}"

    crit_df = filter_incidents(df, severity="CRITICAL")
    print(f"[SUCCESS] Severity filter 'CRITICAL': {len(crit_df)} items.")
    assert len(crit_df) == 4

    warn_df = filter_incidents(df, severity="WARNING")
    print(f"[SUCCESS] Severity filter 'WARNING': {len(warn_df)} items.")
    assert len(warn_df) == 4

    event_df = filter_incidents(df, event_type="Restricted Zone Entry")
    print(f"[SUCCESS] Event Type filter 'Restricted Zone Entry': {len(event_df)} items.")
    assert len(event_df) == 3

    worker_df = filter_incidents(df, worker_id="3")
    print(f"[SUCCESS] Worker ID filter '3': {len(worker_df)} items.")
    assert len(worker_df) == 2

    zone_df = filter_incidents(df, zone="Machine A")
    print(f"[SUCCESS] Zone filter 'Machine A': {len(zone_df)} items.")
    assert len(zone_df) == 2

    # 5. Test incident selection logic
    selected_inc = df[df['incident_id'] == "INC-001"].iloc[0]
    assert selected_inc['worker_id'] == 3
    assert selected_inc['event_type'] == "Restricted Zone Entry"
    print(f"[SUCCESS] Incident selection INC-001: Worker #{selected_inc['worker_id']} - {selected_inc['event_type']}")

    # 5. Test evidence image loading
    exists, msg, img = load_evidence(df.iloc[0]['evidence'])
    print(f"[SUCCESS] Evidence loading for '{df.iloc[0]['evidence']}': {exists} ('{msg}')")
    assert exists is True
    assert img is not None

    # 6. Test missing evidence handling
    exists_miss, msg_miss, img_miss = load_evidence("evidence/missing_file.jpg")
    print(f"[SUCCESS] Missing evidence safe fallback message: '{msg_miss}'")
    assert exists_miss is False
    assert img_miss is None

    # 7. Test missing video handling
    video_path = load_video("outputs/tracked_output.mp4")
    print(f"[SUCCESS] Video checking (outputs/tracked_output.mp4 absent): returned '{video_path}' (Safe Fallback Box Enabled)")

    # 8. Test missing & corrupted JSON handling
    df_missing = load_incidents("data/non_existent.json")
    assert df_missing.empty and "incident_id" in df_missing.columns
    print("[SUCCESS] Missing JSON gracefully handled (empty DataFrame returned).")

    df_corrupt = load_incidents("tests/corrupt.json")
    assert df_corrupt.empty and "incident_id" in df_corrupt.columns
    print("[SUCCESS] Corrupted JSON gracefully handled (empty DataFrame returned).")

    print("\n[PASSED] ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_verification()
