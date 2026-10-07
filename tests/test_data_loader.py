import os
import json
import pytest
import pandas as pd
from data_loader import (
    load_incidents,
    load_video,
    load_evidence,
    calculate_metrics,
    calculate_system_status,
    filter_incidents
)

def test_load_incidents_existing(tmp_path):
    # Test valid JSON load
    test_json = tmp_path / "test_incidents.json"
    data = [
        {
            "incident_id": "INC-999",
            "worker_id": 4,
            "event_type": "Restricted Zone Entry",
            "zone": "Machine A",
            "timestamp": "00:01:00",
            "duration": 5.0,
            "severity": "CRITICAL",
            "evidence": "evidence/test.jpg"
        }
    ]
    test_json.write_text(json.dumps(data), encoding='utf-8')

    df = load_incidents(str(test_json))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]['incident_id'] == "INC-999"
    assert df.iloc[0]['worker_id'] == 4

def test_load_incidents_missing_file():
    df = load_incidents("non_existent_file.json")
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert "incident_id" in df.columns

def test_load_incidents_corrupted_json(tmp_path):
    test_json = tmp_path / "bad.json"
    test_json.write_text("{bad json syntax", encoding='utf-8')
    
    df = load_incidents(str(test_json))
    assert isinstance(df, pd.DataFrame)
    assert df.empty

def test_calculate_metrics():
    df = pd.DataFrame([
        {"worker_id": 1, "severity": "CRITICAL", "zone": "Zone A"},
        {"worker_id": 2, "severity": "WARNING", "zone": "Zone B"},
        {"worker_id": 1, "severity": "CRITICAL", "zone": "Zone A"}
    ])
    metrics = calculate_metrics(df)
    assert metrics['total_workers'] == 2
    assert metrics['total_incidents'] == 3
    assert metrics['critical_incidents'] == 2
    assert metrics['warning_incidents'] == 1
    assert metrics['restricted_zones'] == 2

def test_calculate_system_status():
    df_critical = pd.DataFrame([{"severity": "CRITICAL"}])
    assert calculate_system_status(df_critical) == "CRITICAL"

    df_warning = pd.DataFrame([{"severity": "WARNING"}])
    assert calculate_system_status(df_warning) == "WARNING"

    df_empty = pd.DataFrame()
    assert calculate_system_status(df_empty) == "SAFE"

def test_filter_incidents():
    df = pd.DataFrame([
        {"incident_id": "INC-001", "worker_id": 1, "severity": "CRITICAL", "event_type": "Entry", "zone": "Zone A"},
        {"incident_id": "INC-002", "worker_id": 2, "severity": "WARNING", "event_type": "Presence", "zone": "Zone B"}
    ])

    # All filters set to ALL / All / all
    filtered_all = filter_incidents(df, severity="ALL", event_type="ALL", worker_id="ALL", zone="ALL")
    assert len(filtered_all) == 2

    filtered_all_mixed = filter_incidents(df, severity="All", event_type="all", worker_id="ALL", zone="All")
    assert len(filtered_all_mixed) == 2

    filtered_crit = filter_incidents(df, severity="CRITICAL")
    assert len(filtered_crit) == 1
    assert filtered_crit.iloc[0]['incident_id'] == "INC-001"

    filtered_worker = filter_incidents(df, worker_id="2")
    assert len(filtered_worker) == 1
    assert filtered_worker.iloc[0]['incident_id'] == "INC-002"

def test_load_video_missing():
    path = load_video("non_existent_video.mp4")
    assert path is None

def test_load_evidence_missing():
    exists, msg, img = load_evidence("non_existent_image.jpg")
    assert exists is False
    assert img is None
    assert "missing" in msg.lower()
