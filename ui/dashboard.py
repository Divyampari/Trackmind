import streamlit as st
import pandas as pd
from data_loader import (
    load_incidents, 
    calculate_metrics, 
    calculate_system_status, 
    filter_incidents
)
from ui.components import (
    render_header,
    render_kpi_cards,
    render_safety_status,
    render_source_selection,
    render_filters,
    render_video_monitoring,
    render_recent_alerts,
    render_incident_table,
    render_incident_detail_story,
    render_analytics,
    render_zone_visualization,
    render_worker_summary
)

def render_dashboard():
    """
    Main layout renderer for Phase 4 FactoryGuard AI Dashboard.
    Coordinates data loading, state management, filtering, and UI rendering.
    """
    # 1. Header & Branding
    render_header()

    # 2. Data Loading & Error Handling
    df_raw = load_incidents()

    # 3. Sidebar Source Selector & Filters
    selected_source_mode = render_source_selection()
    filters = render_filters(df_raw)

    # Apply filters dynamically
    df_filtered = filter_incidents(
        df_raw,
        severity=filters['severity'],
        event_type=filters['event_type'],
        worker_id=filters['worker_id'],
        zone=filters['zone'],
        source=filters['source']
    )

    # 4. Metrics & Safety Status
    overall_metrics = calculate_metrics(df_raw)
    current_status = calculate_system_status(df_raw)

    # 5. System Overview KPI Cards & Safety Status Banner
    render_kpi_cards(overall_metrics)
    render_safety_status(current_status, overall_metrics)

    st.markdown("<br>", unsafe_allow_html=True)

    # 6. Tabbed View Navigation for Clean UX
    tab_overview, tab_incidents, tab_analytics, tab_zones = st.tabs([
        "👁️ Live Feed & Monitoring", 
        "📋 Incident Story & Evidence", 
        "📊 Safety Analytics", 
        "📍 Zones & Worker Summary"
    ])

    with tab_overview:
        col_video, col_table = st.columns([1.1, 0.9])
        with col_video:
            render_video_monitoring()
        with col_table:
            st.markdown("#### 🚨 Filtered Incident Feed")
            render_incident_table(df_filtered)

    with tab_incidents:
        render_incident_detail_story(df_filtered)
        st.markdown("---")
        render_incident_table(df_filtered)

    with tab_analytics:
        render_analytics(df_filtered)

    with tab_zones:
        col_z, col_w = st.columns(2)
        with col_z:
            render_zone_visualization(df_raw)
        with col_w:
            render_worker_summary(df_raw)
