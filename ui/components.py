import os
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any
from data_loader import load_evidence, load_video, parse_ppe_status, load_zones

def render_header():
    """Renders the top branding header bar with live status indicator."""
    st.markdown("""
    <div class="fg-header-card">
        <div>
            <h1 class="fg-title">🛡️ FactoryGuard AI</h1>
            <div class="fg-subtitle">Autonomous Factory Safety Monitoring — HackNex 2026 — PS07</div>
        </div>
        <div class="fg-status-pill">
            <span class="pulse-dot"></span>
            SYSTEM STATUS: MONITORING
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_cards(metrics: dict):
    """Renders top metric KPI cards."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Workers Detected</div>
            <div class="kpi-value">{metrics['total_workers']}</div>
            <div class="kpi-subtext">Active Monitored Personnel</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Incidents</div>
            <div class="kpi-value">{metrics['total_incidents']}</div>
            <div class="kpi-subtext">Logged Safety Events</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Critical Incidents</div>
            <div class="kpi-value" style="color: #ef4444;">{metrics['critical_incidents']}</div>
            <div class="kpi-subtext">Immediate Action Needed</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Warnings</div>
            <div class="kpi-value" style="color: #f59e0b;">{metrics['warning_incidents']}</div>
            <div class="kpi-subtext">Dwell / Mild Breaches</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Active Incidents</div>
            <div class="kpi-value" style="color: #38bdf8;">{metrics['active_alerts']}</div>
            <div class="kpi-subtext">Critical & Warning Total</div>
        </div>
        """, unsafe_allow_html=True)


def render_safety_status(status: str, metrics: dict):
    """Renders the dynamic Overall Safety Status banner."""
    if status == "CRITICAL":
        st.markdown(f"""
        <div class="status-banner-critical">
            <div style="font-size: 1.1rem; font-weight: 800; color: #f87171; text-transform: uppercase;">
                ⚠️ OVERALL FACTORY SAFETY STATUS: CRITICAL HAZARD DETECTED
            </div>
            <div style="color: #cbd5e1; font-size: 0.88rem; margin-top: 4px;">
                {metrics['critical_incidents']} critical safety violations currently logged. Immediate floor safety supervisor inspection required.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif status == "WARNING":
        st.markdown(f"""
        <div class="status-banner-warning">
            <div style="font-size: 1.1rem; font-weight: 800; color: #fbbf24; text-transform: uppercase;">
                ⚡ OVERALL FACTORY SAFETY STATUS: WARNING
            </div>
            <div style="color: #cbd5e1; font-size: 0.88rem; margin-top: 4px;">
                Elevated risk detected. {metrics['warning_incidents']} warning level incidents reported (e.g. prolonged presence / PPE violations).
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-banner-safe">
            <div style="font-size: 1.1rem; font-weight: 800; color: #34d399; text-transform: uppercase;">
                ✅ OVERALL FACTORY SAFETY STATUS: SAFE
            </div>
            <div style="color: #cbd5e1; font-size: 0.88rem; margin-top: 4px;">
                All monitored perimeters operating within safe threshold parameters. No active critical safety breaches.
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_source_selection():
    """Renders input source selector instructions and refresh controls."""
    st.sidebar.markdown("### 🎥 Input Pipeline Source")
    selected_source_mode = st.sidebar.radio(
        "Active Input Source Mode:",
        ["Uploaded Video", "Webcam"],
        index=0
    )
    
    st.sidebar.caption("Note: Run `python app.py` to trigger live pipeline processing for webcam/video input.")
    
    if st.sidebar.button("🔄 Refresh Dashboard", key="refresh_dashboard_btn"):
        st.rerun()

    return selected_source_mode


def render_filters(df: pd.DataFrame):
    """
    Renders filter controls in sidebar and returns selected filter options.
    Always provides an 'ALL' default option for every filter.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Incident Filters")

    if df.empty:
        return {"severity": "ALL", "event_type": "ALL", "worker_id": "ALL", "zone": "ALL", "source": "ALL"}

    # 1. Severity Filter
    severities = ["ALL"] + sorted([str(s).strip().upper() for s in df['severity'].dropna().unique() if str(s).strip()])
    selected_severity = st.sidebar.selectbox("Filter by Severity", severities, index=0)

    # 2. Worker ID Filter
    worker_values = sorted([str(w).strip() for w in df['worker_id'].dropna().unique() if str(w).strip()], key=lambda x: int(x) if x.isdigit() else x)
    workers = ["ALL"] + worker_values
    selected_worker = st.sidebar.selectbox("Filter by Worker ID", workers, index=0)

    # 3. Event Type Filter
    event_types = ["ALL"] + sorted([str(e).strip() for e in df['event_type'].dropna().unique() if str(e).strip()])
    selected_event_type = st.sidebar.selectbox("Filter by Event Type", event_types, index=0)

    # 4. Source Filter
    source_options = ["ALL"]
    if "source" in df.columns:
        source_options += sorted([str(src).strip() for src in df['source'].dropna().unique() if str(src).strip()])
    selected_source = st.sidebar.selectbox("Filter by Source", source_options, index=0)

    st.sidebar.markdown("---")
    st.sidebar.caption("FactoryGuard AI • Phase 4 Dashboard")

    return {
        "severity": selected_severity,
        "event_type": selected_event_type,
        "worker_id": selected_worker,
        "zone": "ALL",
        "source": selected_source
    }


def render_video_monitoring():
    """Renders the video monitoring section with safe fallback message."""
    st.markdown('<div class="section-title">🎥 Live Vision Feed & Video Monitoring</div>', unsafe_allow_html=True)
    video_path = load_video()

    if video_path and os.path.exists(video_path):
        try:
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            st.video(video_bytes, format="video/mp4")
            st.caption(f"Showing output video feed: `{video_path}`")
        except Exception:
            st.video(video_path)
            st.caption(f"Showing output video feed: `{video_path}`")
    else:
        st.info("📹 No processed video available. Run the FactoryGuard processing pipeline first (`outputs/behaviour_output.mp4` or `outputs/tracked_output.mp4`).")


def render_recent_alerts(df: pd.DataFrame):
    """Renders recent safety alerts highlighting critical incidents."""
    st.markdown('<div class="section-title">🚨 Recent Safety Alerts</div>', unsafe_allow_html=True)

    if df.empty:
        st.success("No recent safety alerts registered.")
        return

    sorted_df = df.sort_values(by=['severity', 'timestamp'], ascending=[True, False]).head(4)

    for _, row in sorted_df.iterrows():
        sev = row['severity']
        card_class = "alert-card-critical" if sev == "CRITICAL" else "alert-card-warning"
        badge_html = f'<span class="badge-critical">CRITICAL</span>' if sev == "CRITICAL" else '<span class="badge-warning">WARNING</span>'

        st.markdown(f"""
        <div class="{card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>{row['incident_id']}</strong> — {row['event_type']} {badge_html}
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8;">
                    ⏱️ {row['timestamp']} | Duration: {row['duration']}s
                </div>
            </div>
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">
                📍 <strong>Zone:</strong> {row['zone']} &nbsp;&nbsp;|&nbsp;&nbsp; 👷 <strong>Worker ID:</strong> #{row['worker_id']}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_incident_table(df: pd.DataFrame):
    """Renders styled incident history data table."""
    st.markdown('<div class="section-title">📋 Safety Incident History Log</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No incident records match the selected filter criteria.")
        return

    cols_to_show = ['incident_id', 'worker_id', 'event_type', 'zone', 'timestamp', 'duration', 'severity', 'source']
    available_cols = [c for c in cols_to_show if c in df.columns]

    display_df = df[available_cols].copy()
    display_df.columns = [c.replace('_', ' ').title() for c in available_cols]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


def render_ppe_display(ppe_raw: Any):
    """
    Renders PPE status (Helmet, Vest, Boots).
    Strict rule: Never convert UNKNOWN into NO. Never infer status.
    """
    ppe = parse_ppe_status(ppe_raw)

    if not ppe_raw or (isinstance(ppe_raw, dict) and len(ppe_raw) == 0):
        st.caption("ℹ️ PPE information unavailable")
        return

    def get_badge(status: str) -> str:
        s = status.upper()
        if s == "YES":
            return '<span style="color: #34d399; font-weight: bold;">YES ✅</span>'
        elif s == "NO":
            return '<span style="color: #f87171; font-weight: bold;">NO ❌</span>'
        else:
            return '<span style="color: #94a3b8; font-weight: bold;">UNKNOWN ❓</span>'

    st.markdown(f"""
    <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 0.85rem; margin-top: 8px;">
        <div style="font-weight: 700; color: #38bdf8; margin-bottom: 6px;">🦺 PPE COMPLIANCE STATUS</div>
        <div style="display: flex; gap: 20px; font-size: 0.9rem;">
            <div>🪖 <strong>Helmet:</strong> {get_badge(ppe['helmet'])}</div>
            <div>🎽 <strong>Vest:</strong> {get_badge(ppe['vest'])}</div>
            <div>🥾 <strong>Boots:</strong> {get_badge(ppe['boots'])}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_incident_detail_story(df: pd.DataFrame):
    """Renders interactive Incident Detail card & visual story."""
    st.markdown('<div class="section-title">🔍 Incident Details & Evidence Inspector</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No incidents recorded yet or matching filter criteria.")
        return

    incident_list = df['incident_id'].tolist()
    
    col_sel, _ = st.columns([1, 1])
    with col_sel:
        selected_id = st.selectbox(
            "Select Incident ID to Inspect:",
            incident_list,
            index=0,
            key="story_incident_selector"
        )

    selected_row = df[df['incident_id'] == selected_id].iloc[0]

    col_story, col_evidence = st.columns([1.1, 0.9])

    with col_story:
        st.markdown(f"### 📖 Safety Story: `{selected_row['incident_id']}`")
        
        sev = selected_row['severity']
        sev_color = "#ef4444" if sev == "CRITICAL" else "#f59e0b"

        st.markdown(f"""
        - 👤 **WHO:** Worker #{selected_row['worker_id']}
        - ⚡ **WHAT:** {selected_row['event_type']}
        - 📍 **WHERE:** {selected_row['zone']}
        - 🕒 **WHEN:** {selected_row['timestamp']}
        - ⏱️ **HOW LONG:** {selected_row['duration']} seconds
        - 🔴 **SEVERITY:** <span style="color: {sev_color}; font-weight: bold;">{sev}</span>
        - 📹 **SOURCE:** {selected_row.get('source', 'Uploaded Video')}
        - 📁 **EVIDENCE FILE:** `{selected_row.get('evidence', 'N/A')}`
        """, unsafe_allow_html=True)

        render_ppe_display(selected_row.get('ppe', None))

    with col_evidence:
        st.markdown("### 📸 Phase 3 Evidence Frame")
        evidence_path = selected_row.get('evidence', '')
        exists, msg, img = load_evidence(evidence_path)

        if exists and img is not None:
            st.image(img, caption=f"Evidence Snapshot: {selected_id} ({selected_row['event_type']})", use_container_width=True)
            st.caption(f"Evidence File: `{evidence_path}`")
        else:
            st.warning("📷 Evidence unavailable")
            st.caption(f"No physical evidence image file found at `{evidence_path}`.")


def render_analytics(df: pd.DataFrame):
    """Renders Plotly analytical charts."""
    st.markdown('<div class="section-title">📊 Safety Analytics & Incident Metrics</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No data available for analytical charts.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Incidents by Event Type")
        event_counts = df['event_type'].value_counts().reset_index()
        event_counts.columns = ['Event Type', 'Count']
        
        fig_event = px.bar(
            event_counts,
            x='Event Type',
            y='Count',
            color='Event Type',
            text='Count',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_event.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f1f5f9'),
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=40),
            height=300
        )
        fig_event.update_traces(textposition='outside')
        st.plotly_chart(fig_event, use_container_width=True)

    with col2:
        st.markdown("##### Incidents by Severity")
        sev_counts = df['severity'].value_counts().reset_index()
        sev_counts.columns = ['Severity', 'Count']
        
        color_map = {'CRITICAL': '#ef4444', 'WARNING': '#f59e0b', 'INFO': '#3b82f6'}

        fig_sev = px.pie(
            sev_counts,
            names='Severity',
            values='Count',
            color='Severity',
            color_discrete_map=color_map,
            hole=0.4
        )
        fig_sev.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f1f5f9'),
            margin=dict(l=20, r=20, t=30, b=40),
            height=300
        )
        st.plotly_chart(fig_sev, use_container_width=True)


def render_zone_visualization(df: pd.DataFrame):
    """
    Renders Phase 2 zone information, polygon visual overlay, 
    and web-native Streamlit interactive polygon zone designer.
    """
    import cv2
    import numpy as np
    from PIL import Image
    from behaviour.zone_manager import ZoneManager, Zone, ZoneType

    st.markdown('<div class="section-title">📍 Monitored Factory Zones & Polygon Designer</div>', unsafe_allow_html=True)
    zones_data = load_zones()

    # If no zones defined yet, auto-initialize defaults
    if not zones_data:
        try:
            zm = ZoneManager()
            zm.generate_default_preset_zones((1280, 720))
            out_path = os.path.join("data", "zones", "zones.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            zm.save_zones(out_path)
            zones_data = load_zones()
        except Exception:
            pass

    # 1. Video Source & Reference Frame Selection
    col_v1, col_v2 = st.columns([0.7, 0.3])
    with col_v1:
        st.markdown("**Reference Video Frame for Polygon Design:**")
    with col_v2:
        show_grid = st.checkbox("📐 Show Pixel Coordinate Grid", value=True)

    video_path = load_video()
    ref_frame = None

    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            # Read at frame 15 for clear visibility
            cap.set(cv2.CAP_PROP_POS_FRAMES, 15)
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if ret and frame is not None:
                ref_frame = frame
            cap.release()

    if ref_frame is None:
        ref_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    h_img, w_img = ref_frame.shape[:2]

    # Draw Pixel Grid Rulers if enabled
    display_img = ref_frame.copy()
    if show_grid:
        grid_step = 100
        for x in range(0, w_img, grid_step):
            cv2.line(display_img, (x, 0), (x, h_img), (60, 60, 60), 1)
            cv2.putText(display_img, f"{x}", (x + 3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        for y in range(0, h_img, grid_step):
            cv2.line(display_img, (0, y), (w_img, y), (60, 60, 60), 1)
            cv2.putText(display_img, f"{y}", (5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Draw current zones with numbered vertex badges on reference frame
    if zones_data:
        for z_idx, z in enumerate(zones_data):
            poly = z.get("polygon", [])
            if isinstance(poly, list) and len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32)
                z_type = str(z.get("type", "warning")).lower()
                # BGR Colors: Restricted=Red, Warning=Amber
                color = (0, 0, 239) if z_type == "restricted" else (0, 215, 255)
                
                overlay = display_img.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.30, display_img, 0.70, 0, display_img)
                cv2.polylines(display_img, [pts], isClosed=True, color=color, thickness=3)

                # Draw Vertex Points & Numbered Labels
                for idx, pt in enumerate(poly):
                    px, py = int(pt[0]), int(pt[1])
                    cv2.circle(display_img, (px, py), 7, (255, 255, 255), -1)
                    cv2.circle(display_img, (px, py), 9, color, 2)
                    label_str = f"V{idx+1}({px},{py})"
                    cv2.putText(display_img, label_str, (px + 10, py + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                    cv2.putText(display_img, label_str, (px + 10, py + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

                cx = int(np.mean([p[0] for p in poly]))
                cy = int(np.mean([p[1] for p in poly]))
                z_name = z.get("name", "Zone")
                cv2.putText(display_img, f"[{z_type.upper()}] {z_name}", (cx - 60, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Convert BGR to RGB for Streamlit image display
    rgb_display = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    st.image(rgb_display, caption=f"Factory Monitored Polygon Safety Zones Overlay ({w_img}x{h_img}px)", use_container_width=True)

    # 2. Web-Native Interactive Zone & Vertex Editor
    st.markdown("#### 📐 Interactive Polygon Zone & Vertex Editor")
    
    existing_names = [z.get("name", f"Zone {i+1}") for i, z in enumerate(zones_data)]
    edit_option = st.selectbox("Select Zone to Edit / Configure:", ["➕ Create New Custom Zone"] + existing_names)

    # Pre-fill values if editing an existing zone
    selected_zone = None
    if edit_option != "➕ Create New Custom Zone":
        for z in zones_data:
            if z.get("name") == edit_option:
                selected_zone = z
                break

    init_name = selected_zone.get("name", f"Safety Zone {len(zones_data) + 1}") if selected_zone else f"Safety Zone {len(zones_data) + 1}"
    init_type = "RESTRICTED" if (selected_zone and str(selected_zone.get("type")).lower() == "restricted") else "WARNING"
    init_dwell = float(selected_zone.get("dwell_threshold", 5.0 if init_type == "RESTRICTED" else 10.0)) if selected_zone else (5.0 if init_type == "RESTRICTED" else 10.0)
    init_desc = selected_zone.get("description", "") if selected_zone else ""
    init_poly = selected_zone.get("polygon", []) if selected_zone else []

    with st.expander("🛠️ Edit Zone Name, Hazard Type & Vertex Coordinates", expanded=True):
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            z_name_input = st.text_input("Zone Identifier Name:", value=init_name)
            z_type_input = st.selectbox("Hazard Classification:", ["RESTRICTED", "WARNING"], index=0 if init_type == "RESTRICTED" else 1)
        with col_meta2:
            z_dwell_input = st.number_input("Dwell Violation Threshold (seconds):", min_value=1.0, max_value=60.0, value=init_dwell)
            z_desc_input = st.text_input("Zone Description:", value=init_desc, placeholder="e.g. Hazardous robotics area")

        st.markdown("**Vertex Coordinates Configuration:**")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            preset_choice = st.selectbox(
                "Quick Vertex Layout Presets:",
                [
                    "Current / Custom Coordinates",
                    "Machine Zone A (Default Left)",
                    "Chemical Storage & Corridor (Default Right)",
                    "Center Hazard Perimeter",
                    "Lower Floor Workstation"
                ]
            )

        if preset_choice == "Machine Zone A (Default Left)":
            current_poly = [[int(w_img * 0.15), int(h_img * 0.25)], [int(w_img * 0.45), int(h_img * 0.25)], [int(w_img * 0.45), int(h_img * 0.70)], [int(w_img * 0.15), int(h_img * 0.70)]]
        elif preset_choice == "Chemical Storage & Corridor (Default Right)":
            current_poly = [[int(w_img * 0.55), int(h_img * 0.30)], [int(w_img * 0.85), int(h_img * 0.30)], [int(w_img * 0.85), int(h_img * 0.80)], [int(w_img * 0.55), int(h_img * 0.80)]]
        elif preset_choice == "Center Hazard Perimeter":
            current_poly = [[int(w_img * 0.35), int(h_img * 0.20)], [int(w_img * 0.65), int(h_img * 0.20)], [int(w_img * 0.65), int(h_img * 0.60)], [int(w_img * 0.35), int(h_img * 0.60)]]
        elif preset_choice == "Lower Floor Workstation":
            current_poly = [[int(w_img * 0.20), int(h_img * 0.60)], [int(w_img * 0.80), int(h_img * 0.60)], [int(w_img * 0.80), int(h_img * 0.90)], [int(w_img * 0.20), int(h_img * 0.90)]]
        else:
            if init_poly and len(init_poly) >= 3:
                current_poly = [[int(p[0]), int(p[1])] for p in init_poly]
            else:
                current_poly = [[100, 100], [400, 100], [400, 400], [100, 400]]

        # Interactive Vertex Coordinate Inputs (V1, V2, V3, V4...)
        v_cols = st.columns(len(current_poly))
        new_poly = []
        for i, pt in enumerate(current_poly):
            with v_cols[i]:
                st.markdown(f"**Vertex V{i+1}**")
                vx = st.number_input(f"V{i+1} X (0-{w_img}):", min_value=0, max_value=w_img, value=int(pt[0]), key=f"vx_{i}_{edit_option}")
                vy = st.number_input(f"V{i+1} Y (0-{h_img}):", min_value=0, max_value=h_img, value=int(pt[1]), key=f"vy_{i}_{edit_option}")
                new_poly.append([float(vx), float(vy)])

        col_act1, col_act2, col_act3 = st.columns([1, 1, 1])
        with col_act1:
            if st.button("💾 Save Polygon Zone", key="btn_save_zone"):
                try:
                    zm = ZoneManager()
                    output_path = os.path.join("data", "zones", "zones.json")
                    if os.path.exists(output_path):
                        zm.load_zones(output_path)

                    zone_obj = Zone(
                        name=z_name_input.strip(),
                        type=z_type_input.lower(),
                        polygon=new_poly,
                        dwell_threshold=float(z_dwell_input),
                        description=z_desc_input.strip()
                    )
                    zm.add_zone(zone_obj)
                    zm.save_zones(output_path)
                    st.success(f"✅ Saved zone '{z_name_input}' ({z_type_input}) to `data/zones/zones.json`!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save zone: {e}")

        with col_act2:
            if selected_zone and st.button("🗑️ Delete Selected Zone", key="btn_delete_zone"):
                try:
                    zm = ZoneManager()
                    output_path = os.path.join("data", "zones", "zones.json")
                    if os.path.exists(output_path):
                        zm.load_zones(output_path)
                    zm.remove_zone(edit_option)
                    zm.save_zones(output_path)
                    st.success(f"Deleted zone '{edit_option}'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete zone: {e}")

        with col_act3:
            if st.button("🔄 Reset Default Presets", key="btn_reset_zones"):
                try:
                    zm = ZoneManager()
                    zm.generate_default_preset_zones((w_img, h_img))
                    output_path = os.path.join("data", "zones", "zones.json")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    zm.save_zones(output_path)
                    st.success("Reset zones to default factory configuration.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to reset default zones: {e}")

    # Desktop OpenCV Window Launcher Option
    with st.expander("🖥️ Desktop OpenCV Interactive Window (Optional Desktop GUI)", expanded=False):
        st.info("If running locally on your desktop machine, you can launch the interactive desktop OpenCV GUI where you can click vertices directly on a window using your mouse.")
        if st.button("🚀 Launch Desktop Interactive OpenCV Zone Designer", key="btn_launch_desktop_opencv"):
            try:
                import subprocess
                subprocess.Popen(["python", "-m", "behaviour.zone_designer", "--video", video_path or "videos/my_test1.mp4"])
                st.success("Launched desktop OpenCV Zone Designer window! Check your taskbar for the window.")
            except Exception as e:
                st.error(f"Could not launch desktop window: {e}")

    # 3. Display Zone Summary Table
    if zones_data:
        zones_df = pd.DataFrame(zones_data)
        if "polygon" in zones_df.columns:
            zones_df["vertex_count"] = zones_df["polygon"].apply(lambda p: len(p) if isinstance(p, list) else 0)
        st.dataframe(
            zones_df[['name', 'type', 'dwell_threshold', 'vertex_count', 'description']] if 'vertex_count' in zones_df.columns else zones_df,
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.info("No zone configuration loaded. Edit vertices above to create polygon safety zones.")


def render_worker_summary(df: pd.DataFrame):
    """Renders worker summary using available tracking/incident records."""
    st.markdown('<div class="section-title">👷 Worker Safety Summary</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No worker tracking data recorded yet.")
        return

    summary = df.groupby('worker_id').agg(
        detection_count=('incident_id', 'count'),
        critical_violations=('severity', lambda x: (x == 'CRITICAL').sum()),
        warning_violations=('severity', lambda x: (x == 'WARNING').sum()),
        last_timestamp=('timestamp', 'last'),
        last_zone=('zone', 'last')
    ).reset_index()

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "worker_id": "Worker ID",
            "detection_count": "Incident Count",
            "critical_violations": "Critical Violations",
            "warning_violations": "Warning Violations",
            "last_timestamp": "Latest Timestamp",
            "last_zone": "Last Monitored Zone"
        }
    )
