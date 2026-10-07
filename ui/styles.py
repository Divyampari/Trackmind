import streamlit as st

def apply_custom_styles():
    """
    Applies custom dark-theme industrial styling to the Streamlit app.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

    /* Global Dark Theme Overrides */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Top Branding Header Card */
    .fg-header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .fg-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.5px;
        margin: 0;
    }

    .fg-subtitle {
        font-size: 0.92rem;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* Pulse Status Indicator */
    .fg-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #10b981;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.5px;
    }

    .pulse-dot {
        width: 10px;
        height: 10px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
        animation: pulse 1.8s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        text-align: left;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }

    .kpi-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 700;
        color: #94a3b8;
        letter-spacing: 0.5px;
    }

    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 4px;
    }

    .kpi-subtext {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Overall Safety Status Banners */
    .status-banner-safe {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.2) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-left: 6px solid #10b981;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }

    .status-banner-warning {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.2) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-left: 6px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }

    .status-banner-critical {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.25) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-left: 6px solid #ef4444;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }

    /* Demo Disclaimer Banner */
    .demo-disclaimer {
        background: rgba(56, 189, 248, 0.08);
        border: 1px dashed rgba(56, 189, 248, 0.4);
        border-radius: 8px;
        padding: 0.65rem 1rem;
        font-size: 0.82rem;
        color: #38bdf8;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Severity Badges */
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.75rem;
    }

    .badge-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.75rem;
    }

    /* Alert Cards */
    .alert-card-critical {
        background: #1e1b2e;
        border-left: 4px solid #ef4444;
        padding: 0.85rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }

    .alert-card-warning {
        background: #25231c;
        border-left: 4px solid #f59e0b;
        padding: 0.85rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }

    /* Section Cards */
    .section-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }

    .section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.85rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
