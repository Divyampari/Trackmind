import streamlit as st
from ui.styles import apply_custom_styles
from ui.dashboard import render_dashboard

# Configure Streamlit Page Settings for Phase 4 Dashboard
st.set_page_config(
    page_title="FactoryGuard AI — Autonomous Factory Safety Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Inject Custom Industrial Dark Theme Styles
    apply_custom_styles()
    
    # Render Phase 4 Streamlit Dashboard
    render_dashboard()

if __name__ == "__main__":
    main()
