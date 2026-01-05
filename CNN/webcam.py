import streamlit as st
from CNN.components.camera import camera
def webcam(learn):
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False
    if "gesture_buffer" not in st.session_state:
        st.session_state.gesture_buffer = []

    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.html("""<div style="
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
    flex-wrap: wrap;
    gap: 1rem;
">
    <h3 style="margin: 0;">Real-time Sign Recognition</h3>""", )
    
    with col2:
        button_label = "Stop Camera" if st.session_state.camera_running else "Start Camera"
        button_type = "secondary" if st.session_state.camera_running else "primary"
        
        if st.button(button_label, type=button_type, width="stretch"):
            st.session_state.camera_running = not st.session_state.camera_running
            st.rerun()
    camera(learn)
    