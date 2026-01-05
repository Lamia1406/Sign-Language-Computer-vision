import streamlit as st

def styled_radio_container(
    title,
    icon_svg,
    options,
    key,
):
    
    
    
    
    
    # Use native Streamlit container
    with st.container():
        # Title with icon
        st.markdown(
        f"""
        <div style="
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: #00ccff;
        ">
            {icon_svg}{title}
        </div>
        """,
        unsafe_allow_html=True
    )

        
        # Radio button group
        selected_option = st.radio(
            f"Select {title.lower()}",
            options=options,
            key=key,
            label_visibility="collapsed"
        )
    
    return selected_option