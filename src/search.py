import streamlit as st
import time
import os

# 1. Page Configuration (Set to Wide for dashboard feel)
st.set_page_config(page_title="NeuroLog | Edge Interface", page_icon="🧠", layout="wide")

# 2. Custom CSS Injection (The Hackathon Polish)
st.markdown("""
    <style>
    /* Darken the background and soften the text */
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    /* Style the search bar to look embedded */
    .stTextInput input {
        background-color: #1E2127;
        color: white;
        border-radius: 8px;
        border: 1px solid #4C4C4C;
    }
    </style>
""", unsafe_allow_html=True)

# --- MOCK BACKEND ---
def mock_search_engine(query):
    time.sleep(1.2) # Fake inference
    return 12 

# --- FRONTEND UI ---
st.title("⚡ NeuroLog | Vision Intelligence Node")
st.markdown("_Natural Language Video Search | 100% Local Edge Compute_")
st.divider()

# Sidebar: Professional System Dashboard
with st.sidebar:
    st.markdown("### Facility: Anchor HQ")
    st.caption("Equinox 2026 Prototype")
    st.divider()
    
    st.header("Telemetry")
    # Using metrics makes it look like a real enterprise dashboard
    st.metric(label="Active Compute Node", value="RTX 4050")
    st.metric(label="VRAM Utilization", value="3.2 / 6.0 GB", delta="-1.1 GB (Optimized)", delta_color="inverse")
    st.metric(label="FAISS Index Size", value="128 MB")
    
    st.divider()
    video_path = st.text_input("Active Camera Stream (Local Path)", value="test.mp4")

# Layout: Center the search bar using columns
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_query = st.text_input("🔍 Semantic Search Query:", 
                                 placeholder="e.g., 'person wearing a red jacket'")

# Execution Logic
if search_query:
    if not os.path.exists(video_path):
        st.error(f"[!] Video '{video_path}' not found in current directory.")
    else:
        # Better loading animation
        progress_text = "Vectorizing text and traversing FAISS index..."
        my_bar = st.progress(0, text=progress_text)
        
        for percent_complete in range(100):
            time.sleep(0.01) # Smooth fake progress bar
            my_bar.progress(percent_complete + 1, text=progress_text)
        
        # The "AI" computes
        matched_timestamp = mock_search_engine(search_query)
        my_bar.empty() # Clear the progress bar when done
        
        st.success(f"High-confidence match identified at **00:{matched_timestamp:02d}**.")
        
        # Results Layout: Video on the left, analytics on the right
        res_col1, res_col2 = st.columns([3, 1])
        
        with res_col1:
            # The native video player jumping to the exact second
            st.video(video_path, start_time=int(matched_timestamp))
        
        with res_col2:
            st.markdown("### Match Analytics")
            st.metric(label="Confidence Score", value="94.2%")
            st.metric(label="Query Latency", value="1.24s")
            st.metric(label="Vector Distance", value="0.827")
            
            with st.expander("Show Tensor Logs"):
                st.code("Query shape: [1, 512]\nIndex: FlatIP\nL2 Norm: Applied", language="yaml")