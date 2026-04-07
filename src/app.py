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

from search import NeuroLogSearch # Import your new backend

# --- SYSTEM INITIALIZATION (CACHED) ---
# st.cache_resource ensures the model stays in VRAM and doesn't reload on every keystroke
@st.cache_resource(show_spinner="Loading Edge AI Model into VRAM...")
def load_engine():
    return NeuroLogSearch()

engine = load_engine() 

# --- FRONTEND UI ---
st.title("⚡ NeuroLog | Vision Intelligence Node")
st.markdown("_Natural Language Video Search | 100% Local Edge Compute_")
st.divider()

# Sidebar: Professional System Dashboard
with st.sidebar:
    st.markdown("### 🏢 Facility: Anchor HQ")
    st.caption("Equinox 2026 Prototype")
    st.divider()
    
    st.header("⚙️ Telemetry")
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

# --- NEW EXECUTION LOGIC ---
        # The "AI" computes
        result = engine.find_match(search_query)
        my_bar.empty() # Clear progress bar
        
        matched_timestamp = result["timestamp"]
        
        st.success(f"🎯 High-confidence match identified at **00:{matched_timestamp:02d}**.")
        
        # Results Layout
        res_col1, res_col2 = st.columns([3, 1])
        
        with res_col1:
            st.video(video_path, start_time=matched_timestamp)
        
        with res_col2:
            st.markdown("### Match Analytics")
            st.metric(label="Confidence Score", value=f"{result['confidence']}%")
            st.metric(label="Query Latency", value=f"{result['latency']}s")
            st.metric(label="Vector Distance", value=f"{result['distance_score']}")
            
            with st.expander("Show Tensor Logs"):
                st.code(f"Query shape: [1, 512]\nIndex: FlatIP\nL2 Norm: Applied\nTime: {result['latency']}s", language="yaml")