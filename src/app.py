import streamlit as st
import time
import os
from search import NeuroLogSearch

# 1. Page Configuration
st.set_page_config(page_title="NeuroLog | Edge Interface", page_icon="🧠", layout="wide")

# 2. Custom CSS Injection (Dark Enterprise Theme)
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stTextInput input {
        background-color: #1E2127;
        color: white;
        border-radius: 8px;
        border: 1px solid #4C4C4C;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load AI Engine (Cached to prevent VRAM overflow on UI reload)
@st.cache_resource(show_spinner="Loading Edge AI Model into VRAM... (This takes a few seconds)")
def load_engine():
    return NeuroLogSearch()

try:
    engine = load_engine()
    engine_loaded = True
except Exception as e:
    st.error(f"Failed to load AI Engine. Did you run `ingest.py` first? Error: {e}")
    engine_loaded = False

# --- FRONTEND UI ---
st.title("⚡ NeuroLog | Vision Intelligence Node")
st.markdown("_Natural Language Video Search | 100% Local Edge Compute_")
st.divider()

# Sidebar: System Dashboard
with st.sidebar:
    st.markdown("### 🏢 Facility: Anchor HQ")
    st.caption("Equinox 2026 Prototype")
    st.divider()
    
    st.header("⚙️ Telemetry")
    st.metric(label="Active Compute Node", value="RTX 4050")
    st.metric(label="VRAM Status", value="Optimized (FP16)")
    if engine_loaded:
        st.metric(label="Database Size", value=f"{engine.index.ntotal} Frames")
    
    st.divider()
    video_path = st.text_input("Active Camera Stream (Local Path)", value="test.mp4")

# Main Layout
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_query = st.text_input("🔍 Semantic Search Query:", 
                                 placeholder="e.g., 'a red car' or 'person running'")

# Execution Logic
if search_query and engine_loaded:
    if not os.path.exists(video_path):
        st.error(f"[!] Video '{video_path}' not found in current directory.")
    else:
        # Progress Bar
        with st.spinner("Vectorizing text and traversing FAISS index..."):
            # The AI computes the match!
            result = engine.find_match(search_query)
            
        matched_timestamp = result["timestamp"]
        
        st.success(f"🎯 High-confidence match identified at **00:{matched_timestamp:02d}**.")
        
        # Results Layout
        res_col1, res_col2 = st.columns([3, 1])
        
        with res_col1:
            # Play video at the exact second
            st.video(video_path, start_time=matched_timestamp)
        
        with res_col2:
            st.markdown("### Match Analytics")
            st.metric(label="Confidence Score", value=f"{result['confidence']}%")
            st.metric(label="Query Latency", value=f"{result['latency']}s")
            st.metric(label="Vector Distance", value=f"{result['distance_score']}")
            
            with st.expander("Show Tensor Logs"):
                st.code(f"""
Query shape: [1, 512]
Index: FlatIP
Precision: FP16
Time: {result['latency']}s
                """, language="yaml")