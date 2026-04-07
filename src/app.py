import streamlit as st
import time
import os
from search import NeuroLogSearch

st.set_page_config(page_title="NeuroLog | Edge Interface", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stTextInput input { background-color: #1E2127; color: white; border-radius: 8px; border: 1px solid #4C4C4C; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Warming up Edge AI Engines...")
def load_engine():
    return NeuroLogSearch()

try:
    engine = load_engine()
    engine_ready = True
except Exception as e:
    st.error(f"Engine Offline: {e}")
    engine_ready = False

st.title("⚡ NeuroLog | Vision Intelligence Node")
st.markdown("_Semantic Video Search | 100% Offline Edge Compute_")
st.divider()

with st.sidebar:
    st.header("⚙️ Telemetry")
    st.metric(label="Compute Hardware", value="RTX 4050 (6GB)")
    if engine_ready:
        st.metric(label="Index Size", value=f"{len(engine.timestamps)} Frames")
    st.divider()
    video_path = st.text_input("Source Video", value="/home/vichruth/GitHub/NeuroLog/hackathon_videoplayback.mp4")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    query = st.text_input("🔍 Search Facility History:", placeholder="e.g. 'person in a green jacket'")

if query and engine_ready:
    with st.spinner("Traversing Vector Space..."):
        # Get Top 3 matches
        results = engine.find_match(query, top_k=3)
    
    st.subheader(f"🎯 Top {len(results)} Semantic Matches")
    
    # Create side-by-side columns
    cols = st.columns(len(results))
    
    for idx, res in enumerate(results):
        with cols[idx]:
            m, s = divmod(res['timestamp'], 60)
            st.markdown(f"**Match #{idx+1}** (`{m:02d}:{s:02d}`)")
            
            # The video player
            st.video(video_path, start_time=res['timestamp'])
            
            # Analytics below each video
            st.metric("Confidence", f"{res['confidence']}%")
            with st.expander("Technical Logs"):
                st.caption(f"Latency: {res['latency']}s")
                st.caption(f"Distance: {res['distance_score']}")