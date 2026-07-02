"""
NeuroLog — Streamlit query dashboard.

A lightweight, fully offline UI for semantic video search. Point it at the video
you ingested (via the sidebar or the NEUROLOG_VIDEO environment variable) and
search it with natural language.
"""

import os

import streamlit as st

from search import NeuroLogSearch

# Configurable defaults (override via environment, no code edits required).
DEFAULT_VIDEO = os.environ.get("NEUROLOG_VIDEO", "")
INDEX_DIR = os.environ.get("NEUROLOG_INDEX_DIR", ".")

st.set_page_config(page_title="NeuroLog | Edge Interface", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stTextInput input { background-color: #1E2127; color: white; border-radius: 8px; border: 1px solid #4C4C4C; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Warming up edge AI engines...")
def load_engine():
    return NeuroLogSearch(index_dir=INDEX_DIR)


try:
    engine = load_engine()
    engine_ready = True
except Exception as e:
    st.error(f"Engine offline: {e}")
    engine_ready = False

st.title("⚡ NeuroLog | Vision Intelligence Node")
st.markdown("_Semantic Video Search · 100% Offline Edge Compute_")
st.divider()

with st.sidebar:
    st.header("Telemetry")
    st.metric(label="Compute Hardware", value="RTX 4050 (6 GB)")
    if engine_ready:
        st.metric(label="Index Size", value=f"{len(engine.timestamps)} Frames")
    st.divider()
    video_path = st.text_input(
        "Source Video",
        value=DEFAULT_VIDEO,
        placeholder="/path/to/footage.mp4",
        help="Path to the video you ingested (used for playback).",
    )

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    query = st.text_input("🔍 Search footage:", placeholder="e.g. 'person in a green jacket'")

if query and engine_ready:
    with st.spinner("Traversing vector space..."):
        results = engine.find_match(query, top_k=3)

    st.subheader(f"Top {len(results)} Semantic Matches")

    cols = st.columns(len(results))
    for idx, res in enumerate(results):
        with cols[idx]:
            seconds = int(res["timestamp"])
            m, s = divmod(seconds, 60)
            st.markdown(f"**Match #{idx + 1}** (`{m:02d}:{s:02d}`)")

            if video_path and os.path.exists(video_path):
                st.video(video_path, start_time=seconds)
            else:
                st.info("Set a valid source video path in the sidebar to preview matches.")

            st.metric("Confidence", f"{res['confidence']}%")
            with st.expander("Technical Logs"):
                st.caption(f"Latency: {res['latency']}s")
                st.caption(f"Distance: {res['distance_score']}")
