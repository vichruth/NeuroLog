# NeuroLog | Edge-Native Semantic Video Intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

**NeuroLog** is a 100% offline, zero-shot multimodal semantic video search engine. Designed to run entirely on consumer-grade edge hardware (6GB VRAM constraint), it allows users to search hours of raw CCTV or local video footage using complex natural language queries (e.g., *"person wearing a black backpack"*, *"yellow taxi crossing the street"*) without any manual tagging or cloud API dependencies.

> **Development Note:** This core architecture was engineered, optimized, and deployed as a solo development sprint in under 15 hours.

---

## System Architecture

NeuroLog bypasses traditional object-detection paradigms (like YOLO) which suffer from closed-set vocabulary limits. Instead, it utilizes a **Contrastive Language-Image Pre-training (CLIP)** pipeline coupled with a high-speed vector database to enable open-vocabulary semantic retrieval.

1. **Temporal Ingestion:** OpenCV extracts frames at a configurable temporal sample rate (default 1 FPS).
2. **Precision-Optimized Inference:** Frames are embedded using `openai/clip-vit-base-patch32`. To bypass the 6GB VRAM limitation of the target deployment hardware (RTX 4050), the model and tensors are manually cast to **FP16 (Half-Precision)** natively in PyTorch, reducing the memory footprint by 50% with near-zero degradation in recall.
3. **L2-Normalized Vector Indexing:** The resulting 512-dimensional dense vectors are L2-normalized. They are then stored locally via **FAISS** (`faiss.IndexFlatIP`). Because the vectors are normalized, the Inner Product search mathematically executes as instantaneous **Cosine Similarity**.
4. **Edge UI:** A lightweight Streamlit dashboard acts as the query node, rendering Top-K temporal matches in sub-second latency.

---

## Engineering Challenges & Iterations

Building a high-performance vision model for edge deployment requires aggressive optimization and data-driven architectural pivots. 

### 1. The 6GB VRAM Bottleneck
* **Problem:** Loading standard Vision Transformers alongside OpenCV buffering and a UI frequently triggered Out-Of-Memory (OOM) fatal errors.
* **Solution:** Bypassed standard Hugging Face pipeline wrappers to manipulate the raw PyTorch tensors directly, forcing the environment into a strict FP16 precision state. This stabilized the system to operate comfortably within a ~1.2GB VRAM envelope.

### 2. Fine-Grained Object Recall vs. Stability
* **The Experiment:** To improve the detection of small, distant objects (e.g., pedestrians in wide-angle street views), I engineered a **Dual-Embedding Center-Crop Pipeline**. This custom ingestor simultaneously processed the full $640 \times 360$ frame *and* a 50% center-zoomed tile using the higher-resolution `patch16` model, effectively doubling the FAISS index density.
* **The Rollback & Pivot:** While fine-grained recall improved, peripheral structural integrity dropped (e.g., vehicles on the edge of the frame were truncated by the crop, dropping confidence scores from 30% to <5%). I made the engineering decision to roll back to the stable full-frame `patch32` architecture. To reclaim the lost fine-grained accuracy, I upgraded the source ingestion pipeline to handle higher bitrate HD inputs, relying on cleaner pixel density rather than artificial cropping.

---

## Quick Start

### Prerequisites
* Python 3.10+
* NVIDIA GPU (CUDA toolkit installed)

### Installation
```bash
# Clone the repository
git clone [https://github.com/yourusername/NeuroLog.git](https://github.com/yourusername/NeuroLog.git)
cd NeuroLog

# Create a virtual environment
python -m venv neurolog_env
source neurolog_env/bin/activate  # On Windows use `neurolog_env\Scripts\activate`

# Install core dependencies (CPU-optimized FAISS for index storage)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install transformers opencv-python Pillow numpy faiss-cpu streamlit
