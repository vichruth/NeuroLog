"""
NeuroLog — Query engine.

Loads a prebuilt FAISS index and answers natural-language queries by embedding
the query text with the same CLIP model used at ingestion time and running an
inner-product (cosine, since vectors are normalized) nearest-neighbour search.
"""

import os
import time

import faiss
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class NeuroLogSearch:
    """Semantic search over a NeuroLog FAISS index."""

    def __init__(self, model_id=DEFAULT_MODEL, index_name="neurolog", index_dir="."):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Initializing search node on {self.device.upper()}")

        # Load CLIP and cast to FP16 on GPU to match the ingestion pipeline.
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()

        self.processor = CLIPProcessor.from_pretrained(model_id)

        index_file = os.path.join(index_dir, f"{index_name}.index")
        meta_file = os.path.join(index_dir, f"{index_name}_times.npy")

        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            raise FileNotFoundError(
                f"[!] Index not found in '{index_dir}'. Run ingest.py first."
            )

        self.index = faiss.read_index(index_file)
        self.timestamps = np.load(meta_file)

    def find_match(self, text_query, top_k=3):
        """Return the ``top_k`` best-matching frames for a text query."""
        top_k = min(top_k, self.index.ntotal)
        start_time = time.time()

        inputs = self.processor(text=[text_query], return_tensors="pt", padding=True).to(self.device)
        if self.device == "cuda":
            inputs = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        vector_np = text_features.cpu().numpy().astype("float32")
        distances, indices = self.index.search(vector_np, top_k)

        latency = round(time.time() - start_time, 3)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            results.append(
                {
                    "timestamp": float(self.timestamps[idx]),
                    "confidence": round(max(0.0, min(100.0, float(score) * 100)), 1),
                    "distance_score": round(float(score), 4),
                    "latency": latency,
                }
            )
        return results
