import torch
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import time
import os

class NeuroLogSearch:
    def __init__(self, model_id="openai/clip-vit-base-patch32", index_name="neurolog"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Initializing Search Node on {self.device.upper()}")
        
        # Manual Cast for Version Compatibility
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
            
        self.processor = CLIPProcessor.from_pretrained(model_id)
        
        index_file = f"{index_name}.index"
        meta_file = f"{index_name}_times.npy"
        
        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            raise FileNotFoundError(f"[!] Database missing. Run ingest.py first.")
            
        self.index = faiss.read_index(index_file)
        self.timestamps = np.load(meta_file)

    def find_match(self, text_query, top_k=3):
        start_time = time.time()
        inputs = self.processor(text=[text_query], return_tensors="pt", padding=True).to(self.device)
        
        if self.device == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
        vector_np = text_features.cpu().numpy().astype('float32')
        # Search for top_k matches
        distances, indices = self.index.search(vector_np, top_k)
        
        latency = round(time.time() - start_time, 3)
        results = []
        
        for i in range(top_k):
            idx = indices[0][i]
            score = distances[0][i]
            results.append({
                "timestamp": int(self.timestamps[idx]),
                "confidence": round(max(0, min(100, float(score) * 100)), 1),
                "distance_score": round(float(score), 4),
                "latency": latency
            })
            
        return results