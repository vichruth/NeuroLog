import torch
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import time
import os

class NeuroLogSearch:
    def __init__(self, model_id="openai/clip-vit-base-patch32", index_name="neurolog"):
        # 1. Hardware Setup (Matching our Ingestion script)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"[*] Initializing Search Node on {self.device.upper()}...")
        
        # 2. Load Model 
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        
        # 3. Load FAISS Index and Metadata
        index_file = f"{index_name}.index"
        meta_file = f"{index_name}_times.npy"
        
        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            raise FileNotFoundError(f"[!] Database missing. Did you run ingest.py first?")
            
        self.index = faiss.read_index(index_file)
        self.timestamps = np.load(meta_file)
        print(f"[+] NeuroLog Search Ready. Monitoring {self.index.ntotal} frames.")

    def find_match(self, text_query, top_k=1):
        """Converts text to a vector and searches the FAISS database."""
        start_time = time.time()
        
        # 1. Vectorize the Text
        inputs = self.processor(text=[text_query], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # CRITICAL: Normalize text vector just like we did with the image vectors!
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
        # 2. Query FAISS
        vector_np = text_features.cpu().numpy().astype('float32')
        distances, indices = self.index.search(vector_np, top_k)
        
        latency = time.time() - start_time
        
        # 3. Extract Results
        best_idx = indices[0][0]
        raw_score = distances[0][0] # Inner product score
        
        # Convert raw inner-product score to a clean percentage for the UI
        confidence_pct = round(max(0, min(100, float(raw_score) * 100)), 1)
        
        return {
            "timestamp": int(self.timestamps[best_idx]),
            "confidence": confidence_pct,
            "latency": round(latency, 3),
            "distance_score": round(float(raw_score), 4)
        }

if __name__ == "__main__":
    # Quick Terminal Test
    engine = NeuroLogSearch()
    result = engine.find_match("a person wearing a backpack")
    print(f"Result: {result}")