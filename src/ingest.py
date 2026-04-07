import cv2
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import time
import os

class NeuroLogIngestor:
    def __init__(self, model_id="openai/clip-vit-base-patch32", batch_size=32):
        # 1. Device and Precision Setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.batch_size = batch_size
        
        print(f"[*] Initializing NeuroLog Ingestion on {self.device.upper()} (Precision: {self.dtype})")

        # 2. Load Model & Processor (Optimized for 6GB VRAM)
        self.model = CLIPModel.from_pretrained(
            model_id, 
            torch_dtype=self.dtype
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        
        # 3. Initialize FAISS Index (CLIP outputs 512-dim vectors)
        self.dimension = 512
        self.index = faiss.IndexFlatIP(self.dimension)
        self.timestamps = [] 

    def process_video(self, video_path):
        """Extracts 1 frame per second efficiently and batches them for CLIP."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"[!] Video not found at {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"[*] Processing {video_path} | Duration: {duration:.2f}s | FPS: {fps}")

        batch_images = []
        batch_times = []
        
        start_time = time.time()

        # Iterate through the video second by second
        for current_sec in range(int(duration)):
            # Fast-forward to the specific timestamp (in milliseconds)
            cap.set(cv2.CAP_PROP_POS_MSEC, current_sec * 1000)
            ret, frame = cap.read()
            
            if not ret:
                continue

            # Convert BGR (OpenCV) to RGB (Pillow/CLIP standard)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            
            batch_images.append(pil_img)
            batch_times.append(current_sec)

            # Once batch is full, push to GPU and index
            if len(batch_images) == self.batch_size:
                self._embed_and_index(batch_images, batch_times)
                batch_images, batch_times = [], []

        # Process any remaining frames
        if batch_images:
            self._embed_and_index(batch_images, batch_times)

        cap.release()
        elapsed = time.time() - start_time
        print(f"[+] Ingestion Complete! Embedded {self.index.ntotal} frames in {elapsed:.2f}s.")

    def _embed_and_index(self, images, times):
        """Passes images through CLIP and adds vectors to FAISS."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device, dtype=self.dtype)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize for Cosine Similarity
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        embeddings_np = image_features.cpu().numpy().astype('float32')
        self.index.add(embeddings_np)
        self.timestamps.extend(times)

    def save_index(self, index_name="neurolog"):
        """Persists the FAISS index and timestamp metadata to disk."""
        faiss.write_index(self.index, f"{index_name}.index")
        np.save(f"{index_name}_times.npy", np.array(self.timestamps))
        print(f"[*] DB Saved: {index_name}.index and {index_name}_times.npy")

if __name__ == "__main__":
    # --- RUN THIS FIRST ---
    ingestor = NeuroLogIngestor(batch_size=32) 
    
    # Make sure you have a sample video named 'test.mp4' in the same folder!
    try:
        ingestor.process_video("test.mp4")
        ingestor.save_index()
    except Exception as e:
        print(f"[!] Error: {e}")