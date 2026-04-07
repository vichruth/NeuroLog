import cv2
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
# If you get a 'tokenizers' error, uncomment the line below:
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import os

class NeuroLogIngestor:
    def __init__(self, model_id="openai/clip-vit-base-patch32", batch_size=32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        
        print(f"[*] Initializing NeuroLog Ingestion on {self.device.upper()}")

        # Manual Cast Method (Bypasses version-specific keyword errors)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        if self.device == "cuda":
            self.model = self.model.half() # Optimized for RTX 4050 6GB
            
        self.processor = CLIPProcessor.from_pretrained(model_id)
        
        self.dimension = 512
        self.index = faiss.IndexFlatIP(self.dimension)
        self.timestamps = [] 

    def process_video(self, video_path):
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

        for current_sec in range(int(duration)):
            cap.set(cv2.CAP_PROP_POS_MSEC, current_sec * 1000)
            ret, frame = cap.read()
            if not ret: continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            
            batch_images.append(pil_img)
            batch_times.append(current_sec)

            if len(batch_images) == self.batch_size:
                self._embed_and_index(batch_images, batch_times)
                batch_images, batch_times = [], []

        if batch_images:
            self._embed_and_index(batch_images, batch_times)

        cap.release()
        print(f"[+] Ingestion Complete! Embedded {self.index.ntotal} frames in {time.time() - start_time:.2f}s.")

    def _embed_and_index(self, images, times):
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        
        # Ensure input tensors match the model's half-precision
        if self.device == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        embeddings_np = image_features.cpu().numpy().astype('float32')
        self.index.add(embeddings_np)
        self.timestamps.extend(times)

    def save_index(self, index_name="neurolog"):
        faiss.write_index(self.index, f"{index_name}.index")
        np.save(f"{index_name}_times.npy", np.array(self.timestamps))
        print(f"[*] DB Saved: {index_name}.index and {index_name}_times.npy")

if __name__ == "__main__":
    ingestor = NeuroLogIngestor(batch_size=32) 
    video_path = "/home/vichruth/GitHub/NeuroLog/hackathon_videoplayback.mp4"
    try:
        ingestor.process_video(video_path)
        ingestor.save_index()
    except Exception as e:
        print(f"[!] Critical Error during ingestion: {e}")