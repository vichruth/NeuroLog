"""
NeuroLog — Ingestion engine.

Samples frames from a video at a configurable rate, embeds each frame with CLIP
(FP16 on GPU to fit a 6 GB VRAM budget), L2-normalizes the embeddings, and stores
them in a local FAISS index alongside their timestamps.

Usage:
    python src/ingest.py --video path/to/footage.mp4 --fps 1
"""

import argparse
import os
import time

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# If you hit a 'tokenizers' fork warning, uncomment the line below:
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
EMBED_DIM = 512  # CLIP ViT-B/32 projection dimension


class NeuroLogIngestor:
    """Builds a FAISS index of CLIP frame embeddings from a video file."""

    def __init__(self, model_id=DEFAULT_MODEL, batch_size=32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        print(f"[*] Initializing NeuroLog ingestion on {self.device.upper()}")

        # Load CLIP and manually cast to FP16 on GPU. Casting the raw model +
        # input tensors (rather than relying on pipeline wrappers) keeps GPU
        # memory ~600 MB and avoids version-specific keyword errors.
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()  # FP16 — fits the RTX 4050 6 GB budget

        self.processor = CLIPProcessor.from_pretrained(model_id)

        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.timestamps = []

    def process_video(self, video_path, sample_fps=1.0):
        """Sample the video at ``sample_fps`` frames per second and index them."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"[!] Video not found at {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"[!] Could not open video: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if native_fps <= 0:
            cap.release()
            raise RuntimeError("[!] Could not read a valid frame rate from the video.")

        duration = total_frames / native_fps
        # Read every N-th frame to approximate the requested sampling rate.
        frame_interval = max(1, round(native_fps / sample_fps))

        print(
            f"[*] Processing {video_path} | Duration: {duration:.2f}s | "
            f"Native FPS: {native_fps:.2f} | Sampling every {frame_interval} frame(s)"
        )

        batch_images, batch_times = [], []
        frame_no = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_no % frame_interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch_images.append(Image.fromarray(rgb_frame))
                batch_times.append(frame_no / native_fps)

                if len(batch_images) == self.batch_size:
                    self._embed_and_index(batch_images, batch_times)
                    batch_images, batch_times = [], []

            frame_no += 1

        if batch_images:
            self._embed_and_index(batch_images, batch_times)

        cap.release()
        print(
            f"[+] Ingestion complete — embedded {self.index.ntotal} frames "
            f"in {time.time() - start_time:.2f}s."
        )

    def _embed_and_index(self, images, times):
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)

        # Match input tensors to the model's half precision on GPU.
        if self.device == "cuda":
            inputs = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        embeddings_np = image_features.cpu().numpy().astype("float32")
        self.index.add(embeddings_np)
        self.timestamps.extend(times)

    def save_index(self, index_name="neurolog", output_dir="."):
        """Persist the FAISS index and the parallel timestamp array."""
        os.makedirs(output_dir, exist_ok=True)
        index_path = os.path.join(output_dir, f"{index_name}.index")
        times_path = os.path.join(output_dir, f"{index_name}_times.npy")

        faiss.write_index(self.index, index_path)
        np.save(times_path, np.array(self.timestamps, dtype="float32"))
        print(f"[*] Index saved: {index_path} and {times_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest a video into a local NeuroLog FAISS index."
    )
    parser.add_argument("--video", required=True, help="Path to the source video file.")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames sampled per second (default: 1).")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size (default: 32).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face CLIP model id.")
    parser.add_argument("--index-name", default="neurolog", help="Base name for the output index files.")
    parser.add_argument("--output-dir", default=".", help="Directory to write the index into (default: cwd).")
    return parser.parse_args()


def main():
    args = parse_args()
    ingestor = NeuroLogIngestor(model_id=args.model, batch_size=args.batch_size)
    ingestor.process_video(args.video, sample_fps=args.fps)
    ingestor.save_index(index_name=args.index_name, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
