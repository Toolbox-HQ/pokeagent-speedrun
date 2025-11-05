import torch
import json
from transformers import CLIPProcessor, CLIPModel
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm
from pathlib import Path
import glob # Import glob for pattern matching files

# --- CONFIGURATION ---
# Note: Ensure MODEL_PATH is accessible or update to 'openai/clip-vit-base-patch32' for online download
MODEL_PATH = "/home/bsch/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
# List of common video extensions to look for
VIDEO_EXTENSIONS = ['*.mp4', '*.avi', '*.mov', '*.mkv']

# --- CORE FUNCTIONS (Unchanged as they are designed for a single video) ---

def sample_frames(video_path: str, interval_sec: float):
    """
    Sample frames from video using torchcodec.
    Returns (frames, metadata) for inference.
    """
    try:
        decoder = VideoDecoder(video_path)
    except Exception as e:
        print(f"🛑 Error decoding {video_path}: {e}")
        return [], []

    fps = decoder.frame_rate
    total_frames = decoder.num_frames
    
    # Handle the case where frame_rate might be 0 or total_frames 0
    if fps is None or total_frames == 0 or fps == 0:
         print(f"⚠️ Skipping {video_path}: Invalid frame rate ({fps}) or no frames ({total_frames}).")
         return [], []

    frame_interval = int(fps * interval_sec)
    if frame_interval <= 0:
        frame_interval = 1 # Sample at least one frame if interval is too small or fps is huge

    frames, metadata = [], []

    print(f"\n🎬 Decoding {video_path}: {total_frames} frames, sampling every {frame_interval} frames ({interval_sec}s)")
    for i, frame in enumerate(decoder):
        if i % frame_interval == 0:
            frames.append(frame.to_pil())
            metadata.append({"frame_index": i, "timestamp_sec": i / fps})
    return frames, metadata


def generate_clip_frame_embeddings(video_path: str, interval_sec: float = 2.0, batch_size: int = 8,
                                   device: str = "cuda"):
    """
    Generate CLIP embeddings for sampled frames, stack them into one tensor.
    Returns (embeddings_tensor, metadata)
    """
    # Load model and processor only once per process, but kept inside for simplicity
    # For performance, this could be moved outside and passed as an argument.
    model = CLIPModel.from_pretrained(MODEL_PATH).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_PATH)
    
    frames, meta = sample_frames(video_path, interval_sec)
    
    if not frames:
        return None, None # Return None if no frames were sampled

    print(f"Sampled {len(frames)} frames")

    all_embeddings = []

    for start in tqdm(range(0, len(frames), batch_size), desc=f"Embedding {Path(video_path).stem}"):
        batch_frames = frames[start:start + batch_size]
        inputs = processor(images=batch_frames, return_tensors="pt").to(device)

        with torch.no_grad():
            image_embeds = model.get_image_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        all_embeddings.append(image_embeds.cpu())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    return embeddings_tensor, meta


def save_video_embeddings(video_path: str, embeddings: torch.Tensor, metadata: list, output_dir: str):
    """
    Save embeddings as a .pt tensor and metadata as a .json file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    base_name = Path(video_path).stem

    tensor_path = Path(output_dir) / f"{base_name}_embeddings.pt"
    meta_path = Path(output_dir) / f"{base_name}_metadata.json"

    torch.save(embeddings, tensor_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved embeddings tensor to: {tensor_path}")
    print(f"✅ Saved frame metadata to: {meta_path}")
    print(f"Shape: {tuple(embeddings.shape)}")


# --- EXECUTION BLOCK (Modified) ---

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate and save CLIP frame embeddings for all videos in a directory.")
    # Changed 'video_path' to 'input_path'
    parser.add_argument("input_path", type=str, help="Path to input video DIRECTORY or a single video file.")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between sampled frames")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--output_dir", type=str, default="./.cache/pokeagent/embeddings", help="Directory to save outputs")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    video_paths = []

    if input_path.is_dir():
        # Find all files matching the video extensions recursively
        print(f"Searching for videos in directory: {input_path}")
        for ext in VIDEO_EXTENSIONS:
            video_paths.extend(glob.glob(str(input_path / '**' / ext), recursive=True))
    elif input_path.is_file():
        # Handle the case where the input is still a single video file
        print(f"Input is a single video file.")
        video_paths.append(str(input_path))
    else:
        print(f"Error: Input path '{input_path}' is neither a valid directory nor a file.")
        exit(1)

    if not video_paths:
        print(f"No videos found in {input_path} with extensions: {', '.join(VIDEO_EXTENSIONS)}")
        exit(0)

    print(f"\nProcessing {len(video_paths)} video(s)...")
    
    # --- Main Loop to Process Videos ---
    for video_path in video_paths:
        try:
            print(f"\n--- Starting process for: {video_path} ---")
            
            # The model and processor are reloaded inside generate_clip_frame_embeddings 
            # which is fine for simplicity but can be optimized if needed.
            embeddings, metadata = generate_clip_frame_embeddings(
                video_path=video_path,
                interval_sec=args.interval,
                batch_size=args.batch_size,
                device=args.device
            )

            if embeddings is not None:
                save_video_embeddings(video_path, embeddings, metadata, args.output_dir)
            else:
                 print(f"Skipping saving for {video_path} due to failed/empty embedding generation.")

        except Exception as e:
            print(f"An error occurred while processing {video_path}: {e}")
            
    print("\n**All video processing complete!**")