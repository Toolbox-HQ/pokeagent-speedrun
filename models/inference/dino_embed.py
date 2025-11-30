import torch
from torch.utils.data import Dataset
from transformers import AutoModel
from pathlib import Path
from torchcodec.decoders import VideoDecoder
from typing import List, Tuple
from joblib import delayed, Parallel
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
import os
from models.util.data import save_json
from models.util.misc import local_model_map
from pathlib import Path

MODEL_PATH = local_model_map("facebook/dinov2-base")
OUTPUT_DIR = ".cache/pokeagent/dinov2"
INPUT_DIR = ".cache/pokeagent/internet_data"

def in_cache(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    meta_filename = os.path.join(OUTPUT_DIR, f"{filename}.json")
    pt_filename = os.path.join(OUTPUT_DIR, f"{filename}.pt")
    os.makedirs(os.path.dirname(pt_filename), exist_ok=True)

    # guard if script was resumed
    if all(Path(p).exists() for p in [meta_filename, pt_filename]):
        return True
    else:
        return False
    
class EmbeddingDataset(Dataset):
    
    def __init__(self, path=INPUT_DIR):
        self.video_list = list(filter(lambda x: not in_cache(x), Path(path).iterdir()))
        self.interval_second = 2
        self.batch_size = 32

        print(f"[DATASET] Embedding {len(self.video_list)} videos")
        print(f"[DATASET] From {path}")
        print(f"[DATASET] into {OUTPUT_DIR}")

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        video_path = self.video_list[index]
        batch_ind, vr = EmbeddingDataset.batch_video(video_path, self.batch_size, self.interval_second)
        return video_path, batch_ind, vr

    @staticmethod
    def batch_video(path, batch_size: int, interval_second: float) -> Tuple[List[List], VideoDecoder]:
        decoder = VideoDecoder(path)
        fps = decoder.metadata.average_fps
        total_frames = len(decoder)
        frame_interval = round(fps * interval_second)

        framebatch_lol, framebatch = [], []
        for frame_idx in range(0, total_frames, frame_interval):
            framebatch.append(frame_idx)
            if len(framebatch) == batch_size:
                framebatch_lol.append(framebatch)
                framebatch = []
        if framebatch:
            framebatch_lol.append(framebatch)
        return framebatch_lol, decoder


def prcoess_batch(model, processor, path: str, batch: List[List]):
    vr = VideoDecoder(path)
    frames = vr.get_frames_at(batch).data
    inputs = processor(images=frames, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.last_hidden_state[:, 0]  # CLS token
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    return image_embeds


def main():
    model = AutoModel.from_pretrained(MODEL_PATH)
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    ds = EmbeddingDataset()

    for i in tqdm(range(len(ds))): #path, batches, vr in tqdm(ds):
        try:
            path, batches, vr = ds[i]
            filename = os.path.splitext(os.path.basename(path))[0]
            meta_filename = os.path.join(OUTPUT_DIR, f"{filename}.json")
            pt_filename = os.path.join(OUTPUT_DIR, f"{filename}.pt")
            os.makedirs(os.path.dirname(pt_filename), exist_ok=True)

            # guard if script was resumed
            if all(Path(p).exists() for p in [meta_filename, pt_filename]):
                print(f"{filename} skipped")
                continue
            results = Parallel(n_jobs=64, backend="threading")(
                delayed(prcoess_batch)(model, processor, path, batch) for batch in batches
            )
            results = torch.cat(results)
            

            # save embeddings
            pt_filename = os.path.join(OUTPUT_DIR, f"{filename}.pt")
            
            torch.save(results, pt_filename)

            # save metadata
            metadata = [{
                "video_path": str(path),
                "video_fps": vr.metadata.average_fps,
                "video_total_frames": len(vr),
                "sampled_frame_index": idx,
            } for batch in batches for idx in batch]
            save_json(meta_filename, metadata)
            print([f"[SUCCESS] Saved {str(path)}"])
        
        except Exception as e:
            print(f"[ERROR] {str(e)}")

if __name__ == "__main__":
    main()