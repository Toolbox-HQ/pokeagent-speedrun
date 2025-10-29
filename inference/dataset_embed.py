import torch
from torch.utils.data import Dataset
from transformers import AutoModel
import json
from pathlib import Path
from torchcodec.decoders import VideoDecoder
from typing import List, Tuple
from joblib import delayed, Parallel
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
import random 
import os
from util.data import save_json


MODEL_PATH = "/scratch/bsch/hf_cache/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
OUTPUT_DIR = ".cache/pokeagent/clip32"

class EmbeddingDataset(Dataset):
    
    def __init__(self, path=".cache/pokeagent/video_list.json"):
        
        with open(path, "r") as f:
            self.video_list = json.loads(f.read())

        self.interval_second = 2
        self.batch_size = 32

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        video_path = self.video_list[index]
        batch_ind, vr = EmbeddingDataset.batch_video(video_path, self.batch_size, self.interval_second)
        return video_path, batch_ind, vr


    @staticmethod
    def batch_video(path, batch_size: int, interval_second: float)-> Tuple[ List[List], VideoDecoder ] :

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



def split():
    path = ".cache/pokeagent/internet_data"
    files = [str(p) for p in Path(path).iterdir() if p.is_file()]
    random.shuffle(files)
    with open(".cache/pokeagent/video_list.json", "w") as f:
        json.dump(files, f)

def prcoess_batch(model, processor, path: str, batch: List[List]):
    
    vr = VideoDecoder(path)
    frames = vr.get_frames_at(batch).data
    inputs =  processor(images=frames, return_tensors="pt")
    with torch.no_grad():
        image_embeds = model.get_image_features(**inputs)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

    return image_embeds

def main():
    model = AutoModel.from_pretrained(MODEL_PATH)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    ds = EmbeddingDataset()

    for path, batches, _ in tqdm(ds):
        results = Parallel(n_jobs=64, backend="threading")(
            delayed(prcoess_batch)(model, processor, path, batch) for batch in batches
        )
        results = torch.cat(results)
        filename = os.path.splitext(os.path.basename(path))[0]

        # save embeddings
        pt_filename = os.path.join(OUTPUT_DIR, f"{filename}.pt")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(results, pt_filename)

        # save metadata
        metadata = {} 
        meta_filename = os.path.join(OUTPUT_DIR, f"{filename}.json")
        save_json(meta_filename, metadata)

if __name__ == "__main__":
    import sys
    if sys.argv[1] == "meta":
        split()
    else:
        main()