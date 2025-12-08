import os
import json
import torch
from .find_matching_video_intervals import load_embeddings_and_metadata
import math

def build_merged_embeddings(input_folder: str, out_folder: str, split: int):
    all_meta, out_tensor = load_embeddings_and_metadata(input_folder)
    if len(all_meta) != out_tensor.size(0):
        raise AssertionError("metadata and tensor must be the same size")
    total_length = len(all_meta)

    chunk_size = math.ceil(total_length / split)
    i = 0
    os.makedirs(out_folder, exist_ok=True)

    while i < split:
        start = i*chunk_size
        end = min((i + 1)*chunk_size, total_length)
        torch.save(out_tensor[start : end].clone(), f"{out_folder}/{i}.pt")
        with open(f"{out_folder}/{i}.json", "w") as f:
            json.dump(all_meta[start : end], f)   
        i += 1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--splits", type=int)
    args = parser.parse_args()
    build_merged_embeddings(args.input_folder, args.output_folder, args.splits)