import os
import json
import torch
from models.inference.find_matching_video_intervals import load_embeddings_and_metadata

# assumes load_embeddings_and_metadata is already defined

def build_merged_embeddings(folder_path: str, out_prefix: str, device: str = "cpu"):
    all_meta, out_tensor = load_embeddings_and_metadata(folder_path, device)

    out_tensor_path = f"{out_prefix}.pt"
    out_meta_path = f"{out_prefix}.json"

    torch.save(out_tensor, out_tensor_path)
    with open(out_meta_path, "w") as f:
        json.dump(all_meta, f)

    return out_meta_path, out_tensor_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--out-prefix", type=str, default="merged_embeddings")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    build_merged_embeddings(args.folder, args.out_prefix, args.device)