cache_dir = ".cache/pokeagent/cupy"
import os
os.environ["CUPY_CACHE_DIR"] = cache_dir
os.makedirs(cache_dir, exist_ok=True)


import time
import numpy as np
import torch
import cuml
import json
cuml.set_global_output_type("numpy")

arr: np.ndarray = torch.load("./tmp/embs.nd", weights_only=False) # (B x S) x D
arr_unflattened = torch.load("./tmp/video_emb.nd", weights_only=False) # B x S x D
with open("/tmp/videos.json", "r") as f:
    info = json.load(f)
    
arr = np.ascontiguousarray(arr, dtype=np.float32)
print(arr.shape)

t = time.time()
from cuml import DBSCAN
cluster_labels = DBSCAN(
    eps=0.1,
    min_samples=100,
    algorithm="brute",
    metric="euclidean",
).fit_predict(arr)
print(f"time: {time.time() - t}")

unique, counts = np.unique(cluster_labels, return_counts=True)
noise_mask = unique == -1
n_noise = counts[noise_mask].sum() if np.any(noise_mask) else 0
cluster_ids = unique[~noise_mask]
cluster_sizes = counts[~noise_mask]
n_clusters = len(cluster_ids)

print(f"n_points: {len(cluster_labels)}")
print(f"n_noise: {n_noise}")
print(f"n_clusters: {n_clusters}")

frame_counts = np.array([t.shape[0] for t in arr_unflattened], dtype=np.int64)
video_offsets = np.concatenate([[0], np.cumsum(frame_counts)[:-1]])
video_paths = [v["video_path"] for v in info]

def flat_idx_to_video_frame(flat_idx):
    video_idx = np.searchsorted(video_offsets, flat_idx, side="right") - 1
    frame_idx = int(flat_idx - video_offsets[video_idx])
    return video_idx, frame_idx

cluster_info = {}
for cid in cluster_ids:
    flat_indices = np.where(cluster_labels == cid)[0]
    videos_in_cluster = set()
    frames_by_video = {}
    for idx in flat_indices:
        video_idx, frame_idx = flat_idx_to_video_frame(idx)
        path = video_paths[video_idx]
        videos_in_cluster.add(path)
        frames_by_video.setdefault(path, []).append(frame_idx)
    cluster_info[int(cid)] = {
        "n_points": len(flat_indices),
        "n_distinct_videos": len(videos_in_cluster),
        "video_paths": sorted(videos_in_cluster),
        "frames_by_video": {k: sorted(v) for k, v in frames_by_video.items()},
    }

for cid in cluster_ids:
    inf = cluster_info[int(cid)]
    print(f"\nCluster {cid}: size={inf['n_points']}, distinct_videos={inf['n_distinct_videos']}")
    for path in inf["video_paths"]:
        frames = inf["frames_by_video"][path]
        print(f"  {path}: frames {min(frames)}..{max(frames)} (n={len(frames)})")

