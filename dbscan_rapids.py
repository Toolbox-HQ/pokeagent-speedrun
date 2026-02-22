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

