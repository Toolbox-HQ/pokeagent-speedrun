cache_dir = ".cache/pokeagent/cupy"
import os
os.environ["CUPY_CACHE_DIR"] = cache_dir
os.makedirs(cache_dir, exist_ok=True)
import time
import numpy as np
import torch
import cuml

cuml.set_global_output_type("numpy")

arr: np.ndarray = torch.load("./tmp/embs.pt", weights_only=False)
arr = np.ascontiguousarray(arr, dtype=np.float32)
print(arr.shape)

t = time.time()
from cuml import DBSCAN
cluster_labels = DBSCAN(
    eps=0.005,
    min_samples=5,
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
if n_clusters > 0:
    print(f"cluster_sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, mean={cluster_sizes.mean():.1f}, median={np.median(cluster_sizes):.0f}")
    order = np.argsort(-cluster_sizes)
    print("cluster_id\tcount")
    for i in order:
        print(f"{cluster_ids[i]}\t{cluster_sizes[i]}")
