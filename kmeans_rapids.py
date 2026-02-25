cache_dir = ".cache/pokeagent/cupy"
import os
import random

os.environ["CUPY_CACHE_DIR"] = cache_dir
os.makedirs(cache_dir, exist_ok=True)


import sys
import time
from io import StringIO

import numpy as np
from pathlib import Path
from PIL import Image
from torchcodec.decoders import VideoDecoder
import torch
import cuml
import json
import shutil
from joblib import Parallel, delayed
from tqdm import tqdm

cuml.set_global_output_type("numpy")


class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()


log_buffer = StringIO()
_original_stdout = sys.stdout
sys.stdout = _Tee(_original_stdout, log_buffer)


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def _nearest_centroid_cosine_with_rejection(
    x_norm: np.ndarray,
    centroid_norm: np.ndarray,
    reject_threshold: float,
    batch_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    n = x_norm.shape[0]
    labels = np.empty(n, dtype=np.int32)
    max_sims = np.empty(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = x_norm[start:end] @ centroid_norm.T
        best_idx = np.argmax(sims, axis=1)
        best_sim = sims[np.arange(end - start), best_idx]
        labels[start:end] = best_idx
        max_sims[start:end] = best_sim

    labels[max_sims < reject_threshold] = -1
    return labels, max_sims


# -----------------------------------------------------------------------------
arr: np.ndarray = torch.load("./tmp/embs.nd", weights_only=False)  # (B x S) x D
arr_unflattened = torch.load("./tmp/video_emb.nd", weights_only=False)  # B x S x D
with open("./tmp/videos.json", "r") as f:
    info = json.load(f)

arr = np.ascontiguousarray(arr, dtype=np.float32)
arr = np.ascontiguousarray(_l2_normalize_rows(arr), dtype=np.float32)

print(f"data shape: {arr.shape}")
kmeans_n_clusters = 100
kmeans_max_iter = 300
kmeans_random_state = 0
cosine_reject_threshold = 0.35
min_unique_videos = 30
predict_batch_size = 100_000
print(
    "KMeans settings: "
    f"n_clusters={kmeans_n_clusters}, max_iter={kmeans_max_iter}, "
    f"random_state={kmeans_random_state}, cosine_reject_threshold={cosine_reject_threshold}, "
    f"min_unique_videos={min_unique_videos}, predict_batch_size={predict_batch_size}"
)

t = time.time()
from cuml.cluster import KMeans

n_points = arr.shape[0]
if n_points == 0:
    raise ValueError("No embeddings found in ./tmp/embs.nd")

kmeans_n_clusters = min(kmeans_n_clusters, n_points)
if kmeans_n_clusters < 1:
    raise ValueError(f"Invalid n_clusters={kmeans_n_clusters}")

kmeans = KMeans(
    n_clusters=kmeans_n_clusters,
    max_iter=kmeans_max_iter,
    random_state=kmeans_random_state,
)
kmeans.fit(arr)

centroids = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
centroids = np.ascontiguousarray(_l2_normalize_rows(centroids), dtype=np.float32)
cluster_labels, max_sims = _nearest_centroid_cosine_with_rejection(
    arr,
    centroids,
    reject_threshold=cosine_reject_threshold,
    batch_size=predict_batch_size,
)
print(f"time: {time.time() - t}")

unique, counts = np.unique(cluster_labels, return_counts=True)
noise_mask = unique == -1
n_noise = counts[noise_mask].sum() if np.any(noise_mask) else 0
cluster_ids = unique[~noise_mask]
n_clusters = len(cluster_ids)

print(f"n_points: {len(cluster_labels)}")
print(f"n_noise: {n_noise}")
print(f"n_clusters: {n_clusters}")
print(
    "max cosine similarity stats: "
    f"min={float(max_sims.min()):.4f}, "
    f"mean={float(max_sims.mean()):.4f}, "
    f"p50={float(np.percentile(max_sims, 50)):.4f}, "
    f"p95={float(np.percentile(max_sims, 95)):.4f}, "
    f"max={float(max_sims.max()):.4f}"
)


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

n_clusters_before_filter = n_clusters
cluster_ids = np.array([
    cid for cid in cluster_ids
    if cluster_info[int(cid)]["n_distinct_videos"] >= min_unique_videos
])
n_clusters = len(cluster_ids)
print(f"clusters after min_unique_videos filter: {n_clusters} (dropped {n_clusters_before_filter - n_clusters})")

# -----------------------------------------------------------------------------

clusters_sorted = sorted(
    cluster_ids,
    key=lambda cid: cluster_info[int(cid)]["n_distinct_videos"],
    reverse=True,
)

for cid in cluster_ids:
    inf = cluster_info[int(cid)]
    print(f"\nCluster {cid}: size={inf['n_points']}, distinct_videos={inf['n_distinct_videos']}")
    for path in inf["video_paths"]:
        frames = inf["frames_by_video"][path]
        print(f"  {path}: frames {min(frames)}..{max(frames)} (n={len(frames)})")


def _save_one_frame(args):
    cluster_dir, path, frame_idx, embed_interval_sec = args
    try:
        decoder = VideoDecoder(path)
        sec = embed_interval_sec * frame_idx
        frame_tensor = decoder.get_frames_played_at(seconds=[sec]).data[0]
        frame_np = frame_tensor.cpu().numpy().transpose(1, 2, 0)
        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype(np.uint8)
        else:
            frame_np = frame_np.astype(np.uint8)
        img = Image.fromarray(frame_np, mode="RGB")
        video_name = Path(path).stem
        out_path = cluster_dir / f"{video_name}.png"
        img.save(out_path)
        return (path, None)
    except Exception as e:
        return (path, str(e))


embed_interval_sec = 2.0
out_root = Path("./tmp/cluster_test")
if out_root.exists():
    shutil.rmtree(out_root)
out_root.mkdir(parents=True, exist_ok=True)
tasks = []
max_frames_per_cluster = 5
for cluster_num, cid in enumerate(clusters_sorted):
    inf = cluster_info[int(cid)]
    cluster_dir = out_root / str(cluster_num)
    cluster_dir.mkdir(exist_ok=True)
    paths = inf["video_paths"]
    if len(paths) > max_frames_per_cluster:
        paths = random.sample(paths, max_frames_per_cluster)
    for path in paths:
        frames = inf["frames_by_video"][path]
        frame_idx = random.choice(frames)
        tasks.append((cluster_dir, path, frame_idx, embed_interval_sec))
results = Parallel(n_jobs=-1)(
    delayed(_save_one_frame)(t) for t in tqdm(tasks, desc="Exporting frames")
)
for path, err in results:
    if err is not None:
        print(f"  skip {path}: {err}")

sys.stdout = _original_stdout
Path("./tmp/out.txt").parent.mkdir(parents=True, exist_ok=True)
with open("./tmp/out.txt", "w") as f:
    f.write(log_buffer.getvalue())
