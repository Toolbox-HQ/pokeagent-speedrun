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


# -----------------------------------------------------------------------------
arr: np.ndarray = torch.load("./tmp/embs.nd", weights_only=False) # (B x S) x D
arr_unflattened = torch.load("./tmp/video_emb.nd", weights_only=False) # B x S x D
with open("./tmp/videos.json", "r") as f:
    info = json.load(f)
    
arr = np.ascontiguousarray(arr, dtype=np.float32)

print(f"data shape: {arr.shape}")
dbscan_eps = 0.1
dbscan_min_samples = 20
dbscan_algorithm = "brute_force"
dbscan_metric = "euclidean"
dbscan_min_unique_videos = 5
print(f"HDBSCAN settings: cluster_selection_epsilon={dbscan_eps}, min_samples={dbscan_min_samples}, build_algo={dbscan_algorithm}, metric={dbscan_metric}, min_unique_videos={dbscan_min_unique_videos}")

t = time.time()
from cuml.cluster import HDBSCAN
cluster_labels = HDBSCAN(
    min_cluster_size=dbscan_min_samples,
    min_samples=dbscan_min_samples,
    cluster_selection_epsilon=dbscan_eps,
    metric=dbscan_metric,
    build_algo=dbscan_algorithm,
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

n_clusters_before_filter = n_clusters
cluster_ids = np.array([
    cid for cid in cluster_ids
    if cluster_info[int(cid)]["n_distinct_videos"] >= dbscan_min_unique_videos
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

