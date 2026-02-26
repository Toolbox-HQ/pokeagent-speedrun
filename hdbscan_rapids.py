cache_dir = ".cache/pokeagent/cupy"
import argparse
import json
import os
import random
import shutil
import sys
import time
from io import StringIO
from pathlib import Path

import cuml
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from models.util.misc import local_model_map

os.environ["CUPY_CACHE_DIR"] = cache_dir
os.makedirs(cache_dir, exist_ok=True)
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


def _build_timestamps(duration_seconds: float, interval_seconds: float) -> list[float]:
    timestamps = []
    t = 0.0
    while t < duration_seconds:
        timestamps.append(t)
        t += interval_seconds
    return timestamps


def _embed_video_every_interval(
    video_path: str,
    model: AutoModel,
    processor: AutoImageProcessor,
    interval_seconds: float,
    batch_size: int,
    device: torch.device,
) -> tuple[list[float], np.ndarray]:
    decoder = VideoDecoder(video_path)
    duration = float(decoder.metadata.duration_seconds)
    timestamps = _build_timestamps(duration, interval_seconds)
    if not timestamps:
        return [], np.empty((0, 0), dtype=np.float32)

    chunks = []
    for start in tqdm(range(0, len(timestamps), batch_size), desc="Embedding query video"):
        end = min(start + batch_size, len(timestamps))
        batch_ts = timestamps[start:end]
        frames = decoder.get_frames_played_at(seconds=batch_ts).data
        inputs = processor(images=frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.last_hidden_state[:, 0]
            image_embeds = F.normalize(image_embeds, p=2, dim=-1, eps=1e-12)
        chunks.append(image_embeds.cpu())

    embeddings = torch.cat(chunks, dim=0).numpy().astype(np.float32, copy=False)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    return timestamps, embeddings


def _predict_hdbscan_labels(clusterer, x: np.ndarray) -> np.ndarray:
    if hasattr(clusterer, "predict"):
        return np.asarray(clusterer.predict(x), dtype=np.int32)

    try:
        from cuml.cluster.hdbscan import approximate_predict
    except Exception as exc:
        raise RuntimeError("HDBSCAN prediction is unavailable in this cuML build.") from exc

    pred = approximate_predict(clusterer, x)
    labels = pred[0] if isinstance(pred, tuple) else pred
    return np.asarray(labels, dtype=np.int32)


def _save_one_cluster_example_frame(args):
    cluster_dir, video_path, frame_idx, embed_interval_sec = args
    try:
        decoder = VideoDecoder(video_path)
        sec = embed_interval_sec * frame_idx
        frame_tensor = decoder.get_frames_played_at(seconds=[sec]).data[0]
        frame_np = frame_tensor.cpu().numpy().transpose(1, 2, 0)
        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype(np.uint8)
        else:
            frame_np = frame_np.astype(np.uint8)
        img = Image.fromarray(frame_np, mode="RGB")
        out_path = cluster_dir / f"{Path(video_path).stem}.png"
        img.save(out_path)
        return (video_path, None)
    except Exception as exc:
        return (video_path, str(exc))


def _parse_args():
    parser = argparse.ArgumentParser(description="Fit HDBSCAN and classify a query video by embedding.")
    parser.add_argument("video_path", type=str, help="Path to a query video file.")
    parser.add_argument("--train-embeddings-path", default="./tmp/embs.nd", type=str)
    parser.add_argument("--train-video-embeddings-path", default="./tmp/video_emb.nd", type=str)
    parser.add_argument("--videos-info-path", default="./tmp/videos.json", type=str)
    parser.add_argument("--interval-seconds", default=2.0, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--model-id", default="facebook/dinov2-base", type=str)
    parser.add_argument("--min-cluster-size", default=30, type=int)
    parser.add_argument("--min-samples", default=30, type=int)
    parser.add_argument("--min-unique-videos", default=30, type=int)
    parser.add_argument("--metric", default="euclidean", type=str)
    parser.add_argument("--build-algo", default="brute_force", type=str)
    parser.add_argument("--cluster-images-dir", default="./tmp/cluster_test", type=str)
    parser.add_argument("--max-frames-per-cluster", default=5, type=int)
    return parser.parse_args()


def main():
    args = _parse_args()
    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    arr: np.ndarray = torch.load(args.train_embeddings_path, weights_only=False)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D train embeddings, got shape={arr.shape}")

    arr_unflattened = torch.load(args.train_video_embeddings_path, weights_only=False)
    with open(args.videos_info_path, "r") as f:
        info = json.load(f)

    print(f"train embedding shape: {arr.shape}")
    print(
        "HDBSCAN settings: "
        f"min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}, "
        f"build_algo={args.build_algo}, metric={args.metric}, "
        f"min_unique_videos={args.min_unique_videos}"
    )

    t = time.time()
    from cuml.cluster import HDBSCAN

    clusterer = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.metric,
        build_algo=args.build_algo,
        prediction_data=True,
    )
    clusterer.fit(arr)
    print(f"hdbscan fit time: {time.time() - t:.2f}s")
    train_labels = np.asarray(clusterer.labels_, dtype=np.int64)

    frame_counts = np.array([t.shape[0] for t in arr_unflattened], dtype=np.int64)
    if frame_counts.sum() != len(train_labels):
        raise RuntimeError(
            f"Train label count ({len(train_labels)}) does not match flattened frame count ({frame_counts.sum()})"
        )
    video_offsets = np.concatenate([[0], np.cumsum(frame_counts)[:-1]])
    video_paths = [v["video_path"] for v in info]
    if len(video_paths) != len(frame_counts):
        raise RuntimeError(
            f"videos.json count ({len(video_paths)}) does not match number of video embeddings ({len(frame_counts)})"
        )

    cluster_to_videos = {}
    cluster_frames_by_video = {}
    for flat_idx, raw_label in enumerate(train_labels.tolist()):
        raw_label = int(raw_label)
        if raw_label == -1:
            continue
        video_idx = np.searchsorted(video_offsets, flat_idx, side="right") - 1
        frame_idx = int(flat_idx - video_offsets[video_idx])
        video_path_i = video_paths[video_idx]
        cluster_to_videos.setdefault(raw_label, set()).add(video_path_i)
        cluster_frames_by_video.setdefault(raw_label, {}).setdefault(video_path_i, []).append(frame_idx)

    filtered_cluster_ids = [
        cid
        for cid, vids in cluster_to_videos.items()
        if len(vids) >= args.min_unique_videos
    ]
    filtered_cluster_ids = sorted(
        filtered_cluster_ids,
        key=lambda cid: len(cluster_to_videos[cid]),
        reverse=True,
    )
    label_to_filtered_index = {
        int(raw_label): idx for idx, raw_label in enumerate(filtered_cluster_ids)
    }
    print(
        "clusters after min_unique_videos filter: "
        f"{len(filtered_cluster_ids)} (from {len(cluster_to_videos)})"
    )

    out_root = Path(args.cluster_images_dir)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    export_tasks = []
    for filtered_index, raw_label in enumerate(filtered_cluster_ids):
        cluster_dir = out_root / str(filtered_index)
        cluster_dir.mkdir(exist_ok=True)
        paths = sorted(cluster_to_videos[raw_label])
        if len(paths) > args.max_frames_per_cluster:
            paths = random.sample(paths, args.max_frames_per_cluster)
        for video_path_i in paths:
            frame_idx = random.choice(cluster_frames_by_video[raw_label][video_path_i])
            export_tasks.append((cluster_dir, video_path_i, frame_idx, args.interval_seconds))

    results = Parallel(n_jobs=-1)(
        delayed(_save_one_cluster_example_frame)(task)
        for task in tqdm(export_tasks, desc="Exporting cluster example frames")
    )
    for video_path_i, err in results:
        if err is not None:
            print(f"skip {video_path_i}: {err}")

    model_path = local_model_map(args.model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"embedding model: {args.model_id} ({model_path})")
    print(f"embedding device: {device}")
    processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
    model = AutoModel.from_pretrained(model_path).eval().to(device)

    timestamps, query_embeddings = _embed_video_every_interval(
        video_path=str(video_path),
        model=model,
        processor=processor,
        interval_seconds=args.interval_seconds,
        batch_size=args.batch_size,
        device=device,
    )
    if query_embeddings.shape[0] == 0:
        print("matched_cluster_indices: []")
        print(json.dumps({"matched_cluster_indices": []}))
        return

    predicted_labels = _predict_hdbscan_labels(clusterer, query_embeddings)
    if len(predicted_labels) != len(timestamps):
        raise RuntimeError(
            f"Expected {len(timestamps)} predictions, got {len(predicted_labels)}"
        )

    matched_cluster_indices = []
    seen = set()
    for label in predicted_labels.tolist():
        label = int(label)
        if label == -1 or label not in label_to_filtered_index:
            continue
        filtered_index = label_to_filtered_index[label]
        if filtered_index in seen:
            continue
        seen.add(filtered_index)
        matched_cluster_indices.append(filtered_index)

    print(f"query embeddings: {len(timestamps)} (interval={args.interval_seconds}s)")
    print(f"matched_cluster_indices: {matched_cluster_indices}")
    print(json.dumps({"matched_cluster_indices": matched_cluster_indices}))


if __name__ == "__main__":
    log_buffer = StringIO()
    _original_stdout = sys.stdout
    sys.stdout = _Tee(_original_stdout, log_buffer)
    try:
        main()
    finally:
        sys.stdout = _original_stdout
        Path("./tmp/out.txt").parent.mkdir(parents=True, exist_ok=True)
        with open("./tmp/out.txt", "w") as f:
            f.write(log_buffer.getvalue())
