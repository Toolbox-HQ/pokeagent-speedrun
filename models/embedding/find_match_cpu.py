from s3_utils.s3_sync import init_boto3_client, download_prefix, upload_to_s3
import sys
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import json
import glob
from torchcodec.decoders import VideoDecoder
import subprocess
import time
import cv2
import math
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List
from tqdm import tqdm

def prcoess_batch(model, processor, path: str, device=None):
    decoder = VideoDecoder(path)
    duration = float(decoder.metadata.duration_seconds)
    t, timestamps = 0.0, []
    while t <= duration + 1e-6:
        timestamps.append(round(t, 3))
        t += 2
    frames = decoder.get_frames_played_at(seconds=timestamps).data

    inputs = processor(images=frames, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        image_embeds = outputs.last_hidden_state[:, 0]  # CLS token
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    return image_embeds

def dino_embeddings_every(video_path: str, model_id: str = "facebook/dinov2-base", device=None):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    embeds = prcoess_batch(model, processor, video_path, device)
    return embeds

def multi_cosine_search(db_embeddings, query_result):
    query_embeddings = query_result
    E = db_embeddings
    Q = query_embeddings
    sims = Q @ E.T
    return sims.max(dim=0).values.float().cpu().numpy(), sims.argmax(dim=0).cpu().numpy()

def load_embeddings_and_metadata(folder_path: str, device):
    embed_files = sorted(glob.glob(os.path.join(folder_path, "*.pt")))
    all_meta, tensors = [], []

    bad_count = 0
    for ef in tqdm(embed_files):
        base = os.path.basename(ef).rsplit(".pt", 1)[0]
        mf = os.path.join(folder_path, f"{base}.json")
        if not os.path.exists(mf):
            continue
        emb = torch.load(ef, map_location="cpu")
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)
        try:
            with open(mf, "r") as f:
                meta = json.load(f)
            tensors.append(emb.float())
            all_meta.extend(meta)
        except:
            #raise ValueError(f'Error: {mf} did not load correctly')
            bad_count += 1
        
    print(f'BAD: {bad_count} -- investigate')
    
    out_tensor = F.normalize(torch.cat(tensors, dim=0), p=2, dim=1, eps=1e-12).to(device)
    return all_meta, out_tensor

def _segments_by_video(meta):
    segs = []
    if not meta:
        return segs
    start = 0
    cur = meta[0]["video_path"]
    for i in range(1, len(meta)):
        if meta[i]["video_path"] != cur:
            segs.append((start, i - 1))
            start = i
            cur = meta[i]["video_path"]
    segs.append((start, len(meta) - 1))
    return segs

def top_runs_greedy(a, idxs, L=30, entropy_threshold=0.7):
    n = len(a)
    if n < L: return []
    cs = np.concatenate(([0.0], np.cumsum(a, dtype=float)))
    w = cs[L:] - cs[:-L]
    valid = _rolling_entropy_valid(idxs, L, entropy_threshold)

    cands = [(w[i], i) for i in range(len(w)) if valid[i]]
    cands.sort(key=lambda t: t[0], reverse=True)

    used = np.zeros(n, dtype=bool)
    res = []
    for score, start in cands:
        s, e = start, start + L - 1
        if not used[s:e+1].any():
            used[s:e+1] = True
            res.append((score, s, e))
    res.sort(key=lambda x: x[1])
    return res


# top-level
def _process_chunk_p(args):
    sims, idxs, chunk, L = args
    out = []
    for s, e in chunk:
        local = top_runs_greedy(sims[s:e+1], idxs[s:e+1], L=L, entropy_threshold=2.5)
        out.extend((score / L, a + s, b + s) for score, a, b in local)
    return out

def find_top_runs_concurrent(sims, idxs, meta, L=150, threshold=240, max_workers=None):
    segments = list(_segments_by_video(meta))
    if not segments:
        return []
    n = min(max_workers or (os.cpu_count() or 1), len(segments))
    chunk_size = math.ceil(len(segments) / n)
    chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]

    scored = []
    with ProcessPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(_process_chunk_p, (sims, idxs, c, L)) for c in chunks]
        for f in tqdm(as_completed(futs), total=len(futs)):
            scored.extend(f.result())

    scored.sort(key=lambda t: t[0], reverse=True)
    scored = scored[:threshold]
    return [(a, b, score) for score, a, b in scored]

def _rolling_entropy_valid(idxs: np.ndarray, L: int, thr: float) -> np.ndarray:
    # H = log L - (1/L) * sum_i c_i log c_i ; maintain S = sum c log c
    n = len(idxs)
    m = n - L + 1
    if m <= 0:
        return np.zeros(0, dtype=bool)

    counts = {}
    S = 0.0
    for x in idxs[:L]:
        c0 = counts.get(x, 0)
        if c0: S -= c0 * math.log(c0)
        c1 = c0 + 1
        counts[x] = c1
        S += c1 * math.log(c1)

    logL = math.log(L)
    valid = np.empty(m, dtype=bool)
    valid[0] = (logL - S / L) >= thr

    for start in range(1, m):
        out_x = idxs[start - 1]
        c0 = counts[out_x]
        S -= c0 * math.log(c0)
        if c0 == 1:
            del counts[out_x]
        else:
            c1 = c0 - 1
            counts[out_x] = c1
            S += c1 * math.log(c1)

        in_x = idxs[start + L - 1]
        c0 = counts.get(in_x, 0)
        if c0:
            S -= c0 * math.log(c0)
        c1 = c0 + 1
        counts[in_x] = c1
        S += c1 * math.log(c1)

        valid[start] = (logL - S / L) >= thr

    return valid

def save_clip_between(start_meta, end_meta, BUCKET_NAME, rank, s3):
    match_file_path = "/".join(start_meta["video_path"].split("/")[1:])
    download_prefix(bucket=BUCKET_NAME, prefix=match_file_path, s3=s3)
    local_path = start_meta["video_path"]

    dec = VideoDecoder(local_path)
    fps = getattr(dec, "fps", None) or getattr(dec, "frame_rate", None)

    try:
        s_idx = int(start_meta["sampled_frame_index"])
        e_idx = int(end_meta["sampled_frame_index"])
        if e_idx <= s_idx:
            e_idx = s_idx + 1

        # Open with OpenCV for frame-accurate slicing
        cap = cv2.VideoCapture(local_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {local_path}")

        # Prefer decoder FPS; fall back to container FPS
        fps_cv = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps_out = float(fps or fps_cv or 30.0)

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, s_idx)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = f".cache/pokeagent/dino_vid_matches_temp/seq_{rank:02d}.mp4"

        # Use mp4v for broad compatibility (no external ffmpeg call)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_out, (w, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open writer for: {out_path}")

        frames_to_write = max(e_idx - s_idx, 1)
        written = 0
        while written < frames_to_write:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1

        writer.release()
        cap.release()
        return out_path
    finally:
        close_fn = getattr(dec, "close", None)
        if callable(close_fn):
            close_fn()

def entropy_of_range(idxs: np.ndarray, start: int, end: int) -> float:
    vals, counts = np.unique(idxs[start:end+1], return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())

def main():
    BUCKET_NAME = "b4schnei"
    EMB_DIR = ".cache/pokeagent/dinov2"
    device = "cpu"

    s3 = init_boto3_client()
    query_emb = dino_embeddings_every(".cache/pokeagent/runs/output.mp4", device=device)
    meta, E = load_embeddings_and_metadata(EMB_DIR, device=device)
    sims, idxs = multi_cosine_search(E, query_emb)
    top_runs = find_top_runs_concurrent(sims, idxs, meta, L=270, threshold=400, max_workers=8)

    total_seconds = 0
    results = []
    for i, (start, end, _) in enumerate(top_runs):
        start_meta = meta[start]
        end_meta = meta[end]
        results.append({
            "start": int(start_meta.get("sampled_frame_index", start)),
            "end": int(end_meta.get("sampled_frame_index", end)),
            "video_path": start_meta.get("video_path", ""),
            "video_fps": float(start_meta.get("video_fps", start_meta.get("fps", 0.0))),
        })
        total_seconds += ((end_meta.get("sampled_frame_index", end) - start_meta.get("sampled_frame_index", start)) / start_meta.get("video_fps", start_meta.get("fps", 0.0)))

        if i > 200 and i < 210 or i > 390:
            save_clip_between(start_meta, end_meta, BUCKET_NAME, i, s3)

    print(f"hrs: {total_seconds / 3600}")
    with open("results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
