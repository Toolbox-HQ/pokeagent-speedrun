import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel
import json
import glob
from torchcodec.decoders import VideoDecoder
import cv2
import math
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from models.util.misc import local_model_map

def process_batch(model, processor, path: str):
    decoder = VideoDecoder(path)
    duration = float(decoder.metadata.duration_seconds)
    t, timestamps = 0, []
    while t < duration:
        timestamps.append(t)
        t += 2
    frames = decoder.get_frames_played_at(seconds=timestamps).data

    inputs = processor(images=frames, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.last_hidden_state[:, 0]  # CLS token
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    return image_embeds

def dino_embeddings_every(video_path: str, model_id: str = "facebook/dinov2-base"):
    model_id = local_model_map(model_id)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval()
    embeds = process_batch(model, processor, video_path)
    return embeds

def dot_product(db_embeddings, query_embeddings):
    E = db_embeddings
    Q = query_embeddings
    sims = Q @ E.T
    return sims.max(dim=0).values.float().numpy(), sims.argmax(dim=0).numpy()

def load_embeddings_and_metadata(folder_path: str):
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
    
    out_tensor = F.normalize(torch.cat(tensors, dim=0), p=2, dim=1, eps=1e-12)
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
        for f in as_completed(futs):
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

def save_clip_between(s_idx: int, e_idx, path, rank):
    dec = VideoDecoder(path)
    fps = getattr(dec, "fps", None) or getattr(dec, "frame_rate", None)

    try:
    
       

        # Open with OpenCV for frame-accurate slicing
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")

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

def load_embedding_metadata_pair(prefix):
    meta_path = f"{prefix}.json"
    tensor_path = f"{prefix}.pt"
    with open(meta_path, "r") as f:
        all_meta = json.load(f)
    embeddings = torch.load(tensor_path)
    return all_meta, embeddings

def cosine_search(query_embed, folder_path: str):
    num_files = sum(1 for entry in os.scandir(folder_path) if entry.is_file()) // 2
    i = 0
    metadata = []
    similiarity_idxs = np.array([])
    similarity_scores = np.array([])
    while i < num_files:
        metadata_chunk, db_embed_chunk = load_embedding_metadata_pair(f'{folder_path}/{i}')
        print(f"[RETRIEVAL] Loaded embeddings {i} from disk")
        similarity_scores_chunk, similarity_idxs_chunk = dot_product(db_embed_chunk, query_embed)
        del db_embed_chunk
        metadata.extend(metadata_chunk)
        similarity_scores = np.append(similarity_scores, similarity_scores_chunk)
        similiarity_idxs = np.append(similiarity_idxs, similarity_idxs_chunk)
        i += 1
    return similarity_scores, similiarity_idxs, metadata


def get_intervals(query_path: str, emb_prefix: str, interval_length: int, num_intervals: int):
    print(f"[RETRIEVAL] Begin retrieval process")
    num_embeds_per_sample = interval_length // 2

    query_emb = dino_embeddings_every(query_path)
    print(f"[RETRIEVAL] Created dino embeddings")

    sims, idxs, meta = cosine_search(query_emb, emb_prefix)
    print(f"[RETRIEVAL] Completed embeddings load")

    top_runs = find_top_runs_concurrent(sims, idxs, meta, L=num_embeds_per_sample, threshold=num_intervals, max_workers=8)
    print(f"[RETRIEVAL] Completed similarity search")

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
    
    print(f"[RETRIEVAL] Hrs: {total_seconds / 3600}")
    return results, query_emb

def main():
    intervals = get_intervals(".cache/pokeagent/online/query_video/query0.mp4", ".cache/pokeagent/db_embeddings", interval_length=540, num_intervals=400)
    for i, interval in enumerate(intervals):
        if i > 390:
            save_clip_between(interval["start"], interval["end"], interval['video_path'], i)

if __name__ == "__main__":
    main()
