import os
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
import torch.distributed as dist
import random

def get_embeddings(model, processor, path: str):
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
    embeds = get_embeddings(model, processor, video_path)
    return embeds

def dot_product_max(db_embeddings, query_embeddings):
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

def find_best_interval(sims, idxs, num_embeds_per_sample, entropy_threshold):
    n = len(sims)
    if n < num_embeds_per_sample:
        return None

    cs = np.concatenate(([0.0], np.cumsum(sims, dtype=float)))
    w = cs[num_embeds_per_sample:] - cs[:-num_embeds_per_sample]
    valid = _rolling_entropy_valid(idxs, num_embeds_per_sample, entropy_threshold)

    cands = [(w[i], i) for i in range(len(w)) if valid[i]]
    if not cands:
        return None

    score, start = max(cands, key=lambda t: t[0])
    end = start + num_embeds_per_sample - 1
    return score, start, end

# top-level
def _process_chunk_p(args):
    sims, idxs, chunk, num_embeds_per_sample, entropy_threshold = args
    out = []
    for s, e in chunk:
        interval = find_best_interval(sims[s:e+1], idxs[s:e+1], num_embeds_per_sample=num_embeds_per_sample, entropy_threshold=entropy_threshold)
        if interval is not None:
            score, local_start, local_end = interval
            global_start = s + local_start
            global_end = s + local_end
            out.append((score, global_start, global_end))
    return out

def get_top_videos_by_score(sims, idxs, meta, num_embeds_per_sample, max_workers, entropy_threshold):
    segments = list(_segments_by_video(meta))
    n = min(max_workers or (os.cpu_count() or 1), len(segments))
    chunk_size = math.ceil(len(segments) / n)
    chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]

    scored = []
    with ProcessPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(_process_chunk_p, (sims, idxs, c, num_embeds_per_sample, entropy_threshold)) for c in chunks]
        for f in as_completed(futs):
            scored.extend(f.result())

    scored.sort(key=lambda t: t[0], reverse=True)
    return [(a, b, score) for score, a, b in scored]

def _rolling_entropy_valid(idxs: np.ndarray, L: int, thr: float) -> np.ndarray:
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

def cosine_search(query_embed, emb_dir: str):
    num_files = sum(1 for entry in os.scandir(emb_dir) if entry.is_file()) // 2
    i = 0
    metadata = []
    similiarity_idxs = np.array([])
    similarity_scores = np.array([])
    while i < num_files:
        metadata_chunk, db_embed_chunk = load_embedding_metadata_pair(f'{emb_dir}/{i}')
        print(f"[RETRIEVAL] Loaded embeddings {i} from disk")
        similarity_scores_chunk, similarity_idxs_chunk = dot_product_max(db_embed_chunk, query_embed)
        del db_embed_chunk
        metadata.extend(metadata_chunk)
        similarity_scores = np.append(similarity_scores, similarity_scores_chunk)
        similiarity_idxs = np.append(similiarity_idxs, similarity_idxs_chunk)
        i += 1
    return similarity_scores, similiarity_idxs, metadata

def get_videos(query_path: str, emb_dir: str, interval_length: int, num_intervals: int, max_vid_len: float = None):
    print(f"[GPU {dist.get_rank()} RETRIEVAL] Begin retrieval process")
    num_embeds_per_sample = interval_length // 2

    query_emb = dino_embeddings_every(query_path)
    print(f"[GPU {dist.get_rank()} RETRIEVAL] Created dino embeddings")

    sims, idxs, meta = cosine_search(query_emb, emb_dir)
    print(f"[GPU {dist.get_rank()} RETRIEVAL] Completed embeddings load")

    top_videos = get_top_videos_by_score(sims, idxs, meta, num_embeds_per_sample, max_workers=8, entropy_threshold=2.5)
    print(f"[GPU {dist.get_rank()} RETRIEVAL] Completed similarity search")

    total_seconds = 0
    videos = []
    i = 0
    j = 0
    while i < len(top_videos) and j < num_intervals:
        (start, _, score) = top_videos[i]

        start_meta = meta[start]
        video_path = start_meta["video_path"]
        fps = float(start_meta["video_fps"])

        curr = start
        while curr < len(meta) and meta[curr]["video_path"] == video_path:
            curr += 1
        video_length = float(meta[curr - 1]["sampled_frame_index"]) / fps
        i += 1
        if max_vid_len and video_length > max_vid_len:
            continue
        
        j += 1
        videos.append({
                "video_path": video_path,
                "score": score
            })
        
        total_seconds += video_length
        
    print(f"[GPU {dist.get_rank()} RETRIEVAL] Hrs: {total_seconds / 3600}")

    return videos


def get_random_videos(emb_dir: str, num_intervals: int):
    """
    Return random videos from the embedding database.
    Each video has score = 0.
    """

    print("[RANDOM RETRIEVAL] Sampling random videos")

    video_paths = set()
    i = 0

    # Load all metadata files (0.json, 1.json, ...)
    while True:
        meta_path = os.path.join(emb_dir, f"{i}.json")
        if not os.path.exists(meta_path):
            break

        with open(meta_path, "r") as f:
            meta_chunk = json.load(f)

        for m in meta_chunk:
            video_paths.add(m["video_path"])

        i += 1

    video_paths = list(video_paths)
    random.shuffle(video_paths)

    videos = [
        {"video_path": vp, "score": 0}
        for vp in video_paths[:num_intervals]
    ]

    return videos

def _iter_db_prefixes(emb_dir: str):
    """
    Yields numeric prefixes (0,1,2,...) for files emb_dir/{i}.pt and emb_dir/{i}.json
    Stops at the first missing json or pt.
    """
    i = 0
    while True:
        pt_path = os.path.join(emb_dir, f"{i}.pt")
        js_path = os.path.join(emb_dir, f"{i}.json")
        if not (os.path.exists(pt_path) and os.path.exists(js_path)):
            break
        yield i
        i += 1

def load_embedding_metadata_pair(prefix: str):
    meta_path = f"{prefix}.json"
    tensor_path = f"{prefix}.pt"
    with open(meta_path, "r") as f:
        all_meta = json.load(f)
    embeddings = torch.load(tensor_path, map_location="cpu")
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.as_tensor(embeddings)
    return all_meta, embeddings

@torch.no_grad()
def best_single_match_search(
    query_video_path: str,
    emb_dir: str,
    top_n: int = 50,
    rng_seed: int | None = None,
):
    """
    RANDOM single-embedding search:

      1) Choose ONE random query embedding q* from the query video.
      2) For each DB embedding e, score = dot(q*, e).
      3) For each video, keep the maximum score across its embeddings.

    Returns:
      List[{"video_path": str, "score": float}] sorted desc by score, length top_n
    """

    # 1) Query embeddings (normalized) -> pick a random one
    query_emb = dino_embeddings_every(query_video_path)  # [Tq, D]
    if query_emb.device.type != "cpu":
        query_emb = query_emb.cpu()
    query_emb = query_emb.float()
    query_emb = F.normalize(query_emb, p=2, dim=1, eps=1e-12)

    Tq = query_emb.shape[0]
    if Tq == 0:
        return []

    rng = random.Random(rng_seed)  # deterministic if seed provided
    q_idx = rng.randrange(Tq)
    q = query_emb[q_idx]  # [D]
    q = F.normalize(q, p=2, dim=0, eps=1e-12)  # extra safety

    # 2) Best score per DB video
    video_scores: dict[str, float] = {}

    # 3) Walk DB chunks
    for i in _iter_db_prefixes(emb_dir):
        meta_chunk, db_emb = load_embedding_metadata_pair(os.path.join(emb_dir, str(i)))

        db_emb = db_emb.float()
        db_emb = F.normalize(db_emb, p=2, dim=1, eps=1e-12)

        # Scores per DB embedding: [Te]
        # dot(q, e_j) for each DB embedding e_j
        scores = db_emb @ q  # [Te]
        scores_np = scores.cpu().numpy()

        # Aggregate max per video_path
        for j, m in enumerate(meta_chunk):
            vp = m["video_path"]
            s = float(scores_np[j])
            prev = video_scores.get(vp)
            if (prev is None) or (s > prev):
                video_scores[vp] = s

        del db_emb, scores

    ranked = sorted(video_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [{"video_path": vp, "score": score} for vp, score in ranked]


def mean_similarity_per_db_embedding(db_embeddings: torch.Tensor, query_embeddings: torch.Tensor):
    """
    db_embeddings: [Te, D] (assumed L2-normalized)
    query_embeddings: [Tq, D] (assumed L2-normalized)

    Returns:
      scores: numpy array [Te] where scores[j] = mean_q dot(query[q], db[j])
    """
    E = db_embeddings
    Q = query_embeddings
    sims = Q @ E.T  # [Tq, Te]
    return sims.mean(dim=0).float().cpu().numpy()

def load_embedding_metadata_pair(prefix):
    meta_path = f"{prefix}.json"
    tensor_path = f"{prefix}.pt"
    with open(meta_path, "r") as f:
        all_meta = json.load(f)
    embeddings = torch.load(tensor_path, map_location="cpu")
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.as_tensor(embeddings)
    return all_meta, embeddings

def mean_search(query_embed: torch.Tensor, emb_dir: str):
    """
    Walks chunked embedding files (0.pt/.json, 1.pt/.json, ...)
    Returns:
      all_scores: np.ndarray [N_total_embeddings]
      metadata: list of metadata entries length N_total_embeddings
    """
    # Count chunks the same way your cosine_search does
    num_files = sum(1 for entry in os.scandir(emb_dir) if entry.is_file()) // 2

    metadata = []
    all_scores = np.array([], dtype=np.float32)

    for i in range(num_files):
        metadata_chunk, db_embed_chunk = load_embedding_metadata_pair(f"{emb_dir}/{i}")
        print(f"[RETRIEVAL] Loaded embeddings {i} from disk")

        # Ensure tensor + normalization
        if not isinstance(db_embed_chunk, torch.Tensor):
            db_embed_chunk = torch.as_tensor(db_embed_chunk)
        db_embed_chunk = db_embed_chunk.float()
        db_embed_chunk = F.normalize(db_embed_chunk, p=2, dim=1, eps=1e-12)

        scores_chunk = mean_similarity_per_db_embedding(db_embed_chunk, query_embed)

        metadata.extend(metadata_chunk)
        all_scores = np.append(all_scores, scores_chunk)

        del db_embed_chunk

    return all_scores, metadata

def score_videos_by_mean(scores: np.ndarray, meta: list):
    """
    Aggregates per-embedding scores into per-video scores by averaging within each video.
    Returns dict: video_path -> mean score
    """
    sums = {}
    counts = {}

    for s, m in zip(scores, meta):
        vp = m["video_path"]
        sums[vp] = sums.get(vp, 0.0) + float(s)
        counts[vp] = counts.get(vp, 0) + 1

    return {vp: (sums[vp] / counts[vp]) for vp in sums.keys()}

def get_videos_mean(
    query_path: str,
    emb_dir: str,
    num_intervals: int,
    max_vid_len: float = None
):
    """
    Same "outer pipeline" feel as get_videos(), but:
      - uses mean over query frames (instead of max)
      - drops rolling entropy / interval selection
      - scores videos by mean similarity across that video's embeddings
    """
    try:
        rank = dist.get_rank()
    except Exception:
        rank = 0

    print(f"[GPU {rank} RETRIEVAL] Begin MEAN retrieval process")

    # Query embeddings (your function already L2-normalizes, but normalize again defensively)
    query_emb = dino_embeddings_every(query_path).float()
    query_emb = F.normalize(query_emb, p=2, dim=1, eps=1e-12)

    print(f"[GPU {rank} RETRIEVAL] Created dino embeddings")

    scores, meta = mean_search(query_emb, emb_dir)
    print(f"[GPU {rank} RETRIEVAL] Completed embeddings load + mean similarity")

    video_scores = score_videos_by_mean(scores, meta)

    # Optional max length filter (same idea as your original)
    # We'll compute an approximate length per video from metadata if available.
    # If you don't want filtering, remove this block.
    if max_vid_len is not None:
        # pick the max sampled_frame_index seen for each video (roughly last sample)
        max_frame = {}
        fps_map = {}
        for m in meta:
            vp = m["video_path"]
            fps_map[vp] = float(m.get("video_fps", 0.0) or 0.0)
            fi = float(m.get("sampled_frame_index", 0.0) or 0.0)
            if fi > max_frame.get(vp, -1.0):
                max_frame[vp] = fi

        filtered = {}
        for vp, sc in video_scores.items():
            fps = fps_map.get(vp, 0.0)
            length = (max_frame.get(vp, 0.0) / fps) if fps > 0 else 0.0
            if length <= max_vid_len:
                filtered[vp] = sc
        video_scores = filtered

    ranked = sorted(video_scores.items(), key=lambda kv: kv[1], reverse=True)[:num_intervals]

    videos = [{"video_path": vp, "score": float(score)} for vp, score in ranked]

    print(f"[GPU {rank} RETRIEVAL] Completed MEAN search (returned {len(videos)})")
    return videos


def main():
    videos = get_videos(".cache/pokeagent/online/query_video/query0.mp4", ".cache/pokeagent/db_embeddings", interval_length=540, num_intervals=400)
    for i, video in enumerate(videos):
        if i > 390:
            print(video["video_path"])
            
if __name__ == "__main__":
    main()
