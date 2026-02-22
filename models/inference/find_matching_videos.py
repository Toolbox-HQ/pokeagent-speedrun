import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "96")
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel
import json
import glob
from torchcodec.decoders import VideoDecoder
import math
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from models.util.misc import local_model_map
import torch.distributed as dist

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
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id).eval()
    embeds = get_embeddings(model, processor, video_path)
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
        similarity_scores_chunk, similarity_idxs_chunk = dot_product(db_embed_chunk, query_embed)
        del db_embed_chunk
        metadata.extend(metadata_chunk)
        similarity_scores = np.append(similarity_scores, similarity_scores_chunk)
        similiarity_idxs = np.append(similiarity_idxs, similarity_idxs_chunk)
        i += 1
    return similarity_scores, similiarity_idxs, metadata


def cosine_search_with_embeddings(query_embed, emb_dir: str):
    """
    Like cosine_search, but also returns a single concatenated tensor of all
    database embeddings in the same global index order as the returned metadata.
    This avoids re-reading the massive embedding shards from disk when we need
    per-video embeddings. Uses a pre-allocated buffer and in-place copies to
    avoid a full 100GB+ copy from torch.cat.
    Only rank 0 allocates the full embedding tensor; other ranks use a placeholder.
    query_embed is gathered and concatenated across all ranks before search.
    """
    gather_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gather_list, query_embed)
    query_embed = torch.cat([t for t in gather_list], dim=0)

    rank = dist.get_rank()
    num_files = sum(1 for entry in os.scandir(emb_dir) if entry.is_file()) // 2
    total_rows = 0
    for i in range(num_files):
        with open(f"{emb_dir}/{i}.json", "r") as f:
            total_rows += len(json.load(f))

    metadata = []
    similarity_scores_list = []
    similarity_idxs_list = []
    offset = 0
    all_embeddings = None

    for i in range(num_files):
        metadata_chunk, db_embed_chunk = load_embedding_metadata_pair(f"{emb_dir}/{i}")
        print(f"[RETRIEVAL] Loaded embeddings {i} from disk")

        if rank == 0:
            if all_embeddings is None and total_rows > 0:
                embed_dim = db_embed_chunk.shape[1]
                all_embeddings = torch.empty(total_rows, embed_dim, dtype=torch.float32)
        else:
            if all_embeddings is None:
                all_embeddings = torch.empty(0)

        similarity_scores_chunk, similarity_idxs_chunk = dot_product(db_embed_chunk, query_embed)
        metadata.extend(metadata_chunk)
        similarity_scores_list.append(similarity_scores_chunk)
        similarity_idxs_list.append(similarity_idxs_chunk)
        n = db_embed_chunk.shape[0]
        if rank == 0 and all_embeddings.numel() > 0:
            all_embeddings[offset:offset + n].copy_(db_embed_chunk.float())
        offset += n
        del db_embed_chunk

    if all_embeddings is None:
        all_embeddings = torch.empty(0)

    similarity_scores = np.concatenate(similarity_scores_list) if similarity_scores_list else np.array([])
    similarity_idxs = np.concatenate(similarity_idxs_list) if similarity_idxs_list else np.array([])

    return similarity_scores, similarity_idxs, metadata, all_embeddings

def get_videos(query_path: str, emb_dir: str, interval_length: int, num_intervals: int, max_vid_len: float = None):
    print(f"[GPU {dist.get_rank()} RETRIEVAL] Begin retrieval process")
    num_embeds_per_sample = interval_length // 2

    query_emb = dino_embeddings_every(query_path)
    print(f"[GPU {dist.get_rank()} RETRIEVAL] Created dino embeddings")

    self_sim_matrix = (query_emb @ query_emb.T)
    self_similarity = self_sim_matrix.mean().item()
    print(f"[GPU {dist.get_rank()} RETRIEVAL] had self-similarity of {self_similarity}")
    gather_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gather_list, self_similarity)
    world_idx = torch.argmin(torch.tensor(gather_list, dtype=torch.float32)).item()
    print(f"[GPU {dist.get_rank()} RETRIEVAL] GPU {world_idx} has the lowest self-similarity from {gather_list} on {self_sim_matrix.size()} matrix")
    
    top_videos = self_sim_matrix.topk(k=num_intervals, dim=1)[1][0]
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

    return videos, world_idx


def get_videos_with_embeddings(query_path: str, emb_dir: str, interval_length: int, num_intervals: int, max_vid_len: float = None):
    print(f"[GPU RETRIEVAL] Begin retrieval process")
    num_embeds_per_sample = interval_length // 2

    query_emb = dino_embeddings_every(query_path)
    print(f"[GPU RETRIEVAL] Created dino embeddings")

    self_sim_matrix = (query_emb @ query_emb.T)
    self_similarity = self_sim_matrix.mean().item()
    print(f"[GPU RETRIEVAL] had self-similarity of {self_similarity}")
    gather_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gather_list, self_similarity)
    world_idx = torch.argmin(torch.tensor(gather_list, dtype=torch.float32)).item()
    print(f"[GPU RETRIEVAL] GPU {world_idx} has the lowest self-similarity from {gather_list} on {self_sim_matrix.size()} matrix")
    
    top_videos = self_sim_matrix.topk(k=num_intervals, dim=1)[1][0]
    sims, idxs, meta, all_embeddings = cosine_search_with_embeddings(query_emb, emb_dir)
    print(f"[GPU RETRIEVAL] Completed embeddings load")

    top_videos = get_top_videos_by_score(sims, idxs, meta, num_embeds_per_sample, max_workers=8, entropy_threshold=2.5)
    print(f"[GPU RETRIEVAL] Completed similarity search")

    rank = dist.get_rank()
    if rank == 0:
        total_seconds = 0
        videos = []
        video_embeddings = []
        i = 0
        j = 0
        while i < len(top_videos) and j < num_intervals:
            (start, _, score) = top_videos[i]

            start_meta = meta[start]
            video_path = start_meta["video_path"]
            fps = float(start_meta["video_fps"])

            vid_start = start
            while vid_start > 0 and meta[vid_start - 1]["video_path"] == video_path:
                vid_start -= 1

            curr = start
            while curr < len(meta) and meta[curr]["video_path"] == video_path:
                curr += 1
            video_length = float(meta[curr - 1]["sampled_frame_index"]) / fps
            i += 1
            if max_vid_len and video_length > max_vid_len:
                continue

            j += 1
            video_embeddings.append(all_embeddings[vid_start:curr].clone())
            videos.append({
                    "video_path": video_path,
                    "score": score
                })

            total_seconds += video_length

        print(f"[GPU 0 RETRIEVAL] Hrs: {total_seconds / 3600}")
    else:
        videos = []
        video_embeddings = []

    obj_list = [videos, video_embeddings]
    dist.broadcast_object_list(obj_list, src=0)
    videos, video_embeddings = obj_list[0], obj_list[1]

    return videos, video_embeddings, world_idx


def main():
    
    """
    Small helper for debugging retrieval.

    Usage:
        python -m models.inference.find_matching_videos /path/to/query_video.mp4

    This uses the same parameters as the online agent:
      - emb_dir: '.cache/pokeagent/db_embeddings'
      - interval_length: 540 (match_length)
      - num_intervals: 100 (retrieved_videos)
      - max_vid_len: None
    """
    
    # Initialize distributed process group if not already initialized
    if not dist.is_initialized():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # Use environment variables if available
            from models.util.dist import init_distributed
            init_distributed()
        else:
            # Initialize as single-process group for standalone execution
            # Use file-based initialization for single process
            import tempfile
            init_file = os.path.join(tempfile.gettempdir(), f"dist_init_{os.getpid()}")
            dist.init_process_group(
                backend="gloo" if not torch.cuda.is_available() else "nccl",
                init_method=f"file://{init_file}",
                rank=0,
                world_size=1,
            )
    
    # python models/inference/find_matching_videos.py --query_path ./tmp/0.mp4
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-path",
        "--query_path",
        dest="query_path",
        type=str,
        required=True,
        help="Path to the query video (.mp4) to use for retrieval.",
    )
    args = parser.parse_args()

    videos, video_embeddings, _ = get_videos_with_embeddings(
        args.query_path,
        ".cache/pokeagent/db_embeddings",
        interval_length=540,
        num_intervals=100,
        max_vid_len=None,
    )
    embs = torch.cat(video_embeddings, dim=0).numpy()
    del video_embeddings

    cluster_labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(embs)
    import resource
    import psutil

    process = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_mb = process.ru_maxrss / 1024.0
    current_rss_mb = psutil.Process().memory_info().rss / (1024.0 ** 2)
    print(f"\n[Memory Usage] Process RSS - Current: {current_rss_mb:.2f} MB, Peak: {peak_rss_mb:.2f} MB")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / (1024.0 ** 3)
            res = torch.cuda.memory_reserved(i) / (1024.0 ** 3)
            max_alloc = torch.cuda.max_memory_allocated(i) / (1024.0 ** 3)
            max_res = torch.cuda.max_memory_reserved(i) / (1024.0 ** 3)
            print(f"[Memory Usage] GPU {i} - Allocated: {alloc:.2f} GB (peak {max_alloc:.2f}), Reserved: {res:.2f} GB (peak {max_res:.2f})")


if __name__ == "__main__":
    main()
