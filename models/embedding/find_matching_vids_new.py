from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from torchcodec.decoders import VideoDecoder
import torch
import torch.nn.functional as F
import os
import json
import glob
import numpy as np

def _clip_embeddings_every(video_path: str, interval_s: float = 2.0, model_id: str = "openai/clip-vit-base-patch32", device=None):
    decoder = VideoDecoder(video_path)
    duration = float(decoder.metadata.duration_seconds)

    t, timestamps = 0.0, []
    while t < duration:
        timestamps.append(round(t, 3))
        t += interval_s

    processor = CLIPImageProcessor.from_pretrained(model_id)
    model = CLIPVisionModelWithProjection.from_pretrained(model_id).to(device).eval()

    with torch.no_grad():
        frames = decoder.get_frames_played_at(seconds=timestamps).data  # (N, C, H, W) uint8
        imgs = frames.permute(0, 2, 3, 1).cpu().numpy()                 # (N, H, W, C) uint8
        inputs = processor(images=list(imgs), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        embeds = model(**inputs).image_embeds.float()                    # (N, D) on device
        embeds = F.normalize(embeds, p=2, dim=1, eps=1e-12).to(device)   # norm + device here

    return embeds

def _multi_cosine_search_gpu(db_embeddings, query_embeddings):
    E = db_embeddings
    Q = query_embeddings
    sims = Q @ E.T
    return sims.max(dim=0).values.float().cpu().numpy()

def _load_embeddings_and_metadata(folder_path: str, device):
    embed_files = sorted(glob.glob(os.path.join(folder_path, "*.pt")))
    all_meta, tensors = [], []
    for ef in embed_files:
        base = os.path.basename(ef).rsplit(".pt", 1)[0]
        mf = os.path.join(folder_path, f"{base}.json")
        if not os.path.exists(mf):
            continue
        emb = torch.load(ef, map_location="cpu")
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)
        tensors.append(emb.float())
        with open(mf, "r") as f:
            meta = json.load(f)
        all_meta.append(meta)

    out_tensor = torch.cat(tensors, dim=0) if tensors else torch.empty((0, 512), dtype=torch.float32)
    if out_tensor.numel() > 0:
        out_tensor = F.normalize(out_tensor, p=2, dim=1, eps=1e-12).to(device)  # norm + device here
    return all_meta, out_tensor

def _get_meta_similarity_pairs(metadata_list, similarity_tensor, data_embedding_tensor):
    idx = 0
    meta_similarity_pairs = []
    for metadata in metadata_list:
        meta_similarity_pairs.append((metadata, similarity_tensor[idx : idx + len(metadata)], data_embedding_tensor[idx : idx + len(metadata)]))
        idx += len(metadata)
    return meta_similarity_pairs

# def _get_runs_from_meta_similarity_pair(metadata, similarity_arr, data_embedding_tensor, embeds_per_run):
#     n = len(similarity_arr)
#     if n == 0 or embeds_per_run <= 0 or embeds_per_run > n:
#         return []

#     # score for every possible run of length embeds_per_run
#     window = np.ones(embeds_per_run, dtype=similarity_arr.dtype)
#     scores = np.convolve(similarity_arr, window, mode="valid")  # len = n - embeds_per_run + 1

#     # greedy: pick highest scoring non-overlapping windows
#     sorted_starts = np.argsort(scores)[::-1]
#     used = np.zeros(n, dtype=bool)
#     runs = []

#     for start in sorted_starts:
#         end = start + embeds_per_run  # end index is exclusive
#         if not used[start:end].any():
#             runs.append((scores[start], metadata[start]['video_path'], metadata[start]['sampled_frame_index'], metadata[end - 1]['sampled_frame_index']))
#             used[start:end] = True

#     return runs


def _get_runs_from_meta_similarity_pair(metadata, similarity_arr, data_embedding_tensor, embeds_per_run):
    n = len(similarity_arr)
    if n == 0 or embeds_per_run <= 0 or embeds_per_run > n:
        return []

    # --- entropy-based filtering setup ---
    with torch.no_grad():
        # dot product of embeddings with themselves
        sim_mat = data_embedding_tensor @ data_embedding_tensor.T  # (n, n)
        # remove diagonal self-similarities
        diag = torch.diag(sim_mat)
        sim_mat = sim_mat - torch.diag(diag)
        # argmax over off-diagonal similarities -> list of indices
        nn_indices = sim_mat.argmax(dim=1).cpu().numpy()  # shape (n,)

    num_windows = n - embeds_per_run + 1

    # similarity-based score for every possible run
    window = np.ones(embeds_per_run, dtype=similarity_arr.dtype)
    scores = np.convolve(similarity_arr, window, mode="valid")  # len = num_windows

    # compute an entropy-like measure per window and mark valid ones
    entropy_threshold = 0.8  # fraction of unique nn_indices required in a window
    valid_mask = np.zeros(num_windows, dtype=bool)

    for start in range(num_windows):
        end = start + embeds_per_run
        window_nn = nn_indices[start:end]
        unique_ratio = np.unique(window_nn).size / window_nn.size
        if unique_ratio >= entropy_threshold:
            valid_mask[start] = True  # high-entropy interval, keep it

    # keep only high-entropy windows for scoring / selection
    valid_starts = np.where(valid_mask)[0]
    if valid_starts.size == 0:
        return []

    valid_scores = scores[valid_starts]
    sorted_valid = np.argsort(valid_scores)[::-1]  # indices into valid_starts

    used = np.zeros(n, dtype=bool)
    runs = []

    # greedy: pick highest-scoring non-overlapping high-entropy windows
    for idx in sorted_valid:
        start = valid_starts[idx]
        end = start + embeds_per_run
        if used[start:end].any():
            continue

        runs.append((
            scores[start],
            metadata[start]["video_path"],
            metadata[start]["sampled_frame_index"],
            metadata[end - 1]["sampled_frame_index"],
        ))
        used[start:end] = True

    return runs

    
def _get_top_k_runs(meta_similarity_pairs, embeds_per_run, k):
    runs = []
    for meta_similarity_pair in meta_similarity_pairs:
        runs.extend(_get_runs_from_meta_similarity_pair(*meta_similarity_pair, embeds_per_run))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[:k]

def get_video_metadata():
    EMB_DIR = ".cache/clip32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_embedding_tensor = _clip_embeddings_every(".cache/query.mp4", device=device)
    metadata_list, data_embedding_tensor = _load_embeddings_and_metadata(EMB_DIR, device=device)
    similarity_tensor_arr = _multi_cosine_search_gpu(data_embedding_tensor, query_embedding_tensor)
    meta_similarity_pairs = _get_meta_similarity_pairs(metadata_list, similarity_tensor_arr, data_embedding_tensor)
    top_runs = _get_top_k_runs(meta_similarity_pairs, 150, 10)
    print(top_runs[0][1])

if __name__ == "__main__":
    get_video_metadata()

    