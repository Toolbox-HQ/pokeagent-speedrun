from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from torchcodec.decoders import VideoDecoder
import torch
import torch.nn.functional as F
import os
import json
import glob

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

def get_video_metadata():
    EMB_DIR = ".cache/clip32"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    query_embedding_tensor = _clip_embeddings_every(".cache/query.mp4", device=device)
    metadata_list, data_embedding_tensor = _load_embeddings_and_metadata(EMB_DIR, device=device)
    similarity_tensor = _multi_cosine_search_gpu(data_embedding_tensor, query_embedding_tensor)
    meta_similarity_pairs = _get_meta_similarity_pairs(metadata_list, similarity_tensor, data_embedding_tensor)
    