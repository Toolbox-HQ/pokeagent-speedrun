from s3_utils.s3_sync import init_boto3_client, download_prefix
import sys, os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import json
import glob

def clip_embed_pil(image_pil, model_name: str = "openai/clip-vit-base-patch32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device)
    processor = CLIPImageProcessor.from_pretrained(model_name)
    inputs = processor(images=image_pil.convert("RGB"), return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model(**inputs).image_embeds
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy()

def get_frame_at_index(video_path: str, frame_index: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IndexError(f"Cannot read frame {frame_index} {video_path}")
    return frame

def cosine_search_gpu(embeddings, query):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    E = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    q = torch.as_tensor(query, dtype=torch.float32, device=device)
    E = E / E.norm(dim=1, keepdim=True)
    q = q / q.norm()
    sims = E @ q
    return sims.cpu().numpy()

def load_embeddings_and_metadata(folder_path: str):
    embed_files = sorted(glob.glob(os.path.join(folder_path, "*_embeddings.pt")))
    all_meta, tensors = [], []
    for ef in embed_files:
        base = os.path.basename(ef).rsplit("_embeddings.pt", 1)[0]
        mf = os.path.join(folder_path, f"{base}_metadata.json")
        if not os.path.exists(mf):
            continue
        emb = torch.load(ef, map_location="cpu")
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)
        with open(mf, "r") as f:
            meta = json.load(f)
        tensors.append(emb)
        all_meta.extend(meta)
    out_tensor = torch.cat(tensors, dim=0) if tensors else torch.empty((0, 512), dtype=torch.float32)
    return all_meta, out_tensor

def top_k_runs_threshold(a: np.ndarray, tau: float, k: int = 10):
    m = a >= tau
    if not m.any():
        return []
    p = np.r_[False, m, False]
    d = np.diff(p.astype(int))
    starts = np.where(d == 1)[0]
    ends   = np.where(d == -1)[0] - 1
    lengths = ends - starts + 1
    order = np.argsort(-lengths)
    return [(int(starts[i]), int(ends[i])) for i in order[:k]]

def main():
    BUCKET_NAME = "b4schnei"
    KEY = "pokeagent/internet_data/--v7Jat84PE.mp4"
    LOCAL_VIDEO = f"cache/{KEY}"
    FRAME_INDEX = 190
    EMB_DIR = "cache/embeddings"

    s3 = init_boto3_client()
    download_prefix(bucket=BUCKET_NAME, prefix=KEY, s3=s3)
    frame = get_frame_at_index(LOCAL_VIDEO, FRAME_INDEX)
    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save("cache/original.png")

    # TEMP
    pil_img = Image.open("cache/test1.png")

    query_emb = clip_embed_pil(pil_img)
    meta, E = load_embeddings_and_metadata(EMB_DIR)
    if E.numel() == 0:
        raise ValueError(f"No embeddings found in {EMB_DIR}")

    sims = cosine_search_gpu(E, query_emb)
    threshold = np.quantile(sims, 0.6)
    top_runs = top_k_runs_threshold(sims, threshold, k=10)

    for rank, (start, end) in enumerate(top_runs, start=1):
        for tag, i in (("start", start), ("end", end)):
            m = meta[i]
            match_file_path = "/".join(m["video_path"].split("/")[1:])
            match_frame_index = int(m["sampled_frame_index"])
            download_prefix(bucket=BUCKET_NAME, prefix=match_file_path, s3=s3)
            f = get_frame_at_index("cache/" + match_file_path, match_frame_index)
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).save(f"cache/seq_{rank:02d}_{tag}.png")
        print(f"#{rank} sequence: start={start}, end={end}")

    return top_runs

if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
