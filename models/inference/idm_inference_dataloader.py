import os
import json
import torch
import einops
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize
from torchcodec.decoders import VideoDecoder
from models.model.IDM.policy import InverseActionPolicy as IDModel
from emulator.keys import CLASS_TO_KEY, KEY_TO_CLASS
from typing import List
from functools import partial
import orjson 

DEVICE = "cpu"
IDM_FPS = 4
WINDOW = 128  # non-overlapping IDM windows
AGENT_FPS = 2
AGENT_WINDOW = 64

def load_model(chkpt=".cache/pokeagent/rnd_idm_model.pt"):
    m = IDModel()
    m.load_state_dict(torch.load(chkpt, map_location="cpu"))
    m.eval()
    return m

def decode_idm_rate_frames(video_path, start: int, end: int, video_fps, idm_fps: int =IDM_FPS, labels=False):
    stride = max(1, int(round(video_fps / idm_fps)))
    idxs = list(range(int(start), int(end), stride))

    if labels:
        with open(os.path.splitext(video_path)[0]+'.json', "rb") as f:
            actions = torch.tensor([ KEY_TO_CLASS[action["keys"]] for i, action in enumerate(orjson.loads(f.read())) if i in idxs ], dtype=torch.int64)

    dec = VideoDecoder(video_path)
    x = dec.get_frames_at(indices=idxs).data         # (T,C,H,W) RGB                       # spatial size for IDM
    
    return x, actions if labels else x 

class IDMWindowDataset(Dataset):
    def __init__(self, intervals_json, idm_fps=IDM_FPS, window=WINDOW, processor = None):
        
        self.processor = processor

        with open(intervals_json, "r", encoding="utf-8") as f:
            items = json.load(f)
        self.samples = []
        for it in items:
            start = int(it["start"])
            end = int(it["end"])
            fps = float(it["video_fps"])
            stride = max(1, int(round(fps / idm_fps)))
            n_raw = max(0, end - start)
            n_idm = n_raw // stride
            n_full = (n_idm // window) * window
            n_windows = n_full // window
            for w in range(n_windows):
                win_start = start + w * window * stride
                win_end = win_start + window * stride
                self.samples.append({
                    "video_path": it["video_path"],
                    "start": win_start,
                    "end": win_end,
                    "video_fps": fps,
                })

    @staticmethod
    def collate_fn(batch: List[dict]):
        return {
            k: torch.stack([item[k] for item in batch], dim=0)
            for k in batch[0].keys()
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        idm_frames = decode_idm_rate_frames(
            s["video_path"], s["start"], s["end"], s["video_fps"], IDM_FPS
        )

        inputs = None
        if self.processor:
            agent_frames = downsample(idm_frames, 2)
            inputs = self.processor(
            images=agent_frames,
            return_tensors="pt"
            )
            inputs["labels"] = resize(idm_frames, (128, 128))

        return inputs if inputs else idm_frames # (T,C,HW) RGB

class LabelledWindowDataset(IDMWindowDataset):

    def __getitem__(self, idx):
        s = self.samples[idx]
        idm_frames, labels = decode_idm_rate_frames(
            s["video_path"], s["start"], s["end"], s["video_fps"], IDM_FPS, labels=True
        )

        inputs = None
        if self.processor:
            agent_frames = downsample(idm_frames, 2)
            inputs = self.processor(
            images=agent_frames,
            return_tensors="pt"
            )
            inputs["labels"] = resize(idm_frames, (128, 128))
            inputs["ground_labels"] = downsample(labels, 2, offset=1)

        return inputs if inputs else idm_frames

def _to_bgr_u8(x):

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1,3):
            x = x.permute(1,2,0)  # (H,W,C)
        x = x.numpy()
    if x.dtype != np.uint8:
        if x.max() <= 1.0:
            x = (np.clip(x,0,1)*255).astype(np.uint8)
        else:
            x = x.astype(np.uint8)
    if x.ndim == 2:
        x = np.repeat(x[...,None], 3, axis=2)
    if x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    return x[..., ::-1].copy()  # RGB->BGR

def save_mp4_with_keys(x, labels, out_path, fps):
    import cv2
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if len(x) == 0:
        return
    first = _to_bgr_u8(x[0])
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    for f, l in zip(x, labels):
        img = _to_bgr_u8(f)
        text = CLASS_TO_KEY.get(int(l), str(int(l)))
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        writer.write(img)
    writer.release()


def infer_idm_labels(x, idm):
    x_resize = resize(x.clone(), (128, 128))
    frames_thwc = einops.rearrange(x_resize, "t c h w -> t h w c") # (128, 128, 128, 3)

    T = (frames_thwc.shape[0] // WINDOW) * WINDOW
    if T == 0:
        return torch.empty(0, dtype=torch.long)
    f = frames_thwc[:T].to(DEVICE).unsqueeze(0)  # (1, T, H, W, C)
    dummy = {"first": torch.zeros((1,1), device=DEVICE),
             "state_in": idm.initial_state(1)}
    logits = idm({"img": f}, labels=None, **dummy).logits  # (1, T, K)
    return torch.argmax(logits, dim=-1).squeeze(0).cpu()   # (T,)

def get_idm_labeller(device):
    idm = load_model()
    idm.to(device)
    idm.eval()
    return partial(batched_infer_idm_labels, idm=idm)

def batched_infer_idm_labels(x, idm=None):
    with torch.no_grad():
        (B, S, C, H, W) = x.shape
        assert (S, C, H, W) == (128, 3, 128, 128)

        frames_bthwc = einops.rearrange(x, "b t c h w -> b t h w c") # (128, 128, 128, 3)

        dummy = {
            "first": torch.zeros((frames_bthwc.shape[0], 1)).to(frames_bthwc.device),
            "state_in": idm.initial_state(frames_bthwc.shape[0])
        }

        logits = idm({"img": frames_bthwc}, labels=None, **dummy).logits  # (1, T, K)
        labels = torch.argmax(logits, dim=-1)
        labels = labels[:,1::2] # strided downsampling, offset by 1
    return labels



def get_dataloader(intervals_json, batch_size=1, num_workers=0, shuffle=False):
    ds = IDMWindowDataset(intervals_json, IDM_FPS, WINDOW)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

def downsample(x, stride, offset=0):
    idxs = list(range(offset, x.shape[0], stride))
    return x[idxs]

def downsample_batched(x, stride):
    idxs = list(range(0, x.shape[-1], stride))
    assert torch.all(x[:, idxs] == x[:,::stride])
    return x[:,idxs]


def main():
    idm = load_model()
    loader = get_dataloader(".cache/pokeagent/intervals.json", batch_size=1, num_workers=0, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch[0]  # (B,T,C,H,W) RGB

            labels = infer_idm_labels(x, idm)
            agent_frames = downsample(x, 2)
            agent_labels = downsample(labels, 2)

            save_mp4_with_keys(agent_frames, agent_labels, os.path.join(".cache/pokeagent/idm_videos", f"clip_{i:05d}.mp4"), AGENT_FPS)
            if i == 30:
                break

if __name__ == "__main__":
    main()
