import torch
from torch.utils.data import Dataset, DataLoader
from torchcodec.decoders import VideoDecoder
from model.IDM.policy import InverseActionPolicy as IDModel
from policy.policy import CLASS_TO_KEY
from torchvision.transforms.functional import resize, to_pil_image, pil_to_tensor
from torchvision.io import write_video
from PIL import ImageDraw
import einops
import json
import os
import cv2
import numpy as np

DEVICE = "cuda:0"

class AgentDataset(Dataset):

    def __init__(self, items, idm_fps, agent_fps, idm_sample_size, agent_sample_size):
        agent_seconds = agent_sample_size / agent_fps
        idm_seconds = idm_sample_size / idm_fps
        idm_batches = agent_seconds / idm_seconds
        idm_interval = idm_batches * idm_sample_size + idm_sample_size / 2.0

        self.intervals = []
        for it in items:
            vfps = it["video_fps"]
            stride = max(1, int(round(vfps / idm_fps)))
            clip_len = max(1, int(round(idm_interval * stride)))
            pad_q = int(round((idm_sample_size / 4.0) * stride))

            start = int(it["start"]) - pad_q
            end = int(it["end"]) + pad_q
            idx = start
            step = max(1, clip_len - 64 * stride)
            while idx < end:
                self.intervals.append((it["video_path"], int(it["video_total_frames"]), int(idx), int(idx + clip_len), int(stride)))
                idx += step

    def __len__(self):
        return len(self.intervals)

    def __getitem__(self, idx):
        interval = self.intervals[idx]
        decoder = VideoDecoder(interval[0])
        start = max(0, interval[2])
        end = min(interval[1], interval[3])
        indices = list(range(start, end, interval[4]))
        tensor = decoder.get_frames_at(indices = indices).data
        startdiff = start - interval[2]
        endiff = interval[3] - end
        start_padd = torch.zeros((startdiff, *tensor.shape[1:]))
        end_padd = torch.zeros((endiff, *tensor.shape[1:]))
        return torch.cat([start_padd, tensor, end_padd], dim=0)

def infer_labels(frames, idm, device):
    frames = resize(frames.clone(), (128, 128), antialias=True)  # (T, C, 128, 128)
    frames = einops.rearrange(frames, "B C H W -> B H W C")      # (T, 128, 128, C)
    labels = torch.zeros(frames.shape[0], dtype=torch.long).cpu()
    idx = 0
    while idx + 128 <= frames.shape[0]:
        x = frames[idx: idx + 128]                                # (128, H, W, C)
        x = torch.unsqueeze(x, dim=0)                             # (1, 128, H, W, C)
        dummy = {
            "first": torch.zeros((x.shape[0], 1), device=device),
            "state_in": idm.initial_state(x.shape[0]),
        }
        inp = {"img": x.to(device=device)}
        out = idm(inp, labels=None, **dummy)
        logits = out.logits                                       # (1, 128, num_classes)
        preds = torch.argmax(logits, dim=-1).squeeze(0).cpu()     # (128,)
        labels[idx: idx + 64] = preds[32:96]
        idx += 64
    return labels


def process_video_clip(frames, idm, device, idm_fps, agent_fps):
    labels = infer_labels(frames, idm, device)
    labels = labels[32:-32]
    frames = frames[32:-32]  # (T, C, H, W)

    stride = max(1, int(round(idm_fps / agent_fps)))
    T = (frames.shape[0] // stride) * stride
    if T == 0:
        return frames[:0], labels[:0]  # empty

    frames = frames[:T]
    labels = labels[:T]
    frames_ds = frames[::stride]
    labels_windows = labels.view(-1, stride)
    labels_ds = labels_windows.mode(dim=1).values

    return frames_ds, labels_ds


def load_model(chkpt=".cache/pokeagent/rnd_idm_model.pt") -> torch.nn.Module:
    model = IDModel()
    state = torch.load(chkpt, map_location="cpu")
    model.load_state_dict(state)
    return model

def save_clip_with_actions(frames, labels, out_path, fps):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def to_bgr_u8(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            if x.ndim == 3 and x.shape[0] in (1, 3):
                x = x.permute(1, 2, 0)  # (H, W, C)
            x = x.numpy()
        if x.dtype != np.uint8:
            if x.max() <= 1.0:
                x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
            else:
                x = x.astype(np.uint8)
        if x.ndim == 2:
            x = np.repeat(x[..., None], 3, axis=2)
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        return x[..., ::-1].copy()  # RGB->BGR

    first = to_bgr_u8(frames[0])
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h)
    )

    for f, l in zip(frames, labels):
        img = to_bgr_u8(f)
        text = CLASS_TO_KEY.get(int(l), str(int(l)))
        cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(img)

    writer.release()

def main():
    model = load_model()
    model.to(device=DEVICE)
    model.eval()

    with open(".cache/pokeagent/intervals.json", "r", encoding="utf-8") as f:
        items = json.load(f)

    idm_fps = 4
    agent_fps = 1
    dataset = AgentDataset(items, idm_fps=idm_fps, agent_fps=agent_fps, idm_sample_size=128, agent_sample_size=64)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, clip in enumerate(loader):
            if i == 0 or i == 1:
                frames = clip[0]  # (T, C, H, W)
                frames, labels = process_video_clip(frames, model, DEVICE, idm_fps, agent_fps)
                if frames.shape[0] == 0:
                    continue
                out_path = f"clip_{i:05d}.mp4"
                save_clip_with_actions(frames, labels, out_path, fps=agent_fps)
                break

if __name__ == "__main__":
    main()