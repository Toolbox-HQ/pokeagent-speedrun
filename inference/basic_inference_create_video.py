import torch
from dataset.idm import IDMDataset
from model.IDM.policy import InverseActionPolicy as IDModel
from policy.policy import CLASS_TO_KEY
from util.data import save_json
import os
from tqdm import tqdm
import cv2
import numpy as np

INFERENCE_FPS = 4
SAMPLE_LENGTH = 128
H = 128
W = 128
DEVICE = "cuda:0"

def load_model(chkpt=".cache/pokeagent/rnd_idm_model.pt") -> torch.nn.Module:
    model = IDModel()
    model.load_state_dict(torch.load(chkpt, map_location="cpu"))
    return model

def hwc_rgb_to_bgr_uint8(frame_hwc):
    if isinstance(frame_hwc, torch.Tensor):
        x = frame_hwc.detach().cpu()
        if x.dtype != torch.uint8:
            x = (x.clamp(0, 1) * 255).round().to(torch.uint8)
        arr = x.numpy()
    else:
        arr = frame_hwc
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(arr[:, :, ::-1])  # BGR

def main(model, video):
    log = []

    frames_idx = IDMDataset.process_video(video, INFERENCE_FPS, SAMPLE_LENGTH)

    out_dir = os.path.dirname(video)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "keys_overlay.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, INFERENCE_FPS, (W, H))

    with torch.no_grad():
        for sample in tqdm(frames_idx):
            # get_frames returns [T, H, W, C] (RGB, float [0,1] or uint8)
            frames_hwc = IDMDataset.get_frames(video, sample, H, W, is_val=True)  # [T,H,W,C]

            # Model expects [B, T, H, W, C]
            if isinstance(frames_hwc, torch.Tensor):
                inp_img = frames_hwc.unsqueeze(0)  # [1,T,H,W,C]
            else:
                inp_img = torch.from_numpy(frames_hwc).unsqueeze(0)  # [1,T,H,W,C]

            dummy = {
                "first": torch.zeros((inp_img.shape[0], 1)).to(DEVICE),
                "state_in": model.initial_state(inp_img.shape[0]),
            }
            print(inp_img.shape)
            print(inp_img.dtype)
            inp = {"img": inp_img.to(device=DEVICE)}

            out = model(inp, labels=None, **dummy)
            preds = torch.argmax(out.logits, dim=-1).squeeze(0).cpu()  # [T]

            for i in range(len(sample)):
                key_text = CLASS_TO_KEY[preds[i].item()]
                log.append({"frame": sample[i], "keys": key_text})

                frame_bgr = hwc_rgb_to_bgr_uint8(frames_hwc[i])
                cv2.putText(frame_bgr, key_text, (8, H - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(frame_bgr)

    writer.release()
    save_json(os.path.join(out_dir, "keys.json"), log)

if __name__ == "__main__":
    import sys
    video = sys.argv[1]
    model = load_model()
    model.to(device=DEVICE)
    model.eval()
    main(model, video)
