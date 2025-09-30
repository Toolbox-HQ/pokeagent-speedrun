#!/usr/bin/env python3
"""
Downsample a video to --fps and label each output frame with the last
non-empty key pressed within that interval based on keys.json.

Output JSON schema (one entry per *output* frame index):
[
  {"frame": 0, "keys": []},             # or ["a"], ["b"], etc. (at most 1 entry)
  ...
]

Valid keys: a, b, start, select, up, down, left, right
"""

import argparse
import json
import os
from typing import List, Tuple, Optional
from tqdm import tqdm
import cv2
import numpy as np

VALID_KEYS = {"a", "b", "start", "select", "up", "down", "left", "right"}

def load_keys_json(path: str) -> List[Tuple[int, str]]:
    """
    Returns sorted list of (frame_index, first_key_or_empty_string).
    Unknown keys are treated as empty.
    """
    with open(path, "r") as f:
        entries = json.load(f)
    events = []
    for e in entries:
        if not isinstance(e, dict) or "frame" not in e or "keys" not in e:
            raise ValueError(f"Invalid entry in {path}: {e}")
        fidx = int(e["frame"])
        keys = e.get("keys", [])
        label = ""
        if keys:
            k = str(keys).lower()
            label = k if k in VALID_KEYS else ""
        events.append((fidx, label))
    events.sort(key=lambda x: x[0])

    return events

def last_non_empty_key_in_frame_range(
    events: List[Tuple[int, str]], start_f: int, end_f: int
) -> str:
    """
    Scan events with frame in [start_f, end_f], return the last non-empty label.
    If none, return "".
    """
    # lower_bound for frame >= start_f
    lo, hi = 0, len(events)
    while lo < hi:
        mid = (lo + hi) // 2
        if events[mid][0] < start_f:
            lo = mid + 1
        else:
            hi = mid
    last = ""
    i = lo
    while i < len(events) and events[i][0] <= end_f:
        lab = events[i][1]
        if lab:
            last = lab
        i += 1
    return last

def grab_frame(cap: cv2.VideoCapture, idx: int) -> Optional[np.ndarray]:
    """Seek to absolute frame index and return frame (BGR) or None on failure."""
    if idx < 0:
        idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None

def put_label_on_frame(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw label text on the frame (simple white text with shadow)."""
    if not text:
        return frame
    img = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    # shadow
    cv2.putText(img, text, (12, 34), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # foreground
    cv2.putText(img, text, (12, 34), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img

def main():
    ap = argparse.ArgumentParser(description="Create labeled video + JSON at target FPS from mp4 + keys.json")
    ap.add_argument("video", type=str, help="Path to input .mp4 video")
    ap.add_argument("keys", type=str, help="Path to keys.json")
    ap.add_argument("--fps", type=float, default=60.0, help="Target output FPS")
    ap.add_argument("--out-video", type=str, default=None, help="Output mp4 path (default: <video>_<fps>fps.mp4)")
    ap.add_argument("--out-json", type=str, default=None, help="Output JSON path (default: <video>_<fps>fps_keys.json)")
    ap.add_argument("--overlay", action="store_true", help="Burn label text onto frames")
    args = ap.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    in_video = args.video
    in_keys = args.keys
    if not os.path.exists(in_video):
        raise FileNotFoundError(in_video)
    if not os.path.exists(in_keys):
        raise FileNotFoundError(in_keys)

    base, _ = os.path.splitext(in_video)
    fps_tag = f"{args.fps:g}"  # compact (e.g., 4, 7.5)
    out_video = args.out_video or f"{base}_{fps_tag}fps.mp4"
    out_json = args.out_json or f"{base}_{fps_tag}fps_keys.json"

    # Load key events
    events = load_keys_json(in_keys)

    # Open video
    cap = cv2.VideoCapture(in_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {in_video}")


    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, 60, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for {out_video}")

    for (ind, keystr) in tqdm(events):

        if keystr == '':
            keystr = 'none'

        frame = grab_frame(cap, ind)
        frame = put_label_on_frame(frame, keystr)

        writer.write(frame)

    writer.release()
    cap.release()

    print(f"[OK] Wrote video:  {out_video}")

if __name__ == "__main__":
    main()