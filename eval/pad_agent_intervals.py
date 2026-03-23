#!/usr/bin/env python3
"""
Pads each interval in an agent JSON by 32 IDM frames on each side,
clamped to valid video bounds.
"""

import argparse
import json
import os
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

IDM_FPS = 4
PAD_IDM_FRAMES = 32


def pad_intervals(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    results = []
    video_lengths = {}

    for it in tqdm(items, desc="Padding intervals"):
        video_path = it["video_path"]
        fps = float(it["video_fps"])
        start = int(it["start"])
        end = int(it["end"])

        if video_path not in video_lengths:
            try:
                decoder = VideoDecoder(video_path)
                video_lengths[video_path] = decoder.metadata.num_frames
            except Exception as e:
                print(f"Error reading {video_path}: {e}")
                results.append(it)
                continue

        total_frames = video_lengths[video_path]
        stride = max(1, int(round(fps / IDM_FPS)))
        pad_raw = PAD_IDM_FRAMES * stride

        new_start = max(0, start - pad_raw)
        new_end = min(total_frames - 1, end + pad_raw)

        results.append({
            "start": new_start,
            "end": new_end,
            "video_path": video_path,
            "video_fps": fps,
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} intervals to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pad agent intervals by 32 IDM frames on each side")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output JSON file")
    args = parser.parse_args()

    pad_intervals(args.input, args.output)


if __name__ == "__main__":
    main()
