#!/usr/bin/env python3
"""
Split agent_eval videos and JSON files into subfolders based on per-video timestamps.

Usage:
    python script/split_eval.py --input .cache/agent_eval \\
        --video 0dde27e6 00:10:00 00:20:00 \\
        --video 1271ea42 00:05:00

Each --video arg is: <stem> <timestamp1> [timestamp2 ...]
N timestamps produce N+1 splits.

All videos must have the same number of splits (timestamps count must match).
Outputs go into split_0/, split_1/, ... inside the input directory.

Timestamps: HH:MM:SS, MM:SS, or raw seconds.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path


def parse_timestamp(ts: str) -> float:
    if re.match(r"^\d+(\.\d+)?$", ts):
        return float(ts)
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Cannot parse timestamp: {ts!r}")


def seconds_to_ffmpeg(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def split_video(src: Path, dst: Path, start: float, end: float | None) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ss", seconds_to_ffmpeg(start)]
    if end is not None:
        cmd += ["-to", seconds_to_ffmpeg(end)]
    cmd += ["-c", "copy", str(dst)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def split_json(src: Path, dst: Path, start_frame: int, end_frame: int | None) -> None:
    data = json.loads(src.read_text())
    subset = [
        {"frame": entry["frame"] - start_frame, "keys": entry["keys"]}
        for entry in data
        if entry["frame"] >= start_frame and (end_frame is None or entry["frame"] < end_frame)
    ]
    dst.write_text(json.dumps(subset, indent=None))


def parse_video_args(raw: list[list[str]]) -> dict[str, list[float]]:
    """Parse --video stem ts1 ts2 ... into {stem: [ts1, ts2, ...]}."""
    result = {}
    for tokens in raw:
        if len(tokens) < 2:
            raise ValueError(f"--video needs at least <stem> <timestamp>, got: {tokens}")
        stem = tokens[0]
        split_times = sorted(parse_timestamp(t) for t in tokens[1:])
        result[stem] = split_times
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Split agent_eval files per-video by timestamps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", default=".cache/agent_eval", help="Input directory")
    parser.add_argument(
        "--video",
        nargs="+",
        action="append",
        metavar="ARG",
        required=True,
        help="<stem> <ts1> [ts2 ...], e.g. --video 0dde27e6 00:10:00 00:20:00",
    )
    parser.add_argument("--fps", type=float, default=60.0, help="Video FPS (default: 60)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    videos = parse_video_args(args.video)

    # Validate all videos produce the same number of splits
    n_splits_per = {stem: len(ts) + 1 for stem, ts in videos.items()}
    unique_counts = set(n_splits_per.values())
    if len(unique_counts) > 1:
        details = ", ".join(f"{s}={n}" for s, n in n_splits_per.items())
        raise ValueError(f"All videos must produce the same number of splits. Got: {details}")

    n_splits = unique_counts.pop()
    print(f"Splitting {len(videos)} video(s) into {n_splits} splits each:")

    for stem, split_times in videos.items():
        boundaries = []
        prev = 0.0
        for t in split_times:
            boundaries.append((prev, t))
            prev = t
        boundaries.append((prev, None))

        print(f"\n  {stem}:")
        for i, (s, e) in enumerate(boundaries):
            end_str = seconds_to_ffmpeg(e) if e is not None else "end"
            print(f"    split_{i}: {seconds_to_ffmpeg(s)} -> {end_str}")

        mp4_src = input_dir / f"{stem}.mp4"
        json_src = input_dir / f"{stem}.json"

        if not mp4_src.exists():
            raise FileNotFoundError(f"Video not found: {mp4_src}")

        for i, (start_s, end_s) in enumerate(boundaries):
            split_dir = input_dir / f"split_{i}"
            split_dir.mkdir(exist_ok=True)

            start_frame = int(start_s * args.fps)
            end_frame = int(end_s * args.fps) if end_s is not None else None

            split_video(mp4_src, split_dir / mp4_src.name, start_s, end_s)

            if json_src.exists():
                split_json(json_src, split_dir / json_src.name, start_frame, end_frame)
            else:
                print(f"    WARNING: no JSON found for {stem}")

    print("\nDone.")


if __name__ == "__main__":
    main()
