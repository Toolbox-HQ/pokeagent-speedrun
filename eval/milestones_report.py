#!/usr/bin/env python3
import argparse
import json
import torch
from torchcodec.decoders import VideoDecoder
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed


FPS = 60.0  # video is exactly 60 fps


def frame_to_seconds(frame_idx: int, fps: float = FPS) -> float:
    return float(frame_idx) / float(fps)


def format_hhmmss(seconds: float) -> str:
    # Format as HH:MM:SS.mmm
    if seconds < 0:
        seconds = 0.0
    ms = int(round((seconds - int(seconds)) * 1000.0))
    total = int(seconds)
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def load_milestones(milestones_dir):
    """Load all milestones from the milestones directory."""
    milestones_dir = Path(milestones_dir)
    with open(milestones_dir / "milestones.json") as f:
        milestones_meta = json.load(f)

    milestones = []
    milestone_order = []  # Track order of milestones
    for milestone_id, meta in milestones_meta.items():
        name = meta["name"]
        tensor_path = milestones_dir / f"{name}.pt"
        if not tensor_path.exists():
            continue

        milestone_tensor = torch.load(tensor_path, map_location="cpu")  # [T, C, H, W]
        x1, y1, x2, y2 = meta.get("x1"), meta.get("y1"), meta.get("x2"), meta.get("y2")

        # Extract bounding box if available
        if all(coord is not None for coord in [x1, y1, x2, y2]):
            milestone_frames = milestone_tensor[:, :, y1:y2, x1:x2]  # [T, C, h, w]
        else:
            milestone_frames = milestone_tensor  # [T, C, H, W]

        milestones.append((name, milestone_frames, (x1, y1, x2, y2)))
        milestone_order.append((int(milestone_id), name))  # Store ID and name for ordering

    return milestones, milestone_order


def check_video_milestones(video_path, milestones):
    """Check milestones for a single video file. Returns matches with video name."""
    # Load video frames once
    decoder = VideoDecoder(video_path, device="cpu")
    video_frames = [decoder[i] for i in range(len(decoder))]
    video_name = Path(video_path).name

    # Check matches: video frames (outer loop)
    matches = []
    matched_milestone_names = set()  # Track milestones that have been matched (per video)

    for v, video_frame in enumerate(tqdm(video_frames, desc=f"Checking {video_name}")):
        for name, milestone_frames, (x1, y1, x2, y2) in milestones:
            # Skip milestones that have already been matched in this video
            if name in matched_milestone_names:
                continue

            # Extract region from video frame if bounding box exists
            if all(coord is not None for coord in [x1, y1, x2, y2]):
                video_region = video_frame[:, y1:y2, x1:x2]
            else:
                video_region = video_frame

            # Check against all milestone frames
            for t in range(milestone_frames.shape[0]):
                milestone_frame = milestone_frames[t]  # [C, h, w]
                assert milestone_frame.shape == video_region.shape
                if torch.equal(milestone_frame, video_region):
                    # Include times derived from frame indices at 60fps
                    v_time_s = frame_to_seconds(v)
                    t_time_s = frame_to_seconds(t)
                    matches.append((name, video_name, v, t, v_time_s, t_time_s))
                    matched_milestone_names.add(name)  # Mark this milestone as matched in this video
                    break

    return matches


def print_combined_report(all_matches, milestone_order):
    """Print combined milestone report for all videos, reporting ALL matches (with time)."""
    # milestone name -> list of (video_name, v, t, v_time_s, t_time_s)
    matched_milestones = {}
    for name, video_name, v, t, v_time_s, t_time_s in all_matches:
        matched_milestones.setdefault(name, []).append((video_name, v, t, v_time_s, t_time_s))

    # Sort matches within each milestone for nicer output
    for name in matched_milestones:
        matched_milestones[name].sort(key=lambda x: (x[0], x[1], x[2]))

    print("\n" + "=" * 80)
    print("MILESTONE REPORT (ALL MATCHES, WITH TIME @ 60 FPS)")
    print("=" * 80)

    # Sort by milestone ID
    milestone_order_sorted = sorted(milestone_order, key=lambda x: x[0])

    for milestone_id, name in milestone_order_sorted:
        if name in matched_milestones:
            matches = matched_milestones[name]
            status = "✅"

            print(
                f"{status}  [{milestone_id}] {name:20s}"
            )

            if len(matches) > 0:
                for video_name, v, t, v_time_s, t_time_s in matches:
                    print(
                        f"    ↳ {video_name} - Frame {v} @ {format_hhmmss(v_time_s)} ({v_time_s:.3f}s) "
                        f"(milestone frame {t} @ {format_hhmmss(t_time_s)} ({t_time_s:.3f}s))"
                    )
                print(f"    (total matches: {len(matches)})")
        else:
            status = "❌"
            print(f"{status}  [{milestone_id}] {name:20s}  NOT MATCHED")

    print("=" * 80)


@torch.no_grad()
def frames_approx_equal_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: int = 2,
    max_bad_frac: float = 1e-4,  # allow this fraction of elements to exceed atol
) -> bool:
    if a.shape != b.shape:
        return False

    # upcast to avoid uint wraparound on subtraction
    diff = (a.to(torch.int16) - b.to(torch.int16)).abs()
    bad_count = int((diff > atol).sum().item())
    total = diff.numel()

    return bad_count <= max_bad_frac * total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--milestones-dir", type=str, default="eval/milestones")
    parser.add_argument("--filter", type=str, default=None)

    args = parser.parse_args()

    # Load all milestones once (shared across all videos)
    milestones, milestone_order = load_milestones(args.milestones_dir)

    # Determine if video_path is a directory or file
    video_path = Path(args.video_path)
    if video_path.is_dir():
        # Get all .mp4 files in the directory (in arbitrary order)
        video_files = list(video_path.glob("*.mp4"))
        if args.filter is not None:
            video_files = list(filter(lambda x: args.filter in str(x), video_files))

        if not video_files:
            print(f"No .mp4 files found in directory: {video_path}")
            return
        print(f"Found {len(video_files)} video file(s) in directory")
    elif video_path.is_file():
        video_files = [video_path]
    else:
        print(f"Error: {video_path} is not a valid file or directory")
        return

    # Process each video file in parallel and collect all matches
    all_matches = Parallel(n_jobs=-1)(
        delayed(check_video_milestones)(str(video_file), milestones)
        for video_file in video_files
    )

    # Flatten the list of matches from all videos
    all_matches = [match for video_matches in all_matches for match in video_matches]

    # Print combined report
    print_combined_report(all_matches, milestone_order)


if __name__ == "__main__":
    main()
