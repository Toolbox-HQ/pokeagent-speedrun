#!/usr/bin/env python3
"""
Creates intervals for all files in a directory -> used for getting training data for an agent.
"""

import os
import json
import glob
import argparse
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_video_info(video_path: str):
    """
    Get video information using torchcodec.
    Returns (start_frame, end_frame, fps) or None if error.
    """
    try:
        decoder = VideoDecoder(video_path)
        
        # Get total frames and fps
        total_frames = len(decoder)
        fps = decoder.metadata.average_fps
        
        # Handle edge cases
        if total_frames == 0:
            print(f"⚠️  Skipping {video_path}: No frames found")
            return None
        
        if fps is None or fps <= 0:
            print(f"⚠️  Skipping {video_path}: Invalid FPS ({fps})")
            return None
        
        # Start from first frame (0) to last frame (total_frames - 1)
        start_frame = 0
        end_frame = total_frames - 1
        
        return {
            "start": start_frame,
            "end": end_frame,
            "video_path": video_path,
            "video_fps": float(fps)
        }
    except Exception as e:
        print(f"❌ Error processing {video_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Create JSON file with video metadata for all videos in internet data folder")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=".cache/pokeagent/agent_data/all_data.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default=".cache/pokeagent/internet_data",
        help="Input directory containing videos (default: .cache/pokeagent/internet_data)"
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=96,
        help="Maximum number of worker processes (default: number of CPU cores)"
    )
    args = parser.parse_args()
    
    # Path to internet data folder
    internet_data_dir = args.input_dir
    output_json_path = args.output
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    # Find all video files
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(glob.glob(os.path.join(internet_data_dir, ext)))
        video_paths.extend(glob.glob(os.path.join(internet_data_dir, ext.upper())))
    
    # Remove duplicates and sort
    video_paths = sorted(list(set(video_paths)))
    
    print(f"Found {len(video_paths)} video files in {internet_data_dir}")
    
    # Process all videos in parallel
    results = []
    failed = 0
    
    max_workers = args.max_workers or os.cpu_count() or 1
    print(f"Using {max_workers} worker processes")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(get_video_info, video_path): video_path 
                         for video_path in video_paths}
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_path), total=len(video_paths), desc="Processing videos"):
            info = future.result()
            if info is not None:
                results.append(info)
            else:
                failed += 1
    
    print(f"\n✅ Successfully processed {len(results)} videos")
    if failed > 0:
        print(f"⚠️  Failed to process {failed} videos")
    
    # Save to JSON file
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Saved {len(results)} entries to {output_json_path}")
    
    # Print some statistics
    if results:
        total_frames = sum(r["end"] - r["start"] + 1 for r in results)
        total_seconds = sum((r["end"] - r["start"] + 1) / r["video_fps"] for r in results)
        print(f"📊 Total frames: {total_frames:,}")
        print(f"📊 Total duration: {total_seconds / 3600:.2f} hours")

if __name__ == "__main__":
    main()

