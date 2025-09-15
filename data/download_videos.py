from pytubefix import YouTube
from pytubefix.cli import on_progress
import argparse
from typing import List
import json
from tqdm import tqdm
import os
import joblib
import functools


def list_files(directory):
    return [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]


def download_video(youtube_id: str, output_dir: str):
    try:
        url = f"https://www.youtube.com/watch?v={youtube_id}"
        yt = YouTube(url, on_progress_callback=on_progress)
        ys = yt.streams.get_highest_resolution()
        ys.download(output_path=output_dir, filename=f"{youtube_id}.mp4")
        return 1
    except Exception as e:
        print(f"[ERROR] {youtube_id} | {e}")
        return 0


def download_videos_in_parallel(
    youtube_ids: List[str], output_dir: str, threads: int = 1
):
    cached_videos = [file.split(".")[0] for file in list_files(output_dir)]
    ids_to_download = [id for id in youtube_ids if id not in cached_videos]

    if not ids_to_download:
        print("All videos are already cached.")
        return

    results = joblib.Parallel(n_jobs=threads)(
        joblib.delayed(download_video)(id, output_dir) for id in tqdm(ids_to_download)
    )
    fetched_videos = functools.reduce(lambda x, y: x + y, results)
    print(f"{fetched_videos} were retrieved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output_dir", default="./cache/video", type=str)
    parser.add_argument("--threads", default=1, type=int)

    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.input, "r") as f:
        metadata = json.load(f)

    ids = [item["search_result"]["id"]["videoId"] for item in metadata]

    # Use the joblib-wrapped function
    download_videos_in_parallel(ids, args.output_dir, threads=args.threads)
