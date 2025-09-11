from pytubefix import YouTube
from pytubefix.cli import on_progress
import argparse
from typing import List
import json
from tqdm import tqdm
import os


def list_files(directory):
    return [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]


def download_videos(youtube_ids: List[str], output_dir: str):
    cached_videos = [file.split(".")[0] for file in list_files(output_dir)]

    for id in tqdm(youtube_ids):
        if id not in cached_videos:
            url = f"https://www.youtube.com/watch?v={id}"
            yt = YouTube(url, on_progress_callback=on_progress)
            ys = yt.streams.get_highest_resolution()
            ys.download(output_path=output_dir, filename=f"{id}.mp4")
            cached_videos.append(id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output_dir", default="./cache/video", type=str)

    args = parser.parse_args()

    with open(args.input, "r") as f:
        metadata = json.load(f)

    ids = [item["search_result"]["id"]["videoId"] for item in metadata]

    download_videos(ids, args.output_dir)
