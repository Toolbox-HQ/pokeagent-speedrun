import os
import sys
from torchcodec.decoders import VideoDecoder
from models.util.data import list_files_with_extentions

def check_video(video_path):
    try:
        VideoDecoder(video_path).get_frames_at([0])
        return None
    except Exception as e:
        return (video_path, str(e))

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python script/find_corrupt_videos.py <data_dir>"
    data_dir = sys.argv[1]
    local_path = os.path.join(".cache", data_dir)

    video_files = list_files_with_extentions(local_path, ".mp4")

    print(f"Checking {len(video_files)} videos in {local_path}...")
    corrupt = []
    for p in video_files:
        result = check_video(p)
        if result:
            corrupt.append(result)
            print(f"CORRUPT: {result[0]}\n  {result[1]}")

    print(f"\n{len(corrupt)} / {len(video_files)} corrupt files")
    if corrupt:
        for path, err in corrupt:
            print(path)
