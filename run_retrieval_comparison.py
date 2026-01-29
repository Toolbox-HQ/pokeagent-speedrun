
from models.inference.find_matching_videos import get_videos, get_random_videos, best_single_match_search, get_videos_mean
import json
from models.util.dist import init_distributed, clean_dist_and_exit
from models.util.misc import inject_traceback
import os
import argparse

def main():

    # -------------------------
    # Configuration
    # -------------------------
    parser = argparse.ArgumentParser(description="Process a query video.")
    parser.add_argument(
        "query_video_path",
        type=str,
        help="Path to the query video file"
    )
    args = parser.parse_args()

    QUERY_VIDEO_PATH = args.query_video_path

    #QUERY_VIDEO_PATH = "query_video/query_gpu0_bootstrap1.mp4"
    EMB_DIR = ".cache/pokeagent/db_embeddings"
    OUTPUT_DIR = "comparison_output"

    INTERVAL_LENGTH = 540
    NUM_RESULTS = 20
    MAX_VID_LEN = None

    # -------------------------
    # Init
    # -------------------------
    init_distributed()
    inject_traceback()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------
    # 1) Original pipeline (max + windows + entropy)
    # -------------------------
    videos_max_window_entropy = get_videos(
        QUERY_VIDEO_PATH,
        EMB_DIR,
        INTERVAL_LENGTH,
        NUM_RESULTS,
        MAX_VID_LEN
    )
    with open(f"{OUTPUT_DIR}/videos_max_window_entropy.json", "w") as f:
        json.dump(videos_max_window_entropy, f, indent=2)

    # -------------------------
    # 2) Random baseline
    # -------------------------
    videos_random = get_random_videos(
        emb_dir=EMB_DIR,
        num_intervals=NUM_RESULTS
    )
    with open(f"{OUTPUT_DIR}/videos_random.json", "w") as f:
        json.dump(videos_random, f, indent=2)

    # -------------------------
    # 3) Best single match (max over everything)
    # -------------------------
    videos_max_single_match = best_single_match_search(
        query_video_path=QUERY_VIDEO_PATH,
        emb_dir=EMB_DIR,
        top_n=NUM_RESULTS
    )
    with open(f"{OUTPUT_DIR}/videos_max_single_match.json", "w") as f:
        json.dump(videos_max_single_match, f, indent=2)

    # -------------------------
    # 4) Mean-based pipeline
    # -------------------------
    videos_mean = get_videos_mean(
        query_path=QUERY_VIDEO_PATH,
        emb_dir=EMB_DIR,
        num_intervals=NUM_RESULTS,
        max_vid_len=MAX_VID_LEN
    )
    with open(f"{OUTPUT_DIR}/videos_mean.json", "w") as f:
        json.dump(videos_mean, f, indent=2)


if __name__ == "__main__":
    main()
    clean_dist_and_exit()
