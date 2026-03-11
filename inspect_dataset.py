#!/usr/bin/env python3
"""
inspect_dataset.py

Log mode  (run first):
    python inspect_dataset.py --config config/embed_objective_agent.yaml
    Builds the dataset, caches all sample info to ./tmp/dataset_cache.json, prints a log.

Visualize mode (run after log mode):
    python inspect_dataset.py --visualize 5 10 15
    Loads ./tmp/dataset_cache.json and renders the requested samples. No model loading.
"""

import argparse
import bisect
import json
import os
import random
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import yaml

CACHE_PATH = "./tmp/dataset_cache.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--visualize", type=int, nargs="+", metavar="IDX",
                   help="Visualize these sample indices (loads from cache, no model needed)")
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Log mode
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_single_process_dist():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "35332")
    from models.util.dist import init_distributed
    init_distributed()


class EnrichedObjectivesLookup:
    """
    Drop-in for ObjectivesLookup. .lookup() returns enriched dicts:
        [{"cluster_idx": int, "n_embeds": int, "source_frames": [{"video_path", "frame_idx"}, ...]}, ...]
    Tensor embeds are not stored — only what is needed for visualization.
    Safe with processor=None since __getitem__ never reads objectives in that case.
    """

    def __init__(self, per_frame_metadata, cluster_to_frames, cluster_to_embeds):
        self.lookup_dict = {}

        by_video = {}
        for data in per_frame_metadata:
            by_video.setdefault(data["video_path"], []).append(data)

        for video_path, items in by_video.items():
            items = sorted(items, key=lambda x: x["sampled_frame_index"])
            seen_clusters = set()
            history = []
            frame_data_list = []

            for data in items:
                cluster_idx = data.get("cluster_idx", -1)
                if cluster_idx > -1 and cluster_idx not in seen_clusters:
                    seen_clusters.add(cluster_idx)
                    history = history + [{
                        "cluster_idx": cluster_idx,
                        "n_embeds": len(cluster_to_embeds.get(cluster_idx, [])),
                        "source_frames": cluster_to_frames.get(cluster_idx, []),
                    }]

                frame_data_list.append({
                    "idx": data["sampled_frame_index"],
                    "objectives": history.copy(),
                })

            self.lookup_dict[video_path] = frame_data_list

    def lookup(self, video_path, frame_idx):
        entries = self.lookup_dict.get(video_path, [])
        if not entries:
            return []
        pos = bisect.bisect_left(entries, frame_idx, key=lambda e: e["idx"])
        if pos == 0:
            return []
        return entries[pos - 1]["objectives"]


def _make_enriched_mine_objectives():
    from models.train.train_agent import (
        get_objective_dataset_json,
        load_objective_dataset,
        create_clusters,
        filter_clusters,
        AgentObjectiveManager,
    )

    def enriched_mine_objectives(all_videos_json_files):
        videos_json = get_objective_dataset_json(all_videos_json_files)
        per_frame_embed, per_frame_metadata = load_objective_dataset(videos_json)
        clusterer = create_clusters(per_frame_embed)
        filtered_labels = filter_clusters(clusterer.labels_, per_frame_metadata)

        cluster_to_frames = {}
        cluster_to_embeds = {}
        valid_cluster_idxs = []

        for meta_idx, cluster_idx in enumerate(filtered_labels):
            if cluster_idx > -1:
                cluster_to_frames.setdefault(cluster_idx, []).append({
                    "video_path": per_frame_metadata[meta_idx]["video_path"],
                    "frame_idx": per_frame_metadata[meta_idx]["sampled_frame_index"],
                })
                cluster_to_embeds.setdefault(cluster_idx, []).append(
                    torch.tensor(per_frame_embed[meta_idx])
                )

        for meta_idx, metadata in enumerate(per_frame_metadata):
            cluster_idx = filtered_labels[meta_idx]
            metadata["cluster_idx"] = cluster_idx
            if cluster_idx > -1:
                metadata["cluster_embed"] = cluster_to_embeds[cluster_idx]
                valid_cluster_idxs.append(cluster_idx)

        objectives_lookup = EnrichedObjectivesLookup(per_frame_metadata, cluster_to_frames, cluster_to_embeds)
        objective_manager = AgentObjectiveManager(clusterer, valid_cluster_idxs)
        return objectives_lookup, objective_manager

    return enriched_mine_objectives


def _idm_frame_indices(s):
    from models.inference.idm_inference_dataloader import IDM_FPS
    stride = max(1, int(round(s["video_fps"] / IDM_FPS)))
    return list(range(s["start"], s["end"], stride)), stride


def _sample_to_cache_entry(s, num_objectives):
    """Serialize one sample to a JSON-safe dict (no tensors)."""
    idm_indices, stride = _idm_frame_indices(s)
    objs = s["objectives"]
    active = objs[-num_objectives:]
    serialized_objs = []
    for obj in active:
        if isinstance(obj, dict):
            serialized_objs.append({
                "cluster_idx": int(obj["cluster_idx"]),
                "n_embeds": int(obj["n_embeds"]),
                "source_frames": obj["source_frames"],
            })
        else:
            serialized_objs.append({"cluster_idx": -1, "n_embeds": len(obj), "source_frames": []})
    return {
        "video_path": s["video_path"],
        "start": s["start"],
        "end": s["end"],
        "video_fps": s["video_fps"],
        "stride": stride,
        "idm_frame_indices": idm_indices,
        "objectives_total": len(objs),
        "objectives": serialized_objs,
    }


def run_log_mode(config_path, max_samples):
    from models.inference.idm_inference_dataloader import IDM_FPS

    bootstrap_single_process_dist()

    import models.train.train_agent as train_agent
    train_agent.mine_objectives = _make_enriched_mine_objectives()

    sys.argv = [sys.argv[0], "--config", config_path]
    from models.train.train_agent import setup_training, create_dataset
    model, processor, data_args, training_args = setup_training()
    del model

    train_ds, eval_ds, _ = create_dataset(
        data_args.data_path, processor, None, split=0.05, online=False, data_args=data_args
    )

    samples = train_ds.dataset.dataset.samples
    if max_samples is not None:
        samples = samples[:max_samples]

    num_objectives = data_args.num_objectives
    print(f"\nTotal samples (train split): {len(samples)}")

    cache = {
        "num_objectives": num_objectives,
        "idm_fps": IDM_FPS,
        "samples": [_sample_to_cache_entry(s, num_objectives) for s in samples],
    }
    os.makedirs("./tmp", exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    print(f"Cache saved to {CACHE_PATH}")

    lines = [f"Dataset: {len(samples)} samples  (cache: {CACHE_PATH})\n"]
    for i, entry in tqdm(enumerate(cache["samples"])):
        idm_indices = entry["idm_frame_indices"]
        lines.append(f"=== Sample {i} ===")
        lines.append(f"  video_path        : {entry['video_path']}")
        lines.append(f"  start_frame       : {entry['start']}")
        lines.append(f"  end_frame         : {entry['end']}")
        lines.append(f"  video_fps         : {entry['video_fps']}")
        lines.append(f"  stride            : {entry['stride']}  ({entry['video_fps']:.1f} / {IDM_FPS})")
        lines.append(f"  idm_frame_count   : {len(idm_indices)}")
        preview = idm_indices[:10]
        suffix = ", ..." if len(idm_indices) > 10 else ""
        lines.append(f"  idm_frame_indices : [{', '.join(map(str, preview))}{suffix}]")
        lines.append(f"  objectives_total  : {entry['objectives_total']}")
        lines.append(f"  objectives_active : {len(entry['objectives'])} (num_objectives={num_objectives})")
        for j, obj in enumerate(entry["objectives"]):
            n_src = len(obj["source_frames"])
            lines.append(f"    [{j}] cluster_idx={obj['cluster_idx']}  n_embeddings={obj['n_embeds']}  n_source_frames={n_src}")
            for src in obj["source_frames"][:3]:
                lines.append(f"         -> {Path(src['video_path']).name}  frame {src['frame_idx']}")
            if n_src > 3:
                lines.append(f"         -> ... ({n_src - 3} more)")
    print("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Visualize mode
# ──────────────────────────────────────────────────────────────────────────────

def _decode_frame(video_path, frame_idx):
    from torchcodec.decoders import VideoDecoder
    return VideoDecoder(video_path).get_frames_at(indices=[frame_idx]).data[0].permute(1, 2, 0).numpy()


def visualize_sample(idx, entry, num_objectives):
    import matplotlib.pyplot as plt
    from torchcodec.decoders import VideoDecoder

    idm_indices = entry["idm_frame_indices"]
    active = entry["objectives"]

    print(f"\n=== Sample {idx} ===")
    print(f"video      : {entry['video_path']}")
    print(f"frames     : {entry['start']} -> {entry['end']}  (stride={entry['stride']}, idm_fps={entry.get('idm_fps', 4)})")
    print(f"frame_count: {len(idm_indices)}")

    frames = VideoDecoder(entry["video_path"]).get_frames_at(indices=idm_indices).data

    n_display = 16
    step = max(1, len(idm_indices) // n_display)
    disp_idxs = list(range(0, len(idm_indices), step))[:n_display]
    disp_frames = [frames[i].permute(1, 2, 0).numpy() for i in disp_idxs]
    disp_frame_nums = [idm_indices[i] for i in disp_idxs]

    n_objs = len(active)
    n_frame_cols = 8
    n_frame_rows = (len(disp_frames) + n_frame_cols - 1) // n_frame_cols
    n_obj_cols = min(n_frame_cols, n_objs) if n_objs else 1
    n_obj_rows = ((n_objs + n_obj_cols - 1) // n_obj_cols) if n_objs else 0
    total_rows = n_frame_rows + n_obj_rows

    fig = plt.figure(figsize=(n_frame_cols * 2.5, max(4, total_rows * 2.5)))
    fig.suptitle(
        f"Sample {idx}  |  {Path(entry['video_path']).name}\n"
        f"frames {entry['start']}–{entry['end']}  stride={entry['stride']}  ({len(idm_indices)} IDM frames)",
        fontsize=9,
    )

    for i, (frame, fnum) in enumerate(zip(disp_frames, disp_frame_nums)):
        ax = fig.add_subplot(total_rows, n_frame_cols, i + 1)
        ax.imshow(frame)
        ax.set_title(f"f{fnum}", fontsize=6)
        ax.axis("off")

    if n_objs:
        print(f"\nObjectives ({n_objs} active):")
        for j, obj in enumerate(active):
            src_list = obj["source_frames"]
            print(f"  [{j}] cluster_idx={obj['cluster_idx']}  n_embeddings={obj['n_embeds']}  n_source_frames={len(src_list)}")
            for src in src_list[:3]:
                print(f"       -> {src['video_path']}  frame {src['frame_idx']}")
            if len(src_list) > 3:
                print(f"       -> ... ({len(src_list) - 3} more)")

            ax = fig.add_subplot(total_rows, n_obj_cols, n_frame_rows * n_obj_cols + j + 1)
            if src_list:
                src = random.choice(src_list)
                try:
                    ax.imshow(_decode_frame(src["video_path"], src["frame_idx"]))
                    ax.set_title(
                        f"obj[{j}] cluster={obj['cluster_idx']}\n"
                        f"{Path(src['video_path']).name} f{src['frame_idx']}\n"
                        f"({obj['n_embeds']} embeds, {len(src_list)} frames)",
                        fontsize=5,
                    )
                except Exception:
                    ax.text(0.5, 0.5, f"cluster={obj['cluster_idx']}\n(load error)", ha="center", va="center", fontsize=6)
            else:
                ax.text(0.5, 0.5, f"cluster={obj['cluster_idx']}\nno frames", ha="center", va="center", fontsize=6)
            ax.axis("off")
    else:
        print("  (no objectives active at this window position)")

    plt.tight_layout()
    os.makedirs("./tmp", exist_ok=True)
    out_path = f"./tmp/sample_{idx}_viz.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


def run_visualize_mode(indices):
    if not os.path.exists(CACHE_PATH):
        print(f"Error: cache not found at {CACHE_PATH}. Run log mode first (--config ...).")
        sys.exit(1)

    with open(CACHE_PATH) as f:
        cache = json.load(f)

    samples = cache["samples"]
    num_objectives = cache["num_objectives"]
    print(f"Loaded {len(samples)} samples from {CACHE_PATH}")

    for idx in indices:
        if idx < 0 or idx >= len(samples):
            print(f"Warning: index {idx} out of range (0..{len(samples)-1}), skipping")
            continue
        visualize_sample(idx, samples[idx], num_objectives)


# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.visualize is not None:
        run_visualize_mode(args.visualize)
    else:
        if args.config is None:
            print("Error: --config is required for log mode")
            sys.exit(1)
        run_log_mode(args.config, args.max_samples)


if __name__ == "__main__":
    main()
