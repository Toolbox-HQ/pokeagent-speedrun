#!/usr/bin/env python3
"""
inspect_dataset.py
  python inspect_dataset.py --config config/embed_objective_agent.yaml --indices 0:20                                                                                                                                                                                                                   
  python inspect_dataset.py --config config/embed_objective_agent.yaml --indices 0:100:5   # every 5th                                                                                                                                                                                                  
  python inspect_dataset.py --config config/embed_objective_agent.yaml --indices 0 5 10 15                                                                                                                                                                                                              
  python inspect_dataset.py --config config/embed_objective_agent.yaml --indices 0:10 50:60   
"""

import argparse
import os
import sys
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--indices", type=str, nargs="+", default=["0:5"],
                   help="Indices or ranges, e.g. 0 5 10 or 0:20 or 0:100:5")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--jobs", type=int, default=-1, help="Parallel workers (-1 = all cores)")
    return p.parse_args()


def parse_indices(specs, max_len):
    indices = []
    for spec in specs:
        if ":" in spec:
            parts = [int(x) if x else None for x in spec.split(":")]
            indices.extend(range(*slice(*parts).indices(max_len)))
        else:
            indices.append(int(spec))
    return sorted(set(indices))


def bootstrap_single_process_dist():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "35332")
    from models.util.dist import init_distributed
    init_distributed()


def visualize_sample(idx, train_ds, num_objectives):
    import matplotlib.pyplot as plt

    s = train_ds.dataset.dataset.samples[train_ds.dataset.indices[idx]]
    idm_frames, obj_frames = train_ds[idx]  # (T, C, H, W), (N, C, H, W) — processor=None
    active_objs = s["objectives"][-num_objectives:]

    n_display = 16
    step = max(1, idm_frames.shape[0] // n_display)
    disp_frames = idm_frames[::step][:n_display]

    n_frame_cols = 8
    n_obj_cols = min(n_frame_cols, num_objectives)
    n_frame_rows = (len(disp_frames) + n_frame_cols - 1) // n_frame_cols
    n_obj_rows = (num_objectives + n_obj_cols - 1) // n_obj_cols
    total_rows = n_frame_rows + n_obj_rows

    fig = plt.figure(figsize=(n_frame_cols * 2.5, max(4, total_rows * 2.5)))
    fig.suptitle(
        f"Sample {idx}  |  {Path(s['video_path']).name}\n"
        f"frames {s['start']}–{s['end']}  ({len(s['objectives'])} objectives active)",
        fontsize=9,
    )

    for i, frame in enumerate(disp_frames):
        ax = fig.add_subplot(total_rows, n_frame_cols, i + 1)
        ax.imshow(frame.permute(1, 2, 0).numpy())
        ax.axis("off")

    for j in range(num_objectives):
        ax = fig.add_subplot(total_rows, n_obj_cols, n_frame_rows * n_obj_cols + j + 1)
        if j < len(active_objs):
            ax.imshow(obj_frames[j].permute(1, 2, 0).numpy())
            ax.set_title(f"obj[{j}] cluster={active_objs[j]['cluster_idx']}", fontsize=6)
        else:
            ax.text(0.5, 0.5, f"obj[{j}]\n(pad)", ha="center", va="center", fontsize=7, transform=ax.transAxes)
        ax.axis("off")

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


def main():
    args = parse_args()
    bootstrap_single_process_dist()

    sys.argv = [sys.argv[0], "--config", args.config]
    from models.train.train_agent import setup_training, create_dataset
    _, _, data_args, _ = setup_training()

    train_ds, _, _ = create_dataset(
        data_args.data_path, None, None, split=0.05, online=False, data_args=data_args
    )

    n = args.max_samples if args.max_samples is not None else len(train_ds)
    indices = [i for i in parse_indices(args.indices, n) if 0 <= i < len(train_ds)]

    print(f"Total samples (train split): {len(train_ds)}  |  Visualizing {len(indices)}")

    Parallel(n_jobs=args.jobs, backend="loky")(
        delayed(visualize_sample)(idx, train_ds, data_args.num_objectives)
        for idx in tqdm(indices, desc="Visualizing")
    )


if __name__ == "__main__":
    main()
