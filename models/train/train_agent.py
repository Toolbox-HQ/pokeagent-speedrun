# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
import bisect
import json
import pickle
from pathlib import Path
from typing import Tuple, Callable, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import transformers
from models.dataclass import TrainingArguments, DataArguments, ModelArguments
from models.util.trainer import Trainer
from torch.utils.data import Dataset
from models.model.agent_modeling.agent import init_lm_agent, init_vision_prcoessor
from models.util.repro import repro_init
from models.util.dist import init_distributed
from models.inference.idm_inference_dataloader import OnlineAgentDataset, AgentPretrainingDataset, get_idm_labeller
import os
from models.util.data import train_val_split, list_files_with_extensions, ResampleDataset
import torch.distributed as dist
from cuml.cluster import HDBSCAN
from cuml.cluster.hdbscan import approximate_predict
import numpy as np

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def init_model(model_args: ModelArguments, training_args: TrainingArguments):

    device = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{device}")
    model = init_lm_agent(arch=model_args.architecture, lm=model_args.lm_name_or_path, vision=model_args.vision_name_or_path)
    processor = init_vision_prcoessor(vision=model_args.vision_name_or_path)
    model.idm_labelling_fn, idm = get_idm_labeller(device)

    if training_args.gradient_checkpointing:
        model.text_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)
        model.vision_tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)
        training_args.gradient_checkpointing = False
    
    return model, idm, processor, device

def setup_training() -> Tuple[nn.Module, Callable, DataArguments, TrainingArguments]:
    global local_rank
    local_rank = int(os.environ.get("RANK", 0))
    
    from argparse import ArgumentParser

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    save_path = repro_init(args.config)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_yaml_file(yaml_file=args.config)

    local_rank = training_args.local_rank
    training_args.output_dir = save_path

    model, idm, processor, device = init_model(model_args, training_args)
    
    return model, processor, data_args, training_args


class ObjectivesLookup:

    def __init__(self, metadata):
        self.lookup_dict = {}

        # Group frames by video first
        by_video = {}
        for data in metadata:
            by_video.setdefault(data["video_path"], []).append(data)

        for video_path, items in by_video.items():
            # Make bisect safe and deterministic
            items = sorted(items, key=lambda x: x["sampled_frame_index"])

            seen_clusters = set()
            history = []
            frame_objective_data_list = []

            for data in items:
                cluster_idx = data.get("cluster_idx", -1)

                # Only add each cluster once per video
                if cluster_idx > -1 and cluster_idx not in seen_clusters:
                    seen_clusters.add(cluster_idx)
                    current_video_frames = [f for f in data["cluster_frames"] if f["video_path"] == video_path]
                    history = history + [{"cluster_idx": cluster_idx, "embed": data["cluster_embed"], "frames": current_video_frames}]

                frame_objective_data_list.append({
                    "idx": data["sampled_frame_index"],
                    "objective_embeds": history.copy(),
                })

            self.lookup_dict[video_path] = frame_objective_data_list

    def lookup(self, video_path, frame_idx) -> List[dict]:
        entries = self.lookup_dict.get(video_path, [])
        if not entries:
            return []

        pos = bisect.bisect_left(entries, frame_idx, key=lambda e: e["idx"])
        if pos == 0:
            return []
        return entries[pos - 1]["objective_embeds"]

class AgentObjectiveManager:

    def __init__(self, clusterer, valid_cluster_ids):
        self.clusterer = clusterer
        self.matched_objectives = {id : False for id in valid_cluster_ids}
        self.achieved_objectives = []
        
    def mine_and_add_objectives(self, embeds: List[np.ndarray], images: torch.tensor):
        labels, _ = approximate_predict(self.clusterer, np.stack(embeds))
        for idx, label in enumerate(labels):
            if label in self.matched_objectives:
                if not self.matched_objectives[label]:
                    self.matched_objectives[label] = True
                    self.achieved_objectives.append(images[idx])

    def retrieve_last_n_objectives(self, n) -> List[torch.tensor]:
        return self.achieved_objectives[-n:]

def create_clusters(
                    data,
                    min_cluster_size = 30,
                    min_samples = 30,
                    metric = "euclidean",
                    build_algo = "brute_force",
                    ):
    
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        build_algo=build_algo,
        prediction_data=True,
    )
    return clusterer.fit(data)
    
def filter_clusters(labels, per_frame_metadata, min_unique_videos_per_cluster = 30):
    
    unique_vids_per_cluster = {}
    for label_idx, label in enumerate(labels):
        vids = unique_vids_per_cluster.get(label, [])
        if per_frame_metadata[label_idx]["video_path"] not in vids:
            vids.append(per_frame_metadata[label_idx]["video_path"])
            unique_vids_per_cluster[label] = vids
    return [label if len(unique_vids_per_cluster[label]) > min_unique_videos_per_cluster else -1 for label in labels]


def get_objective_dataset_json(all_videos_json_files):
    videos_json = []

    for json_file in all_videos_json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                if not any(vid["video_path"] == item["video_path"] for vid in videos_json):
                    videos_json.append({"video_path": item["video_path"]})
    return videos_json

def load_objective_dataset(videos_json, emb_dir=".cache/pokeagent/dinov2"):
    per_frame_embed, per_frame_metadata = [], []
    for item in videos_json:
        video_id = Path(item["video_path"]).stem
        pt_path = os.path.join(emb_dir, f"{video_id}.pt")
        json_path = os.path.join(emb_dir, f"{video_id}.json")
        emb = torch.load(pt_path, map_location="cpu", weights_only=True).numpy()  # (n, d)
        with open(json_path, "r") as f:
            meta = json.load(f)
        per_frame_embed.append(emb)
        per_frame_metadata.extend(meta)
    return np.concatenate(per_frame_embed, axis=0), per_frame_metadata # N x 768 and list where len(list) == N

def mine_objectives(all_videos_json_files):
    json = get_objective_dataset_json(all_videos_json_files)
    per_frame_embed, per_frame_metadata = load_objective_dataset(json)
    clusterer = create_clusters(per_frame_embed)
    filtered_labels = filter_clusters(clusterer.labels_, per_frame_metadata)
    # Build mapping from cluster_idx -> list of all embeddings and frames in that cluster
    cluster_to_embeds = {} # K : V - K = cluster id, V is a list of all matching frame embeddings
    cluster_to_frames = {} # K : V - K = cluster id, V is a list of {"video_path", "frame_idx"} for each matched frame
    for meta_idx, cluster_idx in enumerate(filtered_labels):
        if cluster_idx > -1:
            cluster_to_embeds.setdefault(cluster_idx, []).append(torch.tensor(per_frame_embed[meta_idx]))
            cluster_to_frames.setdefault(cluster_idx, []).append({
                "video_path": per_frame_metadata[meta_idx]["video_path"],
                "frame_idx": round(per_frame_metadata[meta_idx]["sampled_frame_index"]),
            })
    valid_cluster_idxs = []
    for meta_idx, metadata in enumerate(per_frame_metadata):
        cluster_idx = filtered_labels[meta_idx]
        metadata['cluster_idx'] = cluster_idx
        if cluster_idx > -1:
            metadata['cluster_embed'] = cluster_to_embeds[cluster_idx]
            metadata['cluster_frames'] = cluster_to_frames[cluster_idx]
            valid_cluster_idxs.append(cluster_idx)
    return ObjectivesLookup(per_frame_metadata), AgentObjectiveManager(clusterer, valid_cluster_idxs)

def create_dataset(data_dir: str,
                   processor: Callable,
                   bootstrap: None | int,
                   split: float = 0.1,
                   max_videos = None,
                   query_embeds: list = [],
                   video_frames: list = [],
                   online: bool = True,
                   data_args = None
                   ) -> Tuple[Dataset, Dataset, AgentObjectiveManager]:
    
    videos_json = []
    if os.path.isdir(data_dir):
        all_videos_json_files = list_files_with_extensions(data_dir, ".json")
    else: 
        all_videos_json_files = [data_dir]

    # filter by bootstrap
    if bootstrap is not None:
        videos_json_files = list(filter(lambda x: f"bootstrap{bootstrap}" in x, all_videos_json_files))
        print(f"[AGENT] Creating dataset for bootstrap {bootstrap} with {len(videos_json_files)} files")
    else:
        videos_json_files = all_videos_json_files

    for json_file in videos_json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                if not any(vid["video_path"] == item["video_path"] for vid in videos_json):
                    videos_json.append({"video_path": item["video_path"]})

    # objective mining (rank 0 only, then broadcast)
    objects = [None, None]
    if dist.get_rank() == 0:
        print(f"[AGENT] Mining objectives for bootstrap {bootstrap} with {len(videos_json_files)} files")
        objects = list(mine_objectives(all_videos_json_files))
    dist.broadcast_object_list(objects, src=0)
    objectives_lookup, objective_manager = objects[0], objects[1]
    for idx, query_embed in enumerate(query_embeds):
        objective_manager.mine_and_add_objectives([t.cpu().numpy() for t in query_embed], video_frames[idx]) # Finds completed objectives in current trajectory

    if online:
        dataset = OnlineAgentDataset(videos_json, processor=processor, objectives_lookup=objectives_lookup, num_objectives=data_args.num_objectives)
    else:
        dataset = AgentPretrainingDataset(data_dir, processor=processor, objectives_lookup=objectives_lookup, num_objectives=data_args.num_objectives)

    train_ds, eval_ds = train_val_split(dataset, split=split)

    # error wrapping
    train_ds = ResampleDataset(train_ds)
    eval_ds = ResampleDataset(eval_ds)
    
    return train_ds, eval_ds, objective_manager

def train(model: nn.Module, training_args: TrainingArguments, train_ds: Dataset = None, eval_ds: Dataset = None) -> None:

    for param in model.parameters(): param.requires_grad = True
    trainer = Trainer(model=model, args=training_args, data_collator=OnlineAgentDataset.collate_fn, train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.train()

def train_with_rollback(model: nn.Module, training_args: TrainingArguments, train_ds: Dataset = None, eval_ds: Dataset = None) -> None:

    if eval_ds is None:
        train_ds, eval_ds = train_val_split(train_ds, split=0.05)

    for param in model.parameters(): param.requires_grad = True
    trainer = Trainer(model=model, args=training_args, data_collator=OnlineAgentDataset.collate_fn, train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.rollback_on_overfit(".cache/pokeagent/tmp_checkpoints")
    trainer.train()
    print(f"[RANK {dist.get_rank()} TRAINER] Agent training completes")
    trainer.run_rollback(model)

if __name__ == "__main__":

    init_distributed()
    
    model, processor, data_args, training_args = setup_training()
    print("setup complete")
    train_ds, eval_ds, agent_objective_manager = create_dataset(data_args.data_path, processor, None, split=0.05, online=False, data_args=data_args)
    print("created dataset")
    if dist.get_rank() == 0:
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, "objective_manager.pkl"), "wb") as f:
            pickle.dump(agent_objective_manager, f)
    train(model, training_args, train_ds=train_ds, eval_ds=eval_ds)