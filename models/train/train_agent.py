# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
import json
from typing import Tuple, Callable
import torch
import torch.nn as nn
from torch.utils.data import Subset, Dataset
import transformers
from models.dataclass import TrainingArguments, DataArguments, ModelArguments
from models.util.trainer import Trainer
from torch.utils.data import Dataset
from models.model.agent_modeling.agent import init_lm_agent, init_vision_prcoessor
from models.util.repro import repro_init
from models.util.dist import init_distributed
from models.inference.idm_inference_dataloader import IDMWindowDataset, get_idm_labeller
import os
from models.util.data import train_val_split, list_files_with_extentions
import torch.distributed as dist

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
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
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

def create_dataset(data_dir: str, processor: Callable, split: float = 0.1) -> Tuple[Dataset, Dataset]:
    videos_json = []
    videos_json_files = list_files_with_extentions(data_dir, ".json")

    for json_file in videos_json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                if not any(vid["video_path"] == item["video_path"] for vid in videos_json):
                    videos_json.append({"video_path": item["video_path"]})

    dataset = IDMWindowDataset(videos_json)
    dataset.processor = processor
    train_ds, eval_ds = train_val_split(dataset, split=split)
    return train_ds, eval_ds

def train(model: nn.Module, training_args: TrainingArguments, train_ds: Dataset = None, eval_ds: Dataset = None) -> None:

    for param in model.parameters(): param.requires_grad = True
    trainer = Trainer(model=model, args=training_args, data_collator=IDMWindowDataset.collate_fn, train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.train()

def train_with_rollback(model: nn.Module, training_args: TrainingArguments, train_ds: Dataset = None, eval_ds: Dataset = None) -> None:

    if eval_ds is None:
        train_ds, eval_ds = train_val_split(train_ds, split=0.05)

    for param in model.parameters(): param.requires_grad = True
    trainer = Trainer(model=model, args=training_args, data_collator=IDMWindowDataset.collate_fn, train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.rollback_on_overfit(".cache/pokeagent/tmp_checkpoints")
    trainer.train()
    print(f"[RANK {dist.get_rank()} TRAINER] Agent training completes")
    trainer.run_rollback(model)

if __name__ == "__main__":

    init_distributed()
    
    model, processor, data_args, training_args = setup_training()
    train_ds, eval_ds = create_dataset(data_args.data_path, processor)
    train(model, training_args, train_ds=train_ds, eval_ds=eval_ds)