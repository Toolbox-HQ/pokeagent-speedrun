# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
from dataclasses import dataclass, field
from typing import  Optional, Tuple
import torch
from torch.utils.data import Subset, Dataset
import transformers
from models.util.trainer import Trainer
from torch.utils.data import Dataset
from models.model.agent_modeling.agent import init_lm_agent, init_vision_prcoessor
from models.util.repro import repro_init
from models.inference.idm_inference_dataloader import IDMWindowDataset, get_idm_labeller
import os
import random

def train_val_split(dataset: Dataset, split: float = 0.05)-> Tuple[Dataset, Dataset]:
    num_samples = len(dataset)
    indices = list(range(num_samples))
    eval_idx = random.sample(indices, round(num_samples*split))
    train_idx = [i for i in indices if i not in eval_idx]

    # train, eval
    return Subset(dataset=dataset, indices=train_idx), Subset(dataset=dataset, indices=eval_idx)

@dataclass
class ModelArguments:

    lm_name_or_path: Optional[str] = field(default=None)
    vision_name_or_path: Optional[str] = field(default=None)

@dataclass
class DataArguments:

    data_path: str | None = field(
        default=None, metadata={"help": "Path to the training data."}
    )

    subset: int | None = field(
        default=None,
        metadata={"help": "Choose a random subset of the dataset to train on"},
    )

    eval_data_path: str | None = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):

    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def train() -> None:
    global local_rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    
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

    model = init_lm_agent(lm=model_args.lm_name_or_path, vision=model_args.vision_name_or_path)
    processor = init_vision_prcoessor(vision=model_args.vision_name_or_path)
    model.idm_labelling_fn = get_idm_labeller(device)

    if training_args.gradient_checkpointing:
        model.text_model.gradient_checkpointing_enable()
        model.vision_tower.gradient_checkpointing_enable()
        training_args.gradient_checkpointing = False

    dataset = IDMWindowDataset(data_args.data_path)
    dataset.processor = processor
    train_ds, eval_ds = train_val_split(dataset, split=0.05)

    for param in model.parameters():
        param.requires_grad = True

    trainer = Trainer(
        model=model, args=training_args, data_collator=IDMWindowDataset.collate_fn, train_dataset=train_ds, eval_dataset=eval_ds
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    train()