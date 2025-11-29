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
from models.inference.idm_inference_dataloader import IDMWindowDataset, LabelledWindowDataset, get_idm_labeller
import os
import random
from pprint import pprint
from safetensors.torch import load_file
from models.dataclass import TrainingArguments, ModelArguments, DataArguments, EvalArguments

local_rank = None

def train_val_split(dataset: Dataset, split: float = 0.05)-> Tuple[Dataset, Dataset]:
    num_samples = len(dataset)
    indices = list(range(num_samples))
    eval_idx = random.sample(indices, round(num_samples*split))
    train_idx = [i for i in indices if i not in eval_idx]
    # train, eval
    return Subset(dataset=dataset, indices=train_idx), Subset(dataset=dataset, indices=eval_idx)

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def evaluate() -> None:
    global local_rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    
    from argparse import ArgumentParser

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    save_path = repro_init(args.config)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, EvalArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        eval_args,
    ) = parser.parse_yaml_file(yaml_file=args.config)

    local_rank = training_args.local_rank
    training_args.output_dir = save_path

    for arch, path in zip(eval_args.eval_architectures, eval_args.eval_checkpoints):

        model = init_lm_agent(arch=arch, lm=model_args.lm_name_or_path, vision=model_args.vision_name_or_path)
        processor = init_vision_prcoessor(vision=model_args.vision_name_or_path)
        model.idm_labelling_fn, idm = get_idm_labeller(device)

        
        print(f"[LOADING WEIGHTS] {path}")
        model.load_state_dict(load_file(path))

        # TODO add eval strcutures to config
        dataset = {"clock": LabelledWindowDataset(".cache/pokeagent/agent_eval_data/intervals_1f6ef6b5.json", processor = processor)}

        trainer = Trainer(
            model=model, args=training_args, data_collator=IDMWindowDataset.collate_fn, train_dataset=None, eval_dataset=dataset
        )

        pprint(trainer.evaluate())
    

if __name__ == "__main__":
    evaluate()