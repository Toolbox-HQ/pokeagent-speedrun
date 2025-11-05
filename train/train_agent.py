# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
from dataclasses import dataclass, field
from typing import  Optional
import torch
from torch.utils.data import Dataset
import transformers
from util.trainer import Trainer
from torch.utils.data import Dataset
from model.agent_modeling.agent import init_lm_agent, init_vision_prcoessor
from util.repro import repro_init

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


class DummyData(Dataset):

    def __init__(self):
        self.processor = None

    def __len__(self):
        return 6000

    # tensor uint8 (64 x 3 x 224 x 224) , int32 tensor(64) 
    def __getitem__(self, _):
        # data, labels
        t = torch.zeros((64,3,224,224), dtype=torch.uint8),
        labels = torch.ones((64), dtype=torch.int32)                

        if self.processor:
            t = self.processor.image_processor(t, return_tensors="pt")

        return  t, labels
    
    @staticmethod
    def collate_fn(batch):
        pixels, inps = [], []

        for pixel, ids in batch:
            pixels.append(pixel["pixel_values"])
            inps.append(ids)

        return {
            "pixel_values": torch.stack(pixels, dim=0),
            "input_ids": torch.stack(inps, dim=0),
        }
    

def train() -> None:
    global local_rank
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

    # model stuff
    model = init_lm_agent(lm=model_args.lm_name_or_path, vision=model_args.vision_name_or_path)
    processor = init_vision_prcoessor(vision=model_args.vision_name_or_path)
    
    if training_args.gradient_checkpointing:
        model.text_model.gradient_checkpointing_enable()
        model.vision_tower.gradient_checkpointing_enable()
        training_args.gradient_checkpointing = False

    # data stuff
    training_dataset = DummyData()
    training_dataset.processor = processor

    # Start trainer
    trainer = Trainer(
        model=model, args=training_args, data_collator=DummyData.collate_fn, train_dataset=training_dataset
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    train()