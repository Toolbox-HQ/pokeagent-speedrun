# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
import transformers
from utils.trainer import Trainer
from transformers.tokenization_utils_base import BatchEncoding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from tqdm import tqdm
from functools import reduce
from operator import mul


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)


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
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# TODO Fix this hacky tokenization
def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant.",
    padding="max_length",
    truncation=True,
    assistant_mask=False,
) -> BatchEncoding:
    inputs: BatchEncoding = tokenizer.apply_chat_template(
        sources,
        return_tensors="pt",
        add_generation_prompt=False,
        max_length=max_len,
        return_dict=True,
        padding=padding,
        truncation=truncation,
    )  # type: ignore

    inputs["attention_mask"] = inputs["attention_mask"].squeeze(dim=0)
    inputs["input_ids"] = inputs["input_ids"].squeeze(dim=0)

    input_ids: torch.Tensor = inputs["input_ids"]
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    eos_token_id = tokenizer.eos_token_id or 151643

    eos_indices = (input_ids == eos_token_id).nonzero(as_tuple=False).flatten()
    eos_mask = torch.ones_like(input_ids, dtype=torch.bool)
    if len(eos_indices) > 0:
        eos_mask[eos_indices[0]] = False
        eos_mask[eos_indices[1:]] = True

    if assistant_mask:
        im_start_indices = (input_ids == im_start_id).nonzero(as_tuple=False).flatten()
        second_im_start_pos = (
            im_start_indices[1].item() if len(im_start_indices) >= 2 else len(input_ids)
        )
        position_mask = torch.arange(len(input_ids)) >= second_im_start_pos
    else:
        position_mask = torch.ones_like(input_ids, dtype=torch.bool)

    final_mask = eos_mask & position_mask

    labels = input_ids.clone()
    labels[~final_mask] = -100
    inputs["labels"] = labels

    return inputs


def convert_single(datapoint):
    if "instruction" in datapoint and "response" in datapoint:
        return [
            {"role": "user", "content": datapoint["instruction"]},
            {"role": "assistant", "content": datapoint["response"]},
        ]
    elif "question" in datapoint and "answer" in datapoint:
        return [
            {"role": "user", "content": datapoint["question"]},
            {"role": "assistant", "content": datapoint["answer"]},
        ]
    else:
        raise Exception("KeyNotFoundError")


def format_as_conversations(ds):
    return [convert_single(i) for i in ds]


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        super(SupervisedDataset, self).__init__()
        import matplotlib.pyplot as plt

        rank0_print("Formatting inputs...")
        sources = format_as_conversations(raw_data)
        all_data = []

        for i in tqdm(sources, desc="Preprocessing"):
            all_data.append(
                preprocess(
                    i, tokenizer, max_len, padding="do_not_pad", truncation=False
                )
            )

        # Compute the "lengths" as product of shape dimensions
        lens = [reduce(mul, i["input_ids"].shape) for i in all_data]

        # Create histogram and save it
        # os.makedirs("./output", exist_ok=True)
        # plt.figure(figsize=(10, 6))
        # plt.hist(lens, bins=30, color='skyblue', edgecolor='black')
        # plt.title("Histogram of Input Lengths (flattened shape size)")
        # plt.xlabel("Flattened Length")
        # plt.ylabel("Count")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig("./output/input_length_histogram.png")

        # Optional: print max length for logging/debugging
        print(f"Maximum flattened input length in training data: {max(lens)}")
        raise Exception("NotSupportedError")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess(convert_single(self.raw_data[i]), self.tokenizer, self.max_len)
        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    ds = load_dataset(data_args.data_path)["train"]

    if data_args.subset is not None:
        print(f"Sampling {data_args.subset} data elements")
        from torch.utils.data import Subset
        import random

        random.seed(1234)
        ds = Subset(ds, random.sample(range(len(ds)), data_args.subset))

    train_dataset = dataset_cls(ds, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train() -> None:
    global local_rank
    from argparse import ArgumentParser

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_yaml_file(yaml_file=args.config)

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    is_chat_model = "chat" in model_args.model_name_or_path.lower()

    model = Qwen3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        # trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model.name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        # trust_remote_code=True,
    )

    if training_args.use_lora:
        raise Exception("NotImplemetedError")

    # if True:
    #     print("EXP: try usng torch compile")
    #     model = torch.compile(model, dynamic=True)

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # Start trainner
    trainer = Trainer(
        model=model, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    train()