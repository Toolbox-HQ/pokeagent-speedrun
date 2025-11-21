from dataclasses import dataclass, field
from typing import Optional, List
import transformers

@dataclass
class ModelArguments:
    architecture: Optional[str] = field(default=None)
    lm_name_or_path: Optional[str] = field(default=None)
    vision_name_or_path: Optional[str] = field(default=None)
    load_path: Optional[str] = field(default=None)

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


@dataclass
class EvalArguments:
    eval_checkpoints: List[str] = field(default_factory = lambda: [])
    eval_architectures: List[str] = field(default_factory = lambda: [])