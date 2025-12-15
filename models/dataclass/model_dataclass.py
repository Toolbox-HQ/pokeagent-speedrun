from dataclasses import dataclass, field
from typing import Optional, List, Tuple
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

@dataclass
class IDMArguments:
    # Model
    output_classes: int = field(default=None)
    idm_fps: int = field(default=4)

    # Data
    idm_batch_size: int = field(default=256)
    idm_image_size: Tuple[int, int] = field(default_factory=lambda: (128, 128))
    idm_dataset_dir: str = field(default=None)
    idm_validation_dir: str = field(default=None)
    s3_bucket: str = field(default=None)
    idm_dataloaders_per_device: int = field(default=1)

    # Training
    idm_epochs: int = field(default=10)
    idm_lr: float = field(default=2e-4)
    idm_weight_decay: float = field(default=0.01)
    idm_max_grad_norm: float = field(default=1.0)
    wandb_project: str = field(default="pokeagent")
    idm_gradient_accumulation_steps: int = field(default=1)
    idm_eval_every: int | float = field(default=None)
    idm_scheduler: str = field(default=None)
    # Output
    idm_output_path: str = field(default="model.pt")
    idm_save_every: int = field(default=None)
