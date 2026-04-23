import argparse
from dataclasses import dataclass, field
from typing import Tuple

import random

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from emulator.keys import CLASS_TO_KEY
from models.dataset import IDMDataset
from models.model.IDM.policy import InverseActionPolicy as IDModel


@dataclass
class ValidateIDMArguments:
    idm_checkpoint_path: str = field(
        default=None, metadata={"help": "Path to the trained IDM checkpoint (.pt)."}
    )
    idm_data_dir: str = field(
        default=None, metadata={"help": "Path to a folder of IDM data to evaluate on."}
    )
    output_classes: int = field(default=17)
    idm_fps: int = field(default=4)
    idm_image_size: Tuple[int, int] = field(default_factory=lambda: (128, 128))
    idm_batch_size: int = field(default=8)
    idm_dataloaders_per_device: int = field(default=4)


@torch.no_grad()
def run_validation(cfg: ValidateIDMArguments) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IDModel(output_classes=cfg.output_classes, fps=cfg.idm_fps)
    if cfg.idm_checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(cfg.idm_checkpoint_path)
    else:
        state_dict = torch.load(cfg.idm_checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device=device)
    model.eval()

    h, w = cfg.idm_image_size
    dataset = IDMDataset(
        data_path=cfg.idm_data_dir,
        h=h,
        w=w,
        fps=model.fps,
        is_val=True,
    )
    subset_size = max(1, int(len(dataset) * 0.1))
    indices = random.sample(range(len(dataset)), subset_size)
    dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=cfg.idm_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=IDMDataset.collate,
    )

    total_correct = 0
    total_count = 0
    skipped_batches = 0
    per_class_correct = {c: 0 for c in CLASS_TO_KEY.keys()}
    per_class_count = {c: 0 for c in CLASS_TO_KEY.keys()}

    loader_iter = iter(tqdm(loader, desc="Validating IDM"))
    while True:
        try:
            inp, labels = next(loader_iter)
        except StopIteration:
            break
        except IndexError as e:
            skipped_batches += 1
            print(f"[SKIPPED BATCH] {e}")
            continue

        try:
            batch = inp.shape[0]
            dummy = {
                "first": torch.zeros((batch, 1)).to(device),
                "state_in": model.initial_state(batch),
            }
            obs = {"img": inp.to(device=device)}
            labels = labels.to(dtype=torch.long, device=device)

            out = model(obs, labels=labels, **dummy)
            preds = torch.argmax(out.logits, dim=-1)
            correct = (preds == labels)

            total_correct += correct.sum().item()
            total_count += labels.numel()

            flat_labels = labels.reshape(-1)
            flat_correct = correct.reshape(-1)
            for c in per_class_count.keys():
                mask = flat_labels == c
                n = mask.sum().item()
                if n:
                    per_class_count[c] += n
                    per_class_correct[c] += flat_correct[mask].sum().item()
        except IndexError as e:
            skipped_batches += 1
            print(f"[SKIPPED BATCH] {e}")

    overall_accuracy = total_correct / max(total_count, 1)
    per_class_accuracy = {
        CLASS_TO_KEY[c]: (per_class_correct[c] / per_class_count[c])
        for c in per_class_count
        if per_class_count[c] > 0
    }

    movement_classes = {c for c, k in CLASS_TO_KEY.items() if k in ("up", "down", "left", "right")}
    movement_correct = sum(per_class_correct[c] for c in movement_classes)
    movement_count = sum(per_class_count[c] for c in movement_classes)
    movement_accuracy = movement_correct / max(movement_count, 1)

    return {
        "accuracy": overall_accuracy,
        "total_samples": total_count,
        "skipped_batches": skipped_batches,
        "per_class_accuracy": per_class_accuracy,
        "movement_accuracy": movement_accuracy,
        "movement_samples": movement_count,
    }


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, required=True)
    args = arg_parser.parse_args()

    parser = HfArgumentParser(ValidateIDMArguments)
    cfg: ValidateIDMArguments = parser.parse_yaml_file(args.config)[0]

    assert cfg.idm_checkpoint_path is not None, "idm_checkpoint_path is required"
    assert cfg.idm_data_dir is not None, "idm_data_dir is required"

    results = run_validation(cfg)

    print(f"\n=== IDM Validation Results ===")
    print(f"checkpoint: {cfg.idm_checkpoint_path}")
    print(f"data_dir:   {cfg.idm_data_dir}")
    print(f"samples:    {results['total_samples']}")
    print(f"skipped:    {results['skipped_batches']} batches")
    print(f"accuracy:   {results['accuracy']:.4f}")
    print(f"movement accuracy (up/down/left/right): {results['movement_accuracy']:.4f}  ({results['movement_samples']} samples)")
    print(f"per-class accuracy:")
    for key, acc in results["per_class_accuracy"].items():
        print(f"  {key}: {acc:.4f}")


if __name__ == "__main__":
    main()
