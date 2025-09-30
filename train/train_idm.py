import os
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import wandb
from tqdm.auto import tqdm
import time
from IDM.policy import InverseActionPolicy as IDModel
from dataset import IDMDataset
from transformers import HfArgumentParser
from util.repro import repro_init

@dataclass
class Config:
    # Model
    torch_compile: bool = field(default=False)
    output_classes: int = field(default=None)
    fps: int = field(default=4)

    # Data
    batch_size: int = field(default=256)
    image_size: Tuple[int, int] = field(default_factory=lambda: (128, 128))
    dataset_dir: str = field(default=None)
    s3_bucket: str = field(default=None)


    # Training
    epochs: int = field(default=10)
    lr: float = field(default=2e-4)
    weight_decay: float = field(default=0.01)
    max_grad_norm: float = field(default=1.0)
    activation_checkpoint: bool = field(default=False)
    wandb_project: str = field(default="pokeagent")

    # Output
    output_path: str = field(default="./checkpoints")


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    return local_rank, world_size

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == labels).float()
        accuracy = correct.mean().item()
    return accuracy

def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, required=True)
    args = arg_parser.parse_args()
    repro_init(args.config)

    parser = HfArgumentParser(Config)
    cfg: Config = parser.parse_yaml_file(args.config)[0]

    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        wandb.init(project=cfg.wandb_project, config=vars(cfg))


    model = IDModel(output_classes=cfg.output_classes, fps=cfg.fps)
    #model.reset_parameters()
    model.to(device=device)


    h,w = cfg.image_size
    dataset = IDMDataset(cfg.dataset_dir, h=h, w=w, fps = model.fps, s3_bucket=cfg.s3_bucket)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=IDMDataset.collate
    )


    if cfg.torch_compile:
        model = torch.compile(model, fullgraph=True)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    avg_loss = -1
    avg_acc = -1

    # Training
    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        epoch_bar = tqdm(
            loader,
            desc=f"[Rank {rank}] Epoch {epoch + 1}",
            total=len(loader),
            disable=(rank != 0),
        )
        start_time = time.time()

        # Track metrics for epoch averages
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for inp, labels in epoch_bar:
            # TODO refactor out useless but required inputs
            dummy = {
                "first": torch.zeros((inp.shape[0], 1)).to(device),
                "state_in": model.module.initial_state(inp.shape[0])
            }

            # TODO refactor so that this isn't wrapped in a dict
            inp = {"img": inp.to(device=device)}
            labels = labels.to(dtype=torch.long, device=device)
            
            out = model(inp, labels=labels, **dummy)
            loss = out.loss
            logits = out.logits

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=cfg.max_grad_norm
            )

            optimizer.step()
            accuracy = compute_accuracy(logits, labels)
            elapsed = time.time() - start_time
            start_time = time.time()
            throughput = world_size / elapsed

            total_loss += loss.item()
            total_acc += accuracy
            num_batches += 1

            if rank == 0:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "loss_step": loss.item(),
                        "accuracy_step": accuracy,
                        "throughput": throughput,
                    }
                )
                epoch_bar.set_postfix_str(
                    f"loss={loss:.4f} | "
                    f"acc={accuracy:.2f} | "
                    f"batch/s={throughput:.1f} | "
                    f"avg loss={avg_loss:.4f}" 
                    f"avg acc={avg_acc:.2f}"
                    f"iter={epoch_bar.n}/{epoch_bar.total}"
                )

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches

        if rank == 0:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "loss_epoch": avg_loss,
                    "accuracy_epoch": avg_acc,
                }
            )
            # Update tqdm bar one last time with epoch averages
            epoch_bar.set_postfix_str(
                f"[Epoch Avg] loss={avg_loss:.4f} | acc={avg_acc:.2f}"
            )

    # Save model
    if rank == 0:
        os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
        model.module.save(cfg.output_path)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()