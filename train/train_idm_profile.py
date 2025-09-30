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

import logging
import socket
from datetime import datetime, timedelta

logging.basicConfig(
   format="%(levelname)s:%(asctime)s %(message)s",
   level=logging.INFO,
   datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


def trace_handler(prof: torch.profiler.profile):
   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"./{host_name}_{timestamp}"
   out = f"{file_prefix}.html"
   # Construct the memory timeline file.
   prof.export_memory_timeline(out, device="cuda:0")
   print(f"exported {out}")

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
    (cfg,) = parser.parse_yaml_file(args.config)
    
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        wandb.init(project=cfg.wandb_project, config=vars(cfg))

    # Enable memory recording before training
    if rank == 0:  # only need snapshots on one rank usually
        torch.cuda.memory._record_memory_history(max_entries=100000)

    model = IDModel(output_classes=cfg.output_classes, fps=cfg.fps)
    model.to(device=device)

    h, w = cfg.image_size
    dataset = IDMDataset(cfg.dataset_dir, h=h, w=w, fps=model.fps)
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
    from torch.autograd.profiler import record_function
    count = 0
    cfg.epochs = 1
    # Training
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=trace_handler,
    ) as prof:
        for epoch in range(cfg.epochs):
            sampler.set_epoch(epoch)
            epoch_bar = tqdm(
                loader,
                desc=f"[Rank {rank}] Epoch {epoch + 1}",
                total=len(loader),
                disable=(rank != 0),
            )
            start_time = time.time()
            for inp, labels in epoch_bar:
                prof.step()

                dummy = {
                    "first": torch.zeros((inp.shape[0], 1)).to(device),
                    "state_in": model.module.initial_state(inp.shape[0])
                }

                inp = {"img": inp.to(device)}
                labels = labels.to(dtype=torch.long, device=device)
                with record_function("## forward ##"):
                    out = model(inp, labels=labels, **dummy)
                loss = out.loss
                logits = out.logits
                
                with record_function("## backward ##"):
                    loss.backward()
                
                
                with record_function("## optimizer ##"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                accuracy = compute_accuracy(logits, labels)
                elapsed = time.time() - start_time
                start_time = time.time()
                throughput = world_size / elapsed

                if rank == 0:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "loss": loss.item(),
                            "accuracy": accuracy,
                            "throughput": throughput,
                        }
                    )
                    epoch_bar.set_postfix_str(
                        f"loss={loss:.4f} | "
                        f"acc={accuracy:.2f} | "
                        f"batch/s={throughput:.1f} | "
                        f"iter={epoch_bar.n}/{epoch_bar.total}"
                    )
                count += 1
                if count == 3:
                    break

    # Save model
    if rank == 0:
        # Dump memory snapshot after training
        torch.cuda.memory._dump_snapshot(".cache/memory_snapshot.pickle")

        # Stop memory history recording
        torch.cuda.memory._record_memory_history(enabled=None)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()