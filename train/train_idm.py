import os
import argparse
from dataclasses import dataclass, field
from typing import Tuple
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import wandb
from tqdm.auto import tqdm
import time
from IDM.policy import InverseActionPolicy as IDModel
from dataset import IDMDataset
from transformers import HfArgumentParser
from util.repro import repro_init
from policy import CLASS_TO_KEY
from util.data import reduce_dict
from pprint import pprint

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
    validation_dir: str = field(default=None)
    s3_bucket: str = field(default=None)


    # Training
    epochs: int = field(default=10)
    lr: float = field(default=2e-4)
    weight_decay: float = field(default=0.01)
    max_grad_norm: float = field(default=1.0)
    wandb_project: str = field(default="pokeagent")
    gradient_accumulation_steps: int = field(default=1)
    eval_every: int = field(default=None)
    scheduler: str = field(default=None)
    # Output
    output_path: str = field(default="model.pt")
    save_every: int = field(default=None)

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    return local_rank, world_size

def gather_and_stack(t: torch.Tensor):
    world_size = dist.get_world_size()
    gather_list = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(gather_list, t)
    return torch.stack(gather_list)


@torch.no_grad()
def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, is_val=False) -> float:
    s = "val_" if is_val else ""

    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).float()
    acc = correct.mean()
    dist.all_reduce(acc, op=dist.ReduceOp.AVG)
    accuracy = { f"{s}accuracy": acc.mean().item()}

    for label in list(CLASS_TO_KEY.keys()):
        l = gather_and_stack(labels)
        c = gather_and_stack(correct)
        c = c[l == label]
        accuracy[f"{s}acc_class_{CLASS_TO_KEY[label]}"] = c.mean().item() if c.numel() else -1

    return accuracy

@torch.no_grad()
def validate(model, val_loader, device, rank):

    model.eval()
    stats = []

    for inp, labels in val_loader:
        dummy = {
            "first": torch.zeros((inp.shape[0], 1)).to(device),
            "state_in": model.module.initial_state(inp.shape[0])
        }

        inp = {"img": inp.to(device=device)}
        labels = labels.to(dtype=torch.long, device=device)

        out = model(inp, labels=labels, **dummy)
        loss = out.loss
        logits = out.logits

        stats.append(compute_accuracy(logits, labels, is_val=True) | {"val_loss": loss.cpu().item()})

    
    if rank == 0:
        log = reduce_dict(stats)
        wandb.log(log)
        pprint(log)

    model.train()

def save(model, save_path, cfg):
    if dist.get_rank() == 0:
        path = os.path.join(save_path, cfg.output_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.module.state_dict(), path)
        print(f"Model saved to {path}")

def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, required=True)
    args = arg_parser.parse_args()
    save_path = repro_init(args.config)

    parser = HfArgumentParser(Config)
    cfg: Config = parser.parse_yaml_file(args.config)[0]

    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        wandb.init(project=cfg.wandb_project, config=vars(cfg))


    model = IDModel(output_classes=cfg.output_classes, fps=cfg.fps)
    #model.reset_parameters()
    model.to(device=device)

    # Training Dataset
    h,w = cfg.image_size
    dataset = IDMDataset(cfg.dataset_dir, h=h, w=w, fps = model.fps, s3_bucket=cfg.s3_bucket)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=16,
        pin_memory=True,
        collate_fn=IDMDataset.collate
    )

    val_loader = None
    val_dataset = IDMDataset(cfg.validation_dir, h=h, w=w, fps=model.fps, s3_bucket=cfg.s3_bucket)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=IDMDataset.collate
    )

    if cfg.torch_compile:
        model = torch.compile(model, fullgraph=True)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    avg_loss = 0
    avg_acc = 0
    global_step = 1
    total_steps = cfg.epochs * len(loader)
    scheduler = None

    if cfg.scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    else:
        from torch.optim.lr_scheduler import ConstantLR
        scheduler = ConstantLR(optimizer, factor=1, total_iters=total_steps)

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

            loss.backward()

            if global_step % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=cfg.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            metrics = compute_accuracy(logits, labels)
            elapsed = time.time() - start_time
            start_time = time.time()
            throughput = world_size / elapsed

            total_loss += loss.item()
            total_acc += metrics["accuracy"]
            num_batches += 1
            global_step += 1

            if rank == 0:
                wandb.log(
                    {
                        "epoch": epoch,
                        "lr": scheduler.get_last_lr(),
                        "loss_step": loss.item(),
                        "throughput": throughput,
                        "epoch_loss": avg_loss,
                        "epoch_accuracy": avg_acc,
                    } | metrics
                )
                epoch_bar.set_postfix_str(
                    f"loss={loss:.4f} | "
                    f"batch/s={throughput:.1f} | "
                    f"avg loss={avg_loss:.4f} | " 
                    f"avg acc={avg_acc:.2f} | "
                    f"iter={epoch_bar.n}/{epoch_bar.total}"
                )
                
            if cfg.save_every and global_step % cfg.save_every == 0:
                save(model, os.path.join(save_path,str(global_step)), cfg)

            if cfg.eval_every and global_step % cfg.eval_every == 0:
                validate(model, val_loader, device, rank)

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
    
    save(model, save_path, cfg)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()