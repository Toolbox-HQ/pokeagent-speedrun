import os
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import wandb
from tqdm.auto import tqdm
import time
from models.model.IDM.policy import InverseActionPolicy as IDModel
from models.dataset import IDMDataset
from transformers import HfArgumentParser
from models.util.repro import repro_init
from models.util.data import reduce_dict
from pprint import pprint
from models.dataclass import IDMArguments
from models.util.data import train_val_split, ResampleDataset
from models.util.dist import compute_accuracy, get_shared_uuid

def setup_distributed():

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    return local_rank, world_size

@torch.no_grad()
def validate(model, val_loader, device, rank):

    model.eval()
    stats = []
    total_loss = 0.0
    num_batches = 0

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

        stats.append(compute_accuracy(logits, labels, prefix="eval"))
        total_loss += loss.detach().item()
        num_batches += 1
    

    loss_tensor = torch.tensor([total_loss, num_batches], dtype=torch.float32, device=device)
    if dist.is_initialized():
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    global_total_loss, global_num_batches = loss_tensor.tolist()
    val_loss = global_total_loss / max(global_num_batches, 1.0)

    if rank == 0:
        log = reduce_dict(stats)
        log["val_loss"] = float(val_loss)
        wandb.log(log)
        pprint(log)
    
    model.train()
    return val_loss

def save(model, save_path, cfg):
    if dist.get_rank() == 0:
        path = os.path.join(save_path, cfg.output_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.module.state_dict(), path)
        print(f"Model saved to {path}")


def train_idm_best_checkpoint(model: torch.nn.Module, cfg: IDMArguments, dataset_path: str, split: float = 0.1):
    
    rank = dist.get_rank()
    device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    model.to(device=device)
    world_size = dist.get_world_size()

    if rank == 0:
        wandb.init(project="pokeagent", config=vars(cfg))
    
    h, w = cfg.idm_image_size
    dataset = IDMDataset(data_path=dataset_path,
                         h=h,
                         w=w,
                         fps = model.fps,
                         s3_bucket=cfg.s3_bucket,
                         apply_filter=True,
                         buffer_size=20, # buffer_size must be less than minimum action length
                        )
    
    train_ds, eval_ds = train_val_split(dataset, split=split)
    train_ds = ResampleDataset(train_ds)
    eval_ds = ResampleDataset(eval_ds)

    sampler = DistributedSampler(train_ds)
    loader = DataLoader(
        train_ds,
        batch_size=cfg.idm_batch_size,
        sampler=sampler,
        num_workers=cfg.idm_dataloaders_per_device,
        pin_memory=True,
        collate_fn=IDMDataset.collate
    )

    val_sampler = DistributedSampler(eval_ds, shuffle=False)
    val_loader = DataLoader(
        eval_ds,
        batch_size=cfg.idm_batch_size,
        sampler=val_sampler,
        num_workers=cfg.idm_dataloaders_per_device,
        pin_memory=True,
        collate_fn=IDMDataset.collate
    )

    wrapped_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=cfg.idm_lr, weight_decay=cfg.idm_weight_decay)

    avg_loss = 0
    avg_acc = 0
    global_step = 0
    total_steps = cfg.idm_epochs * len(loader)
    scheduler = None

    if cfg.idm_scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    else:
        from torch.optim.lr_scheduler import ConstantLR
        scheduler = ConstantLR(optimizer, factor=1, total_iters=total_steps)

    run_uuid = get_shared_uuid()
    tmp_ckpt_dir = os.path.join(".cache", "pokeagent", "tmp_checkpoints")
    if rank == 0:
        os.makedirs(tmp_ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_ckpt_path = None

    for epoch in range(cfg.idm_epochs):
        sampler.set_epoch(epoch)
        epoch_bar = tqdm(
            loader,
            desc=f"[Rank {rank}] Epoch {epoch + 1}",
            total=len(loader),
            disable=(rank != 0),
        )
        start_time = time.time()

        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for inp, labels in epoch_bar:
            # TODO refactor out useless but required inputs
            dummy = {
                "first": torch.zeros((inp.shape[0], 1)).to(device),
                "state_in": wrapped_model.module.initial_state(inp.shape[0])
            }

            # TODO refactor so that this isn't wrapped in a dict
            inp = {"img": inp.to(device=device)}
            labels = labels.to(dtype=torch.long, device=device)
            
            out = wrapped_model(inp, labels=labels, **dummy)
            loss = out.loss
            logits = out.logits

            loss.backward()

            if global_step % cfg.idm_gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    wrapped_model.parameters(),
                    max_norm=cfg.idm_max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            metrics = compute_accuracy(logits, labels, prefix="idm_")
            elapsed = time.time() - start_time
            start_time = time.time()
            throughput = world_size / elapsed

            total_loss += loss.item()
            total_acc += metrics["idm_accuracy"]
            
            

            if rank == 0:
                wandb.log(
                {
                "idm_epoch": epoch,
                "idm_lr": scheduler.get_last_lr(),
                "idm_loss_step": loss.item(),
                "idm_throughput": throughput,
                "idm_epoch_loss": avg_loss,
                "idm_epoch_accuracy": avg_acc,
                } | metrics
                )
                epoch_bar.set_postfix_str(
                f"loss={loss:.4f} | "
                f"batch/s={throughput:.1f} | "
                f"avg loss={avg_loss:.4f} | " 
                f"avg acc={avg_acc:.2f} | "
                f"iter={epoch_bar.n}/{epoch_bar.total}"
                )
            
            should_eval = False

            if cfg.idm_eval_every and isinstance(cfg.idm_eval_every, int)  \
            and global_step % cfg.idm_eval_every == 0:
                should_eval = True
            
            if cfg.idm_eval_every and isinstance(cfg.idm_eval_every, float) \
            and global_step % round(cfg.idm_eval_every * len(loader)) == 0:
                should_eval = True      


            if should_eval:
                val_loss = validate(wrapped_model, val_loader, device, rank)
                if rank == 0 and val_loss is not None:

                    ckpt_path = os.path.join(tmp_ckpt_dir, f"idm_{run_uuid}_step_{global_step}.pt")
                    torch.save(model.state_dict(), ckpt_path)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_ckpt_path = ckpt_path
            
            global_step += 1
            num_batches += 1


        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches

    dist.barrier()
    if best_ckpt_path is not None:
        state_dict = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[GPU {rank} IDM] best checkpoint was {best_ckpt_path}")
    dist.barrier()

def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, required=True)
    args = arg_parser.parse_args()
    save_path = repro_init(args.config)

    parser = HfArgumentParser(IDMArguments)
    cfg: IDMArguments = parser.parse_yaml_file(args.config)[0]

    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        wandb.init(project=cfg.wandb_project, config=vars(cfg))


    model = IDModel(output_classes=cfg.output_classes, fps=cfg.idm_fps)
    #model.reset_parameters()
    model.to(device=device)

    # Training Dataset
    h,w = cfg.idm_image_size
    dataset = IDMDataset(cfg.idm_dataset_dir, h=h, w=w, fps = model.fps, s3_bucket=cfg.s3_bucket)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=cfg.idm_batch_size,
        sampler=sampler,
        num_workers=cfg.idm_dataloaders_per_device,
        pin_memory=True,
        collate_fn=IDMDataset.collate
    )

    val_dataset = IDMDataset(cfg.idm_validation_dir, h=h, w=w, fps=model.fps, s3_bucket=cfg.s3_bucket, is_val=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.idm_batch_size,
        sampler=val_sampler,
        num_workers=cfg.idm_dataloaders_per_device,
        pin_memory=True,
        collate_fn=IDMDataset.collate
    )

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.idm_lr, weight_decay=cfg.idm_weight_decay)

    avg_loss = 0
    avg_acc = 0
    global_step = 1
    total_steps = cfg.epochs * len(loader)
    scheduler = None

    if cfg.idm_scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    else:
        from torch.optim.lr_scheduler import ConstantLR
        scheduler = ConstantLR(optimizer, factor=1, total_iters=total_steps)

    # Training
    for epoch in range(cfg.idm_epochs):
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

            if global_step % cfg.idm_gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=cfg.idm_max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            metrics = compute_accuracy(logits, labels, prefix="idm_")
            elapsed = time.time() - start_time
            start_time = time.time()
            throughput = world_size / elapsed

            total_loss += loss.item()
            total_acc += metrics["idm_accuracy"]
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
                
            if cfg.idm_save_every and global_step % cfg.idm_save_every == 0:
                save(model, os.path.join(save_path,str(global_step)), cfg)

            if cfg.idm_eval_every and global_step % cfg.idm_eval_every == 0:
                validate(model, val_loader, device, rank)

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
    
    save(model, save_path, cfg)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()