import os
import argparse
from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import wandb
from tqdm.auto import tqdm
import time
from IDM.policy import InverseActionPolicy as IDModel
from transformers import HfArgumentParser

@dataclass
class Config:
    # Model
    torch_compile: bool = field(default=False)
    output_classes: int = field(default=None, help="Number of output classes, used to initialize classification head dim.")

    # Data
    batch_size: int = field(default=256)
    image_size: int = field(default=None)
    dataset_dir: str = field(default=None, help="Path to dataset, assumed to be in .cache subdirectory.")
    s3_bucket: str = field(default=None, help="S3 bucket name if data not in cache.")

    # Training
    epochs: int = field(default=10)
    lr: float = field(default=2e-4)
    wandb_project: str = field(default="pokeagent")

    # Output
    output_path: str = field(default="./checkpoints")


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    return local_rank, world_size


def main():

    # Parse config
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, required=True)
    args = arg_parser.parse_args()

    parser = HfArgumentParser(Config)
    (cfg,) = parser.parse_yaml_file(args.config)

    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        wandb.init(project=cfg.wandb_project, config=vars(cfg))

    # Dataset
    transform = T.v2.Compose(
        [
            T.v2.Resize((cfg.image_size, cfg.image_size), antialias=True),
            T.v2.ToImage(),  # Ensures we're dealing with an image type
            T.v2.ToDtype(torch.float32, scale=True),  # Like ToTensor(): 0–1 scaling
        ]
    )
    dataset = FastImageFolderNoLabels(
        cfg.dataset_root, cfg.image_size, transform=transform
    )
    sampler = DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    # SigLIP
    processor = AutoProcessor.from_pretrained(cfg.model_name_or_path, use_fast=True)
    siglip = SiglipModel.from_pretrained(
        cfg.model_name_or_path, attn_implementation="sdpa"
    ).to(device)
    if cfg.freeze_encoder:
        siglip.eval()
    else:
        siglip.train()

    # VQ + Decoder
    model = QuantizeDecode(dim=cfg.embed_dim, levels=cfg.levels).to(device)

    # Torch Compile (optional)
    if cfg.torch_compile:
        siglip = torch.compile(siglip, fullgraph=True)
        model = torch.compile(model, fullgraph=True)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    if cfg.loss_fn == "mse":
        loss_fn = nn.MSELoss()
    elif cfg.loss_fn == "cosine":
        loss_fn = cosine_loss

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
        for images in epoch_bar:
            images = images.to(device)

            with torch.no_grad():
                inputs = processor(images=images, return_tensors="pt").to(device)
                outputs: BaseModelOutputWithPooling = siglip.vision_model(**inputs)
                features: torch.Tensor = outputs.last_hidden_state  # [B, S, 768]

            reconstruction = model(features)
            loss = loss_fn(reconstruction, features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            elapsed = time.time() - start_time
            start_time = time.time()
            throughput = (cfg.batch_size * world_size) / elapsed

            if rank == 0:
                l1loss = (
                    torch.nn.functional.l1_loss(reconstruction, features)
                    .detach()
                    .item()
                )
                infloss = infinity_loss(reconstruction, features).detach().item()
                norm_l1 = normalized_l1(reconstruction, features).detach().item()
                norm_mse = normalized_mse(reconstruction, features).detach().item()

                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "loss": loss.item(),
                        "throughput": throughput,
                        "error_norm_1": l1loss,
                        "error_norm_inf": infloss,
                        "normalized_l1": norm_l1,
                        "normalized_mse": norm_mse,
                    }
                )
                epoch_bar.set_postfix_str(
                    f"loss={loss.item():.4f} | img/s={throughput:.1f} | iter={epoch_bar.n}/{epoch_bar.total}"
                )

    # Save model
    if rank == 0:
        os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
        model.module.save(cfg.output_path)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()