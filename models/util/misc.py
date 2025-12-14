import os
from typing import List

hf_dir = os.environ["HF_HOME"] if "HF_HOME" in os.environ else os.path.expanduser("~/.cache/huggingface")

def local_model_map(model_name: str):
    return {
        "google/siglip-base-patch16-224": f"{hf_dir}/hub/models--google--siglip-base-patch16-224/snapshots/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
        "Qwen/Qwen3-1.7B": f"{hf_dir}/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "facebook/dinov2-base" : f"{hf_dir}/hub/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415"
    }[model_name]

def download_models():
    from transformers import AutoModel, AutoProcessor
    AutoModel.from_pretrained("google/siglip-base-patch16-224")
    AutoModel.from_pretrained("Qwen/Qwen3-1.7B")
    AutoModel.from_pretrained("facebook/dinov2-base")

    AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    AutoProcessor.from_pretrained("Qwen/Qwen3-1.7B")
    AutoProcessor.from_pretrained("facebook/dinov2-base")

def finalize_wandb(tags: List[str] = []):
    import torch.distributed as dist
    if dist.get_rank() == 0 and "WANDB_MODE" in os.environ and os.environ["WANDB_MODE"] == "offline":
        import wandb
        run = wandb.run
        run.tags = tags
        run.finish()
    dist.barrier()

if __name__ == "__main__":
    download_models()