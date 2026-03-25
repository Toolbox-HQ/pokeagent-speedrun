import os
import sys
import io
import traceback
from typing import List
import glob

hf_dir = os.environ["HF_HOME"] if "HF_HOME" in os.environ else os.path.expanduser("~/.cache/huggingface")

def local_model_map(model_name: str):
    return {
        "google/siglip-base-patch16-224": f"{hf_dir}/hub/models--google--siglip-base-patch16-224/snapshots/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
        "Qwen/Qwen3-1.7B": f"{hf_dir}/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "facebook/dinov2-base" : f"{hf_dir}/hub/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        "Toolbox-HQ/NitroSigLIP": f"{hf_dir}/hub/models--Toolbox-HQ--NitroSigLIP/snapshots/cc66f54289033d7ad08f7c4f93a40f7c5b7610c9",
        "Qwen/Qwen3-VL-8B-Instruct": f"{hf_dir}/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
    }[model_name]

def download_models():
    from transformers import AutoModel, AutoProcessor
    AutoModel.from_pretrained("google/siglip-base-patch16-224")
    AutoModel.from_pretrained("Qwen/Qwen3-1.7B")
    AutoModel.from_pretrained("facebook/dinov2-base")
    AutoModel.from_pretrained("Toolbox-HQ/NitroSigLIP")

    AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    AutoProcessor.from_pretrained("Qwen/Qwen3-1.7B")
    AutoProcessor.from_pretrained("facebook/dinov2-base")
    AutoProcessor.from_pretrained("Toolbox-HQ/NitroSigLIP")

def collect_query_files(output_dir: str, bootstrap_count: int) -> List[str]:
    import torch.distributed as dist
    
    query_files = []
    if dist.get_rank() == 0:
        query_video_dir = os.path.join(output_dir, 'query_video')
        pattern = os.path.join(query_video_dir, f'query_gpu*_bootstrap{bootstrap_count}.*')
        query_files = glob.glob(pattern)
        print(f"[GPU {dist.get_rank()} LOOP] Found {len(query_files)} query files for bootstrap {bootstrap_count}")
    
    return query_files

def finalize_wandb(tags: List[str] = []):
    import torch.distributed as dist
    if dist.get_rank() == 0:
        import wandb
        run = wandb.run
        if tags:
            run.tags = tags        
        run.finish()
    dist.barrier()

def inject_traceback():
    """
    Monkey-patches traceback.print_exc() to skip printing if the traceback
    contains any NFS-related "Device or resource busy" errors.
    
    Call this function once at the start of your program to inject
    the filtered version globally.
    """
    _original_print_exc = traceback.print_exc
    
    def _filtered_print_exc(file=None, limit=None, chain=True):
        f = io.StringIO()
        _original_print_exc(file=f, limit=limit, chain=chain)
        output = f.getvalue()
        
        if "Device or resource busy" in output and ".nfs" in output:
            return
        
        if file is None:
            file = sys.stderr
        file.write(output)
    
    traceback.print_exc = _filtered_print_exc

def print_gpu_memory(label: str = "", rank: int = None):
    import torch
    import torch.distributed as dist
    import pynvml
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
    allocated = torch.cuda.memory_allocated(local_rank) / 1024**3
    reserved = torch.cuda.memory_reserved(local_rank) / 1024**3
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    driver_used = info.used / 1024**3
    driver_total = info.total / 1024**3
    prefix = f"[MEM rank{rank}{' ' + label if label else ''}]"
    print(f"{prefix} torch allocated: {allocated:.2f} GB | torch reserved: {reserved:.2f} GB | nvidia driver used: {driver_used:.2f} / {driver_total:.2f} GB")

if __name__ == "__main__":
    download_models()