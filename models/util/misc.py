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

if __name__ == "__main__":
    download_models()