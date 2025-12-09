def init_distributed(backend="nccl", timeout=None):
    import torch
    import torch.distributed as dist
    import os

    if dist.is_initialized():
        return

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timeout,
        device_id=local_rank
    )

def clean_dist_and_exit(_,__):
    import torch.distributed as dist
    dist.destroy_process_group()
    exit(0)