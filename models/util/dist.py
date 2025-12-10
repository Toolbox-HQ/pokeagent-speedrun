def init_distributed(backend="nccl"):
    import torch
    import torch.distributed as dist
    import os
    from datetime import timedelta
    
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
        timeout=timedelta(minutes=30),
        device_id=local_rank
    )

def clean_dist_and_exit(_,__):
    import torch.distributed as dist
    dist.destroy_process_group()
    exit(0)