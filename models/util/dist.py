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

def gather_and_stack(t):
    import torch
    import torch.distributed as dist

    t = t.contiguous()
    world_size = dist.get_world_size()
    gather_list = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(gather_list, t)
    return torch.stack(gather_list)

def compute_accuracy(logits, labels, prefix: str = "") -> float:
    import torch
    import torch.distributed as dist
    from models.policy import CLASS_TO_KEY

    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == labels).float()
        acc = correct.mean()
        dist.all_reduce(acc, op=dist.ReduceOp.AVG)
        accuracy = { f"{prefix}accuracy": acc.mean().item()}

        for label in list(CLASS_TO_KEY.keys()):
            l = gather_and_stack(labels)
            c = gather_and_stack(correct)
            c = c[l == label]
            if c.numel():
                accuracy[f"{prefix}acc_class_{CLASS_TO_KEY[label]}"] = c.mean().item()

    return accuracy