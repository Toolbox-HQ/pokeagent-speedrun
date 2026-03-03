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
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=30),
        device_id=local_rank
    )

def get_shared_uuid() -> str:
    import uuid
    import torch.distributed as dist

    if dist.get_rank() == 0:
        obj_list = [str(uuid.uuid4())]   # create on rank 0
    else:
        obj_list = [None]                # placeholder

    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]

def clean_dist_and_exit():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

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
    import torch.nn.functional as F
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

        if logits.dim() == 3:
            B, T, C = logits.shape
            pos_correct = correct.mean(dim=0)
            dist.all_reduce(pos_correct, op=dist.ReduceOp.AVG)
            for i, v in enumerate(pos_correct.tolist()):
                accuracy[f"{prefix}acc_pos_{i}"] = v

            pos_loss = F.cross_entropy(logits.reshape(B * T, C), labels.reshape(B * T), reduction='none')
            pos_loss = pos_loss.view(B, T).mean(dim=0)
            dist.all_reduce(pos_loss, op=dist.ReduceOp.AVG)
            for i, v in enumerate(pos_loss.tolist()):
                accuracy[f"{prefix}loss_pos_{i}"] = v

    return accuracy