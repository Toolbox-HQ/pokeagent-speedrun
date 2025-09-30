
def clean_dist_and_exit(_,__):
    import torch.distributed as dist
    dist.destroy_process_group()
    exit(0)