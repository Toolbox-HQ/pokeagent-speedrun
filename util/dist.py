
def clean_dist_and_exit(_,__):

    print("[SIGNAL HANDLER] signal received - destorying process group")
    import torch.distributed as dist
    dist.destroy_process_group()
    print("[SIGNAL HANDLER] process group destroyed - exiting")
    exit(0)