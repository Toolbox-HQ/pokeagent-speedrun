# TODO ideally change this to not be hard mapped.
# used to redirect load paths on non-networked HPC nodes.
def local_model_map(model_name: str):
    return {
        "google/siglip-base-patch16-224": "/scratch/bsch/hf_cache/hub/models--google--siglip-base-patch16-224/snapshots/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
        "Qwen/Qwen3-1.7B": "/scratch/bsch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    }[model_name]