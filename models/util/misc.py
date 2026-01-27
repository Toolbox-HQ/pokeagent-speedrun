import os
hf_dir = os.environ["HF_HOME"] if "HF_HOME" in os.environ else os.path.expanduser("~/.cache/huggingface")

def local_model_map(model_name: str):
    return {
        "google/siglip-base-patch16-224": f"{hf_dir}/hub/models--google--siglip-base-patch16-224/snapshots/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
        "Qwen/Qwen3-1.7B": f"{hf_dir}/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "facebook/dinov2-base" : f"{hf_dir}/hub/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        "Toolbox-HQ/NitroSigLIP": f"{hf_dir}/hub/models--Toolbox-HQ--NitroSigLIP/snapshots/cc66f54289033d7ad08f7c4f93a40f7c5b7610c9",
    }[model_name]