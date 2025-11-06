import subprocess
import os
import re
import warnings


def repro_init(cfg: str, seed: int = 1234):
    config_file = get_config_file(cfg)
    git_hash = enforce_versioning()
    seed_rng(seed=seed)
    return os.path.join("checkpoints", config_file, git_hash)


def get_config_file(path):
    config_name = re.search(r"/([^/]+?)(?:\.[^./]+)?$", path)
    assert config_name, "Did not find config file name in path."
    os.environ["WANDB_NAME"] = config_name.group(1)
    os.environ["WANDB_PROJECT"] = "pokeagent"
    return config_name.group(1)


def get_modified_files():
    return subprocess.run(
        ["git", "diff-index", "--name-only", "HEAD", "--"],
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def enforce_versioning():
    if "EXPERIMENT_RUN" in os.environ:
        try:
            subprocess.run(["git", "diff-index", "--quiet", "HEAD"], check=True)
        except Exception:
            warning_msg = f"""
            [WARNING] The following files were found to be modified but not committed in version control:
            {get_modified_files()}
            When running an experiment it is recommended to track all changes with version control.
            This behaviour is suggested by the [EXPERIMENT_RUN] environment variable.
            """
            warnings.warn(warning_msg)

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    return result.stdout.strip()

def seed_rng(seed: int = 1234):
    import random
    import torch
    import numpy as np
    import torch.distributed as dist

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    seed = seed + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    print(enforce_versioning())