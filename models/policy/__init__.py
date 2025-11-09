from .random_policy import RandomPolicy, RandomMovementPolicy
from .policy import Policy, MGBA_KEY_LIST, CLASS_TO_KEY, KEY_TO_CLASS, NUM_ACTION_CLASSES
from models.dataclass import parse_dataclass, PolicyConfig

def policy_map(x: str):

    cfg = parse_dataclass(x, PolicyConfig)
    policy_name = cfg.name

    if policy_name == "random_policy":
        return RandomPolicy(cfg)
    elif policy_name == "random_movement_policy":
        return RandomMovementPolicy(cfg)
    else:
        raise Exception("NotImplementedError")
    

__all__ = [
    "policy_map",
    "Policy",
    "MGBA_KEY_LIST",
    "CLASS_TO_KEY",
    "KEY_TO_CLASS",
    "NUM_ACTION_CLASSES"
    ]