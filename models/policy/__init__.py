from .random_policy import RandomPolicy, RandomMovementPolicy
from .policy import Policy
from emulator.keys import MGBA_KEY_LIST, CLASS_TO_KEY, KEY_TO_CLASS, NUM_ACTION_CLASSES

def policy_map(name: str, exclude=None):
    if name == "random_policy":
        return RandomPolicy(exclude=exclude)
    elif name == "random_movement_policy":
        return RandomMovementPolicy(exclude=exclude)
    else:
        raise ValueError(f"Unknown policy: {name}")

__all__ = [
    "policy_map",
    "Policy",
    "MGBA_KEY_LIST",
    "CLASS_TO_KEY",
    "KEY_TO_CLASS",
    "NUM_ACTION_CLASSES",
]
