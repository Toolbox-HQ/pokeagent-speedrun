from .random_policy import RandomPolicy, RandomMovementPolicy
from .policy import Policy, MGBA_KEY_LIST, CLASS_TO_KEY, KEY_TO_CLASS

def policy_map(x: str):
    if x == "random_policy":
        return RandomPolicy()
    elif x == "random_movement_policy":
        return RandomMovementPolicy()
    else:
        raise Exception("NotImplementedError")
    

__all__ = ["policy_map, Policy"]