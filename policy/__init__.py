from .random_policy import RandomPolicy
from .policy import Policy, KEY_LIST, CLASS_TO_KEY, KEY_TO_CLASS

def policy_map(x: str):
    if x == "random_policy":
        return RandomPolicy()
    else:
        raise Exception("NotImplementedError")
    

__all__ = ["policy_map, Policy"]