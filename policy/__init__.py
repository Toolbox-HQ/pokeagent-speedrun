from .random_policy import RandomPolicy
from .policy import Policy

def policy_map(x: str):
    if x == "random_policy":
        return RandomPolicy()
    else:
        raise Exception("NotImplementedError")
    

__all__ = ["policy_map, Policy"]