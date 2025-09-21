from .random_policy import RandomPolicy

def policy_map(x: str):
    if x == "random_policy":
        return RandomPolicy()
    else:
        raise Exception("NotImplementedError")