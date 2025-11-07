from typing import TypeVar
from transformers import HfArgumentParser
T = TypeVar("T")

# module imports
from mydataclass.emulator import PolicyConfig


def parse_dataclass(path: str, cls: T):
    parser = HfArgumentParser(cls)
    cfg : T = parser.parse_yaml_file(path)[0]
    return cfg

__all__ = ["parse_dataclass", "PolicyConfig"]