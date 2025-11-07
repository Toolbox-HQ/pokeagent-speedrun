from dataclasses import dataclass, field
from typing import List

@dataclass
class PolicyConfig:

    name: str = field(default=None)
    exclude: List[str] = field(default_factory=lambda: [])
    max_steps: int = field(default=100_000)