from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoricalClassifier(nn.Module):
    """Linear head that outputs raw logits over K classes."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)  # [..., K] raw logits
