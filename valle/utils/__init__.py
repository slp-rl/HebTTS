import torch
import torch.nn as nn

from .symbol_table import SymbolTable
from .icefall import make_pad_mask
SymbolTable = SymbolTable


class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)
