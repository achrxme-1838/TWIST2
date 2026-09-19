"""Low-rank adapters for nn.Linear (SLowRL / LoRA).

    h = W0 x + b0 + (alpha / r) * B A x,   A ~ N(0, sigma^2) [r x in],  B = 0 [out x r]

W0 / b0 stay frozen; the policy starts exactly at the pre-trained behaviour (BA = 0).
``merge_lora`` folds the adapters back into plain Linear layers for ONNX export.
"""
from __future__ import annotations

import math
from typing import Iterator, List, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 1, alpha: float = 1.0, a_init_std: Optional[float] = None):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_f, out_f = base.in_features, base.out_features
        std = a_init_std if a_init_std is not None else 1.0 / math.sqrt(in_f)
        self.lora_A = nn.Parameter(torch.randn(rank, in_f, dtype=base.weight.dtype) * std)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, dtype=base.weight.dtype))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def merged_linear(self) -> nn.Linear:
        lin = nn.Linear(self.in_features, self.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(self.base.weight + self.delta_weight())
            if self.base.bias is not None:
                lin.bias.copy_(self.base.bias)
        return lin


def inject_lora(seq: nn.Sequential, rank: int, alpha: float, a_init_std: Optional[float] = None,
                layer_indices: Optional[List[int]] = None) -> nn.Sequential:
    """Replace the nn.Linear modules of ``seq`` (all, or those at ``layer_indices``
    counted over Linear layers only) with LoRALinear in place."""
    k = 0
    for i, m in enumerate(seq):
        if isinstance(m, nn.Linear):
            if layer_indices is None or k in layer_indices:
                seq[i] = LoRALinear(m, rank=rank, alpha=alpha, a_init_std=a_init_std)
            k += 1
    return seq


def merge_lora(seq: nn.Sequential) -> nn.Sequential:
    """Deep copy with every LoRALinear replaced by its merged nn.Linear."""
    mods = []
    for m in seq:
        if isinstance(m, LoRALinear):
            mods.append(m.merged_linear())
        else:
            mods.append(m)
    import copy
    return copy.deepcopy(nn.Sequential(*mods))


def lora_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    for m in module.modules():
        if isinstance(m, LoRALinear):
            yield m.lora_A
            yield m.lora_B


def count_parameters(params) -> int:
    return sum(p.numel() for p in params)
