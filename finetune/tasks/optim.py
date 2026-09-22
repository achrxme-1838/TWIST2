from __future__ import annotations

from typing import Dict, Iterable, List

import torch


class _ScaledLRGroup(dict):
    """A param group whose ``["lr"] = x`` stores ``x * scale`` (or keeps ``base_lr`` when the
    group does not follow the schedule)."""

    def __init__(self, group: dict, scale: float, follow: bool = True):
        super().__init__(group)
        self.scale = float(scale)
        self.follow = bool(follow)

    def __setitem__(self, key, value):
        if key == "lr":
            value = float(value) * self.scale if self.follow else float(self["base_lr"])
        super().__setitem__(key, value)


def make_param_groups(named: Iterable[tuple]) -> List[dict]:
    """``[(name, params, lr), ...]`` -> Adam param groups carrying ``name`` and ``base_lr``
    (both are packed into the optimizer state dict, so they survive save / resume)."""
    groups = []
    for name, params, lr in named:
        params = list(params)
        if not params:
            continue
        groups.append({"params": params, "lr": float(lr), "base_lr": float(lr), "name": name})
    return groups


def bind_scaled_groups(optimizer: torch.optim.Optimizer, ref_group: str = "actor",
                       follow_groups: Iterable[str] | None = None) -> float:
    """Wrap ``optimizer.param_groups`` so rsl_rl's ``group["lr"] = learning_rate`` keeps the
    base-lr ratios for the groups in ``follow_groups`` (default: all) and leaves the others at
    their base lr. Returns the reference group's base lr (= what ``PPO.learning_rate`` must be
    set to). Call again after ``optimizer.load_state_dict`` (it rebuilds plain dicts)."""
    groups = optimizer.param_groups
    ref = next((g for g in groups if g.get("name") == ref_group), groups[0])
    ref_lr = float(ref["base_lr"])
    follow = None if follow_groups is None else set(follow_groups)
    wrapped = []
    for g in groups:
        base = dict(g)   # a previous wrapper's scale/follow are recomputed below
        wrapped.append(_ScaledLRGroup(base, float(base["base_lr"]) / ref_lr,
                                      follow=follow is None or base.get("name") in follow))
    optimizer.param_groups = wrapped
    return ref_lr


def group_lrs(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    return {g.get("name", f"g{i}"): float(g["lr"]) for i, g in enumerate(optimizer.param_groups)}


@torch.no_grad()
def gaussian_kl(old_mu: torch.Tensor, old_sigma: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> float:
    """Mean KL(old || new) between diagonal Gaussians -- the quantity rsl_rl adapts on."""
    kl = torch.sum(
        torch.log(sigma / old_sigma + 1.0e-5)
        + (torch.square(old_sigma) + torch.square(old_mu - mu)) / (2.0 * torch.square(sigma))
        - 0.5,
        dim=-1,
    )
    return float(kl.mean())
