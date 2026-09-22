"""Shared CLI plumbing for the check scripts."""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from finetune import _paths  # noqa: E402
from finetune.tasks.ft_cfg import FinetuneCfg, apply_overrides  # noqa: E402


def add_cfg_args(p, policy_arg: str = "student"):
    """``policy_arg`` names the checkpoint option (--student for the checks, --policy for evaluation.py)."""
    p.add_argument("--motion", default=None)
    p.add_argument("--name", default=None)
    p.add_argument(f"--{policy_arg}", dest="policy_pt", default=None,
                   help="policy checkpoint: student .pt (assets/pt) or a finetune model_N.pt")
    p.add_argument("--actor-spec", default=None)
    p.add_argument("--critic-spec", default=None)
    p.add_argument("--set", nargs="*", default=[])
    return p


def spec_from_finetune_ckpt(path: str):
    """actor_spec recorded in a finetune checkpoint's cfg; None for a plain student .pt."""
    import torch
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if isinstance(ck, dict) and "merged_actor_state_dict" in ck:
        return ck.get("cfg", {}).get("paths", {}).get("actor_spec") or None
    return None


def cfg_from_args(args, need_motion=True) -> FinetuneCfg:
    cfg = FinetuneCfg()
    if args.name:
        cfg.paths.name = args.name
    if args.motion:
        cfg.paths.motion_file = args.motion
    if args.policy_pt:
        cfg.paths.student_pt = args.policy_pt
        if not args.actor_spec:
            # a finetune model_N.pt carries the spec (and student) it was trained with
            spec = spec_from_finetune_ckpt(args.policy_pt)
            if spec:
                cfg.paths.actor_spec = spec
    if args.actor_spec:
        cfg.paths.actor_spec = args.actor_spec
    if args.critic_spec:
        cfg.paths.critic_spec = args.critic_spec
    apply_overrides(cfg, args.set)
    cfg.resolve()
    if need_motion and not cfg.paths.motion_file:
        raise SystemExit("--motion is required")
    _paths.setup(cfg.paths.rsl_rl_root)
    return cfg
