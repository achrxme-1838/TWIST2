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
    """actor_spec recorded in a checkpoint: a finetune ``model_N.pt`` (train.py) or an exported
    student ``.pt`` (tasks/export.py, which stores it under ``finetune``). None for a plain
    distillation student."""
    import torch
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(ck, dict):
        return None
    for cfg in (ck.get("cfg") if "merged_actor_state_dict" in ck else None,
                (ck.get("finetune") or {}).get("cfg")):
        spec = ((cfg or {}).get("paths") or {}).get("actor_spec")
        if spec and os.path.isfile(spec):
            return spec
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
