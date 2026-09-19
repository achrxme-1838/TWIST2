"""Shared CLI plumbing for the check scripts."""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from finetune import _paths  # noqa: E402
from finetune.tasks.ft_cfg import FinetuneCfg, apply_overrides  # noqa: E402


def add_cfg_args(p):
    p.add_argument("--motion", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--student", default=None)
    p.add_argument("--actor-spec", default=None)
    p.add_argument("--critic-spec", default=None)
    p.add_argument("--set", nargs="*", default=[])
    return p


def cfg_from_args(args, need_motion=True) -> FinetuneCfg:
    cfg = FinetuneCfg()
    if args.name:
        cfg.paths.name = args.name
    if args.motion:
        cfg.paths.motion_file = args.motion
    if args.student:
        cfg.paths.student_pt = args.student
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
