from __future__ import annotations

import os
import sys
import types

FINETUNE_DIR = os.path.dirname(os.path.abspath(__file__))
TWIST2_ROOT = os.path.dirname(FINETUNE_DIR)
DEPLOY_REAL_DIR = os.path.join(TWIST2_ROOT, "deploy_real")

_RSL_RL_CANDIDATES = [
    os.environ.get("DEX_RSL_RL_ROOT", ""),
    os.path.join(os.path.dirname(TWIST2_ROOT), "DEX_RL_LAB_PHUMA"),   # local
    "/root/isaaclab_ws/DEX_RL_LAB_PHUMA",                              # docker mount
]

_done = False


def _prepend(path: str):
    if path and path not in sys.path:
        sys.path.insert(0, path)


def resolve_rsl_rl_root(explicit: str | None = None) -> str:
    for cand in [explicit or ""] + _RSL_RL_CANDIDATES:
        if cand and os.path.isfile(os.path.join(cand, "rsl_rl", "algorithms", "ppo.py")):
            return cand
    raise FileNotFoundError(
        "DEX_RL_LAB_PHUMA (rsl_rl 2.x) not found. Set DEX_RSL_RL_ROOT or cfg.paths.rsl_rl_root "
        f"to the directory that contains 'rsl_rl/'. Tried: {_RSL_RL_CANDIDATES}"
    )


def setup(rsl_rl_root: str | None = None) -> str:
    """Put deploy_real, TWIST2 root and the DEX rsl_rl on sys.path. Returns the rsl_rl root."""
    global _done
    root = resolve_rsl_rl_root(rsl_rl_root)
    if _done:
        return root
    _prepend(TWIST2_ROOT)
    _prepend(DEPLOY_REAL_DIR)
    try:
        import git  # noqa: F401
    except ImportError:
        stub = types.ModuleType("git")
        stub.Repo = None
        sys.modules["git"] = stub
    if "rsl_rl" in sys.modules and not sys.modules["rsl_rl"].__file__.startswith(root):
        raise ImportError(
            f"rsl_rl already imported from {sys.modules['rsl_rl'].__file__}; "
            "call finetune._paths.setup() before importing rsl_rl."
        )
    _prepend(root)
    import rsl_rl  # noqa: F401
    if not os.path.abspath(rsl_rl.__file__).startswith(os.path.abspath(root)):
        raise ImportError(
            f"'import rsl_rl' resolved to {rsl_rl.__file__}, not the DEX_RL_LAB_PHUMA copy under {root}. "
            "Another rsl_rl (e.g. TWIST2/rsl_rl 1.0.2) shadows it; uninstall it or fix sys.path."
        )
    _done = True
    return root
