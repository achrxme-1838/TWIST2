"""Per-policy observation spec."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class TermSpec:
    name: str
    history: int = 1
    scale: Optional[float] = None
    clip: Optional[Tuple[float, float]] = None
    params: Dict[str, Any] = field(default_factory=dict)   # plain ObsTerm params (e.g. mask_joint_names)

    @property
    def history_len(self) -> int:
        return max(int(self.history), 1)

    def num_future_steps(self, default: int) -> int:
        """future_motion_* terms consume the first ``num_steps`` of the published
        future frames (older runs: all of them -> ``default`` = spec.future_steps)."""
        n = self.params.get("num_steps")
        return int(default if n is None else n)


PELVIS = "pelvis"


@dataclass
class RootFrameSpec:
    """Frame the policy's root-dependent obs were trained in (export_deploy_cfg ``root`` / ``root_frame``).

    The FK / odometry root on the deploy side is always the pelvis; a non-pelvis root is a frame
    rigidly attached to it: ``X = pelvis * (pos, quat)``. Runs exported before this field existed
    have neither key and are pelvis-rooted.
    """
    name: str = PELVIS
    parent_body: str = PELVIS
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)   # wxyz, w.r.t. the parent body

    @property
    def is_pelvis(self) -> bool:
        """True if the frame coincides with the pelvis (no offset needed)."""
        return (self.parent_body == PELVIS and all(v == 0.0 for v in self.pos)
                and tuple(self.quat) == (1.0, 0.0, 0.0, 0.0))

    def describe(self) -> str:
        if self.is_pelvis:
            return f"root: {self.name}"
        return f"root: {self.name} = {self.parent_body} * (pos={list(self.pos)}, quat={list(self.quat)})"


def parse_root_frame(data: dict, path: Optional[str] = None) -> RootFrameSpec:
    """Read ``root`` / ``root_frame`` of a spec. Missing -> pelvis (older exports)."""
    where = f"policy spec {path or ''}"
    name = str(data.get("root") or PELVIS)
    raw = data.get("root_frame")
    if raw is None:
        if name != PELVIS:
            raise ValueError(f"{where}: root '{name}' has no 'root_frame' (re-export with export_deploy_cfg.py)")
        return RootFrameSpec()
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: 'root_frame' must be a dict, got {raw!r}")

    parent = str(raw.get("parent_body") or PELVIS)
    if parent != PELVIS:
        # the deploy FK root (odometry / IMU chain) is the pelvis
        raise ValueError(f"{where}: root_frame.parent_body must be '{PELVIS}', got '{parent}'")
    pos = raw.get("pos") or [0.0, 0.0, 0.0]
    quat = raw.get("quat") or [1.0, 0.0, 0.0, 0.0]
    if len(pos) != 3 or len(quat) != 4:
        raise ValueError(f"{where}: root_frame needs pos[3] and quat[4] (wxyz), got pos={pos} quat={quat}")
    pos = tuple(float(v) for v in pos)
    quat = tuple(float(v) for v in quat)
    norm = math.sqrt(sum(v * v for v in quat))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"{where}: root_frame.quat {list(quat)} is not unit length (|q|={norm:.4f})")
    quat = tuple(v / norm for v in quat)
    return RootFrameSpec(name=name, parent_body=parent, pos=pos, quat=quat)


@dataclass
class PolicySpec:
    terms: List[TermSpec]
    future_steps: int = 1
    future_interval: int = 1
    obs_dim: Optional[int] = None
    path: Optional[str] = None
    source_run: Optional[str] = None
    root_frame: RootFrameSpec = field(default_factory=RootFrameSpec)

    # ----- derived -----
    @property
    def term_names(self) -> List[str]:
        return [t.name for t in self.terms]

    def has_prefix(self, prefix: str) -> bool:
        return any(t.name.startswith(prefix) for t in self.terms)

    @property
    def needs_ref_fk(self) -> bool:
        """diff_body_* terms need FK on the current reference frame."""
        return self.has_prefix("diff_body_")

    @property
    def needs_future(self) -> bool:
        """future_motion_* terms need the motion server's future frames."""
        return self.has_prefix("future_motion_")

    @property
    def future_motion_steps(self) -> List[int]:
        """Env-step offsets of the future frames: (i+1)*interval, i=0..T-1."""
        return [(i + 1) * self.future_interval for i in range(self.future_steps)]

    @property
    def used_future_steps(self) -> int:
        """Largest ``num_steps`` any future_motion_* term consumes (0 if none). The
        motion server still publishes all T frames; only the first this many matter."""
        return max((t.num_future_steps(self.future_steps) for t in self.terms
                    if t.name.startswith("future_motion_")), default=0)

    def describe_future(self) -> str:
        n = self.used_future_steps
        steps = self.future_motion_steps
        if n == 0:
            return "future frames: none used"
        return (f"future frames: {n} of {self.future_steps} published used "
                f"(+{steps[:n]} control steps ahead)")

    def describe(self) -> str:
        lines = [f"PolicySpec({self.path or '<inline>'})"]
        if self.source_run:
            lines.append(f"  source_run: {self.source_run}")
        lines.append(f"  future_steps={self.future_steps} interval={self.future_interval} obs_dim={self.obs_dim}")
        lines.append(f"  {self.describe_future()}")
        lines.append(f"  {self.root_frame.describe()}")
        for t in self.terms:
            extra = []
            if t.history_len > 1:
                extra.append(f"history={t.history_len}")
            if t.scale is not None:
                extra.append(f"scale={t.scale}")
            if t.clip is not None:
                extra.append(f"clip={list(t.clip)}")
            if t.params:
                extra.append(f"params={t.params}")
            lines.append(f"  - {t.name}" + (f" ({', '.join(extra)})" if extra else ""))
        return "\n".join(lines)


def parse_spec(data: dict, path: Optional[str] = None) -> PolicySpec:
    if not isinstance(data, dict) or "terms" not in data:
        raise ValueError(f"policy spec {path or ''} has no 'terms' list")
    terms = []
    for i, raw in enumerate(data["terms"]):
        if isinstance(raw, str):
            raw = {"name": raw}
        if not isinstance(raw, dict) or "name" not in raw:
            raise ValueError(f"policy spec term #{i} must be a name or a dict with 'name': {raw!r}")
        clip = raw.get("clip")
        terms.append(TermSpec(
            name=str(raw["name"]),
            history=int(raw.get("history") or 1),
            scale=None if raw.get("scale") is None else float(raw["scale"]),
            clip=None if clip is None else (float(clip[0]), float(clip[1])),
            params=dict(raw.get("params") or {}),
        ))
    return PolicySpec(
        terms=terms,
        future_steps=int(data.get("future_steps") or 1),
        future_interval=int(data.get("future_interval") or 1),
        obs_dim=None if data.get("obs_dim") is None else int(data["obs_dim"]),
        path=path,
        source_run=data.get("source_run"),
        root_frame=parse_root_frame(data, path),
    )


def load_spec(path: str) -> PolicySpec:
    with open(path) as f:
        data = yaml.safe_load(f)
    return parse_spec(data, path=path)


def spec_path_for_policy(policy_path: str) -> str:
    """`<dir>/<name>.onnx` -> `<dir>/<name>.yaml`."""
    root, _ = os.path.splitext(policy_path)
    return root + ".yaml"


def resolve_spec(policy_path: str, obs_cfg: Optional[str] = None) -> PolicySpec:
    """Explicit --obs_cfg wins; otherwise look for the yaml next to the ONNX."""
    path = obs_cfg or spec_path_for_policy(policy_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"policy spec not found: {path}\n"
            f"Export one with DEX_RL_LAB_PHUMA/scripts/export_deploy_cfg.py --run <run> --out {os.path.splitext(policy_path)[0]}"
        )
    return load_spec(path)
