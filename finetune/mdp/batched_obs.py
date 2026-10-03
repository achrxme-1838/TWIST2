from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from obs_terms import StaticContext, term_dim
from policy_spec import PolicySpec, TermSpec


@dataclass
class BatchedObsContext:
    """One control tick for every env at once. Joint-order conventions match ObsContext."""
    static: StaticContext
    num_envs: int
    device: torch.device
    # --- proprioception (Isaac joint order) ---
    dof_pos_isaac: torch.Tensor = None        # [N, J]
    dof_vel_isaac: torch.Tensor = None        # [N, J]
    ang_vel: torch.Tensor = None              # [N, 3] base angular velocity (body frame)
    rpy: torch.Tensor = None                  # [N, 3] roll, pitch, yaw
    last_action: torch.Tensor = None          # [N, J] previous raw policy output (Isaac order)
    action_mimic: torch.Tensor = None         # [N, 38] motion target (SDK joint order)
    # --- simulator state / reference ---
    robot: Any = None                         # BatchedRobotState
    ref: Any = None                           # BatchedRefBodyState at the current motion time
    ref_frame: Any = None                     # BatchedMotionFrame at the current motion time
    future_frames: Optional[tuple] = None     # (pos [N,T,3], quat_wxyz [N,T,4], dof [N,T,J], lin [N,T,3], ang [N,T,3])
    future_body_pos_w: Optional[torch.Tensor] = None   # [N, n_fk, B_ext, 3]
    ref_root_xy_w: Optional[torch.Tensor] = None       # [N, 2]
    ref_fk: Any = None                        # BatchedRefFK, for terms that need their own FK pass
    anchor_delta_xy: Optional[torch.Tensor] = None     # [N, 2]
    motion_phase: torch.Tensor = None         # [N] t / motion length
    time_left: torch.Tensor = None            # [N] seconds until the motion ends
    feet_body_idx: tuple = (6, 12)            # TRACKED_BODY_NAMES indices of left/right ankle_roll_link
    _cache: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def n_fk(self) -> int:
        n = self.static.future_fk_steps
        return self.static.future_steps if n is None else int(n)


# --------------------------------------------------------------------------- registry
BTermFn = Callable[[BatchedObsContext, dict], torch.Tensor]   # fn(ctx, params) -> [N, dim]

REGISTRY: Dict[str, BTermFn] = {}


def bterm(name: str):
    """Register the batched implementation of the CPU obs term ``name``."""
    def deco(fn: BTermFn) -> BTermFn:
        if name in REGISTRY:
            raise KeyError(f"batched obs term {name!r} registered twice")
        REGISTRY[name] = fn
        return fn
    return deco


def get_bterm(name: str) -> BTermFn:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"obs term {name!r} has no batched implementation. Available: {sorted(REGISTRY)}"
        ) from None


# --------------------------------------------------------------------------- builder
class _BatchedTermSlot:
    """One spec term: its batched function, width, and ring-buffer history."""

    def __init__(self, spec: TermSpec, static: StaticContext, offset: int, num_envs: int,
                 device: torch.device):
        self.spec = spec
        self.fn = get_bterm(spec.name)
        self.dim = int(term_dim(spec.name, static, spec.params))
        self.history = spec.history_len
        self.offset = offset
        self.scale = None if spec.scale is None else float(spec.scale)
        self.clip = spec.clip
        if self.history > 1:
            self.buf = torch.zeros(num_envs, self.history, self.dim, device=device)
            self.ptr = self.history - 1
            # IsaacLab CircularBuffer: until the first push after a reset, every slot must hold
            # that first frame.
            self.fill = torch.ones(num_envs, dtype=torch.bool, device=device)
            self._order = torch.arange(self.history, device=device)
        else:
            self.buf = None

    @property
    def width(self) -> int:
        return self.dim * self.history

    def push(self, frame: torch.Tensor) -> torch.Tensor:
        if self.history == 1:
            return frame
        self.ptr = (self.ptr + 1) % self.history
        self.buf[:, self.ptr] = frame
        if bool(self.fill.any()):
            self.buf[self.fill] = frame[self.fill].unsqueeze(1).expand(-1, self.history, -1)
            self.fill.zero_()
        order = (self._order + self.ptr + 1) % self.history      # oldest .. newest
        return self.buf[:, order].reshape(frame.shape[0], self.width)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if self.history == 1:
            return
        if env_ids is None:
            self.fill[:] = True
        else:
            self.fill[env_ids] = True

    def rewind(self):
        """Step the write pointer back one slot, so the next push overwrites the frame the last
        push wrote instead of appending a new one."""
        if self.history > 1:
            self.ptr = (self.ptr - 1) % self.history


class BatchedObsBuilder:
    def __init__(self, spec: PolicySpec, static: StaticContext, num_envs: int, device: torch.device):
        self.spec = spec
        self.static = static
        self.num_envs = num_envs
        self.device = device
        self.slots: List[_BatchedTermSlot] = []
        offset = 0
        for t in spec.terms:
            slot = _BatchedTermSlot(t, static, offset, num_envs, device)
            self.slots.append(slot)
            offset += slot.width
        self.obs_dim = offset
        if spec.obs_dim is not None and spec.obs_dim != self.obs_dim:
            raise ValueError(f"policy spec says obs_dim={spec.obs_dim} but its terms add up to "
                             f"{self.obs_dim}:\n{self.describe()}")
        self._out = torch.zeros(num_envs, self.obs_dim, device=device)

    def __call__(self, ctx: BatchedObsContext) -> torch.Tensor:
        # the caller builds a fresh context per step, so its cache is shared between the
        # actor and critic builders -- the reference FK passes are done once, not twice.
        for slot in self.slots:
            frame = slot.fn(ctx, slot.spec.params)
            if frame.shape != (self.num_envs, slot.dim):
                raise ValueError(f"term '{slot.spec.name}' returned {tuple(frame.shape)}, "
                                 f"expected {(self.num_envs, slot.dim)}")
            if slot.scale is not None:
                frame = frame * slot.scale
            if slot.clip is not None:
                frame = frame.clamp(slot.clip[0], slot.clip[1])
            self._out[:, slot.offset:slot.offset + slot.width] = slot.push(frame)
        return self._out

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Forget history for these envs, so the next push refills their buffers."""
        for slot in self.slots:
            slot.reset(env_ids)

    def rewind(self):
        """Undo the last push's pointer advance (see _BatchedTermSlot.rewind)."""
        for slot in self.slots:
            slot.rewind()

    # ----- introspection -----
    def slices(self) -> Dict[str, slice]:
        return {s.spec.name: slice(s.offset, s.offset + s.width) for s in self.slots}

    def describe(self) -> str:
        lines = [f"obs layout (term-major, {self.obs_dim} dims):"]
        for s in self.slots:
            hist = f"history={s.history} x " if s.history > 1 else ""
            extra = []
            if s.scale is not None:
                extra.append(f"scale={s.scale}")
            if s.spec.params:
                extra.append(f"params={s.spec.params}")
            lines.append(f"  [{s.offset:5d}:{s.offset + s.width:5d}] {s.spec.name:32s} {hist}dim={s.dim} = {s.width}"
                         + (f"   ({', '.join(extra)})" if extra else ""))
        return "\n".join(lines)
