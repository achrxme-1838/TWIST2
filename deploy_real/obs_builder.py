"""Assemble the policy observation vector from a PolicySpec.

Mirrors IsaacLab's ObservationManager for one observation group with
``concatenate_terms=True``:

  * terms are evaluated in spec order and concatenated (term-major);
  * ``scale`` then ``clip`` are applied to each term's current frame;
  * a term with ``history`` N>1 keeps a CircularBuffer of its last N frames and
    contributes them flattened oldest..newest; on reset IsaacLab fills the whole
    buffer with the first pushed frame, which is reproduced here.

The term implementations live in ``obs_terms.py``; this module only knows their
names, widths and the spec.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np

from obs_terms import ObsContext, StaticContext, get_term
from policy_spec import PolicySpec, TermSpec


class _TermSlot:
    __slots__ = ("spec", "fn", "dim", "history", "buf", "offset")

    def __init__(self, spec: TermSpec, static: StaticContext, offset: int):
        self.spec = spec
        self.fn, dim_fn = get_term(spec.name)
        self.dim = int(dim_fn(static, spec.params))
        self.history = spec.history_len
        self.buf: Optional[Deque[np.ndarray]] = None   # filled on first frame
        self.offset = offset                            # start index in the flat obs

    @property
    def width(self) -> int:
        return self.dim * self.history

    def push(self, frame: np.ndarray) -> np.ndarray:
        if self.history == 1:
            return frame
        if self.buf is None:
            # IsaacLab CircularBuffer: after reset every slot holds the first frame.
            self.buf = deque([frame.copy() for _ in range(self.history)], maxlen=self.history)
        else:
            self.buf.append(frame)
        return np.concatenate(self.buf)     # oldest .. newest

    def reset(self):
        self.buf = None


class ObsBuilder:
    def __init__(self, spec: PolicySpec, static: StaticContext):
        self.spec = spec
        self.static = static
        self.slots: List[_TermSlot] = []
        offset = 0
        for t in spec.terms:
            slot = _TermSlot(t, static, offset)
            self.slots.append(slot)
            offset += slot.width
        self.obs_dim = offset
        if spec.obs_dim is not None and spec.obs_dim != self.obs_dim:
            raise ValueError(
                f"policy spec says obs_dim={spec.obs_dim} but its terms add up to "
                f"{self.obs_dim}:\n{self.describe()}"
            )

    # ----- per tick -----
    def __call__(self, ctx: ObsContext) -> np.ndarray:
        parts = []
        for slot in self.slots:
            frame = np.asarray(slot.fn(ctx, slot.spec.params), dtype=np.float32).reshape(-1)
            if frame.shape[0] != slot.dim:
                raise ValueError(f"term '{slot.spec.name}' returned {frame.shape[0]} values, expected {slot.dim}")
            if slot.spec.scale is not None:
                frame = frame * np.float32(slot.spec.scale)
            if slot.spec.clip is not None:
                frame = np.clip(frame, slot.spec.clip[0], slot.spec.clip[1])
            parts.append(slot.push(frame))
        obs = np.concatenate(parts).astype(np.float32, copy=False)
        assert obs.shape[0] == self.obs_dim, (obs.shape[0], self.obs_dim)
        return obs

    def reset(self):
        """Forget history (call on episode reset so buffers refill with the first frame)."""
        for slot in self.slots:
            slot.reset()

    # ----- introspection -----
    def slices(self) -> dict:
        """term name -> slice into the flat obs (handy for debugging / logging)."""
        return {s.spec.name: slice(s.offset, s.offset + s.width) for s in self.slots}

    def describe(self) -> str:
        lines = [f"obs layout (term-major, {self.obs_dim} dims):"]
        for s in self.slots:
            hist = f"history={s.history} x " if s.history > 1 else ""
            extra = []
            if s.spec.scale is not None:
                extra.append(f"scale={s.spec.scale}")
            if s.spec.params:
                extra.append(f"params={s.spec.params}")
            lines.append(f"  [{s.offset:5d}:{s.offset + s.width:5d}] {s.spec.name:32s} {hist}dim={s.dim} = {s.width}"
                         + (f"   ({', '.join(extra)})" if extra else ""))
        return "\n".join(lines)
