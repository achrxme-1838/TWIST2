"""Adaptive tracking factor for the exponential tracking rewards (gpu backend).

Port of DEX_RL_LAB_PHUMA ``dex_rl_lab/manager_wrapper/adaptive_sigma.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch

from finetune.tasks.ft_cfg import UpdateSigmaCfg

Sigma = Union[float, torch.Tensor]


class AdaptiveSigma:
    """Tracking factors tightened from an EMA of the tracking error."""

    _QUANTILES = (0.25, 0.75)

    def __init__(self, cfg: UpdateSigmaCfg, sigma_init: Dict[str, float], num_motions: int,
                 num_envs: int, device: torch.device, motion_names: Optional[List[str]] = None):
        """
        Args:
            cfg: ``UpdateSigmaCfg``, already validated by ``FinetuneCfg.resolve()``.
            sigma_init: Initial tracking factor per reward term name, as resolved into
                ``RewardCfg``. Seeds the factors, and the error EMA in global mode.
            num_motions: Size of the motion set; ``mid`` passed to ``begin_step`` indexes it.
            num_envs: Number of envs.
            device: Device of the reward tensors.
            motion_names: Motion order the per-motion state is indexed by. Checked on resume.
        """
        self._alpha = cfg.alpha
        self._min_sigma = cfg.min_sigma
        self._per_motion = bool(cfg.per_motion)
        self._term_names = list(sigma_init)
        self._num_motions = int(num_motions)
        self._motion_names = list(motion_names) if motion_names is not None else None
        self._device = device

        if not self._per_motion:
            # 0-dim tensors: they broadcast against the [N] error exactly as the float did.
            self._sigma = {name: torch.tensor(float(s), device=device) for name, s in sigma_init.items()}
            self._error_ema = {name: torch.tensor(float(s), device=device) for name, s in sigma_init.items()}
            return

        self._t_warmup = int(cfg.t_warmup)
        self._sigma = {
            name: torch.full((num_motions,), float(sigma), device=device)
            for name, sigma in sigma_init.items()
        }
        # Zero-initialized, as the bias correction in _update_per_motion() assumes.
        self._error_acc = {
            name: torch.zeros(num_motions, device=device) for name in sigma_init
        }
        # One counter covers every term: every adapted term is computed on every step, so
        # their visit counts coincide.
        self._visits = torch.zeros(num_motions, dtype=torch.long, device=device)
        self._env_counts = torch.zeros(num_motions, device=device)
        self._error_sum = torch.zeros(num_motions, device=device)
        self._env_ones = torch.ones(num_envs, device=device)
        self._motion_gids: Optional[torch.Tensor] = None
        self._visited_ids: Optional[torch.Tensor] = None

    def __contains__(self, term_name: str) -> bool:
        return term_name in self._term_names

    @property
    def term_names(self) -> List[str]:
        return list(self._term_names)

    @property
    def per_motion(self) -> bool:
        return self._per_motion

    def describe(self) -> str:
        scope = f"per motion (t_warmup={self._t_warmup})" if self._per_motion else "shared across motions"
        lines = [f"adaptive sigma: {scope}, alpha={self._alpha:g}, min_sigma={self._min_sigma:g}"]
        for name in self._term_names:
            s = float(self._sigma[name].flatten()[0])
            lines.append(f"  {name:30s} sigma_init={s:.6g}")
        return "\n".join(lines)

    def begin_step(self, mid: torch.Tensor):
        """Refresh the env -> motion mapping and the visit counters for this step.

        A no-op in global mode. Must run before ``sigma()``, which gathers from the mapping
        stored here.

        Args:
            mid: [N] motion id each env is tracking on this step.
        """
        if not self._per_motion:
            return

        self._motion_gids = mid
        # Envs per motion. Motions are drawn with replacement, so a motion is generally
        # tracked by more than one env.
        self._env_counts.zero_()
        self._env_counts.index_add_(0, mid, self._env_ones)
        self._visited_ids = torch.nonzero(self._env_counts, as_tuple=True)[0]
        self._visits[self._visited_ids] += 1

    def sigma(self, term_name: str) -> Sigma:
        """The factor this step's envs are rewarded with: 0-dim (global) or [N] (per motion)."""
        if not self._per_motion:
            return self._sigma[term_name]
        return self._sigma[term_name][self._motion_gids]

    def update(self, term_name: str, error: torch.Tensor):
        """Fold this step's tracking error into the EMA and tighten the factor.

        Args:
            term_name: Name of the reward term.
            error: [N] tracking error x of the term's exp(-x / sigma).
        """
        if self._per_motion:
            self._update_per_motion(term_name, error)
        else:
            self._update_global(term_name, error)

    def _update_global(self, term_name: str, error: torch.Tensor):
        ema = self._error_ema[term_name]
        ema += self._alpha * (error.mean() - ema)
        sigma = self._sigma[term_name]
        sigma.copy_(torch.minimum(ema, sigma).clamp_min(self._min_sigma))

    def _update_per_motion(self, term_name: str, error: torch.Tensor):
        visited = self._visited_ids
        if visited.numel() == 0:
            return

        # Motions tracked by several envs average those envs' errors.
        self._error_sum.zero_()
        self._error_sum.index_add_(0, self._motion_gids, error)
        error_mean = self._error_sum[visited] / self._env_counts[visited]

        alpha = self._alpha
        acc = self._error_acc[term_name]
        acc[visited] = (1.0 - alpha) * acc[visited] + alpha * error_mean

        visits = self._visits[visited]
        error_ema = acc[visited] / (1.0 - (1.0 - alpha) ** visits.to(acc.dtype))

        warm = visits >= self._t_warmup
        ready = visited[warm]
        if ready.numel() == 0:
            return
        sigma = self._sigma[term_name]
        sigma[ready] = torch.minimum(error_ema[warm], sigma[ready]).clamp_min(self._min_sigma)

    def log_dict(self) -> Dict[str, float]:
        """Diagnostics: the live factor and the error estimate behind it."""
        if self._per_motion:
            return self._log_dict_per_motion()

        flat = torch.stack([v for name in self._term_names
                            for v in (self._sigma[name], self._error_ema[name])]).tolist()
        extras = {}
        for i, term_name in enumerate(self._term_names):
            extras["Sigma/" + term_name] = flat[2 * i]
            extras["SigmaErrorEMA/" + term_name] = flat[2 * i + 1]
        return extras

    def _log_dict_per_motion(self) -> Dict[str, float]:
        """Distribution of the factors over the motions seen so far.

        There is one factor per motion, far too many to log individually, and motions that
        have never been sampled would only dilute the summary with their initial value.
        """
        seen = torch.nonzero(self._visits, as_tuple=True)[0]
        if seen.numel() == 0:
            return {}

        quantiles = torch.tensor(self._QUANTILES, device=seen.device)
        correction = 1.0 - (1.0 - self._alpha) ** self._visits[seen].to(torch.float)
        warm = (self._visits[seen] >= self._t_warmup).to(torch.float).sum()

        stats = []
        for term_name in self._term_names:
            sigma = self._sigma[term_name][seen]
            error_ema = self._error_acc[term_name][seen] / correction
            stats.append(torch.cat([
                torch.stack([sigma.mean(), sigma.amin(), sigma.amax(), error_ema.mean()]),
                torch.quantile(sigma, quantiles),
            ]))
        stats.append(torch.stack([torch.tensor(float(seen.numel()), device=seen.device), warm]))
        # One host transfer for the whole summary, instead of one per statistic.
        flat = torch.cat(stats).tolist()

        extras = {}
        for i, term_name in enumerate(self._term_names):
            mean, lo, hi, error_ema, p25, p75 = flat[6 * i: 6 * i + 6]
            extras["Sigma/" + term_name] = mean
            extras["SigmaErrorEMA/" + term_name] = error_ema
            extras["SigmaMin/" + term_name] = lo
            extras["SigmaMax/" + term_name] = hi
            extras["SigmaP25/" + term_name] = p25
            extras["SigmaP75/" + term_name] = p75
        extras["Sigma/motions_seen"] = flat[-2]
        extras["Sigma/motions_warm"] = flat[-1]
        return extras

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        state = {
            "per_motion": self._per_motion,
            "num_motions": self._num_motions,
            "motion_names": self._motion_names,
            "sigma": {k: v.detach().cpu().clone() for k, v in self._sigma.items()},
        }
        if self._per_motion:
            state["error_acc"] = {k: v.detach().cpu().clone() for k, v in self._error_acc.items()}
            state["visits"] = self._visits.detach().cpu().clone()
        else:
            state["error_ema"] = {k: v.detach().cpu().clone() for k, v in self._error_ema.items()}
        return state

    def load_state_dict(self, state: dict):
        """Restore a ``state_dict()``. Raises if it was saved under a different setup, since the
        factors would then belong to other terms or other motions."""
        if bool(state["per_motion"]) != self._per_motion:
            raise ValueError(f"adaptive sigma state per_motion={state['per_motion']} "
                             f"but update_sigma.per_motion={self._per_motion}")
        if set(state["sigma"]) != set(self._term_names):
            raise ValueError(f"adaptive sigma state terms {sorted(state['sigma'])} "
                             f"!= update_sigma.terms {sorted(self._term_names)}")
        if self._per_motion:
            if int(state["num_motions"]) != self._num_motions:
                raise ValueError(f"adaptive sigma state has {state['num_motions']} motions, "
                                 f"the motion set has {self._num_motions}")
            if (state.get("motion_names") is not None and self._motion_names is not None
                    and list(state["motion_names"]) != self._motion_names):
                raise ValueError("adaptive sigma state was saved with a different motion order "
                                 "(motion_file / motion.name_filter / motion.max_motions changed?)")

        for k in self._term_names:
            self._sigma[k].copy_(state["sigma"][k].to(self._device))
            if self._per_motion:
                self._error_acc[k].copy_(state["error_acc"][k].to(self._device))
            else:
                self._error_ema[k].copy_(state["error_ema"][k].to(self._device))
        if self._per_motion:
            self._visits.copy_(state["visits"].to(self._device))
