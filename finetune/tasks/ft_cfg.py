from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import List, Optional

from finetune._paths import TWIST2_ROOT


@dataclass
class PathsCfg:
    name: str = "prop5+GMT+diff-pos-b_hist25_futu1"
    student_pt: str = ""       # default: assets/pt/<name>.pt
    actor_spec: str = ""       # default: assets/pt/<name>.yaml, then assets/ckpts/<name>.yaml
    critic_spec: str = os.path.join(TWIST2_ROOT, "finetune", "specs", "critic_default.yaml")
    motion_file: str = ""      # integrated joblib pkl (many motions) -- required
    xml: str = os.path.join(TWIST2_ROOT, "assets", "g1", "g1_sim2sim_29dof.xml")
    log_root: str = os.path.join(TWIST2_ROOT, "logs", "finetune")
    export_dir: str = os.path.join(TWIST2_ROOT, "assets", "ckpts")
    rsl_rl_root: Optional[str] = None   # DEX_RL_LAB_PHUMA dir; None -> auto (see _paths.py)

    def resolve(self):
        pt_dir = os.path.join(TWIST2_ROOT, "assets", "pt")
        if not self.student_pt:
            self.student_pt = os.path.join(pt_dir, f"{self.name}.pt")
        if not self.actor_spec:
            for cand in (os.path.join(pt_dir, f"{self.name}.yaml"),
                         os.path.join(TWIST2_ROOT, "assets", "ckpts", f"{self.name}.yaml")):
                if os.path.isfile(cand):
                    self.actor_spec = cand
                    break
            else:
                self.actor_spec = os.path.join(pt_dir, f"{self.name}.yaml")
        return self


@dataclass
class MotionCfg:
    name_filter: Optional[List[str]] = None
    max_motions: Optional[int] = None
    start_time_mode: str = "zero"       # zero (frame 0) / random
    align_to_ground: bool = False
    mimic_step_offset: int = 5
    anchor_origin_at_mimic_frame: bool = True


@dataclass
class EnvCfg:
    num_envs: int = 1
    control_hz: int = 50
    sim_dt: float = 0.001
    episode_length_s: float = 20.0
    action_scale: float = 0.25
    action_clip: float = 10.0
    kp_scale: float = 1.0
    kd_scale: float = 1.0
    reset_root_z_offset: float = 0.03
    realtime: bool = False
    contact_force_threshold: float = 1.0
    contact_body_exclude: List[str] = field(default_factory=lambda: [".*ankle.*", ".*wrist.*"])
    soft_joint_limit_factor: float = 0.9
    seed: int = 0


@dataclass
class RewardCfg:
    root_orientation_w: float = 0.5
    root_orientation_sigma: float = 0.4 ** 2
    body_pos_w: float = 20.0
    body_pos_sigma: float = 0.05 ** 2
    body_ori_w: float = 1.0
    body_ori_sigma: float = 0.4 ** 2
    body_lin_vel_w: float = 1.0
    body_lin_vel_sigma: float = 1.0 ** 2
    body_ang_vel_w: float = 1.0
    body_ang_vel_sigma: float = 3.14 ** 2
    action_rate_w: float = -0.3
    joint_pos_limits_w: float = -10.0
    undesired_contacts_w: float = -0.1
    scale_by_dt: bool = True


@dataclass
class TerminationCfg:
    max_ref_motion_dist: float = 0.5
    diverge_any_body: bool = True   # True = training mode (any body) / False: mean


@dataclass
class LoRACfg:
    mode: str = "lora"      # lora / fft
    rank: int = 1
    alpha: float = 1.0
    a_init_std: Optional[float] = None
    lr: float = 1e-2
    fft_lr: float = 1e-3


@dataclass
class CriticCfg:
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    lr: float = 1e-3
    warmup_iters: int = 200


@dataclass
class PPOCfg:
    num_steps_per_env: int = 1024
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    use_clipped_value_loss: bool = True
    max_grad_norm: float = 1.0
    schedule: str = "fixed"
    desired_kl: float = 0.01
    normalize_advantage_per_mini_batch: bool = False


@dataclass
class TrainCfg:
    max_iterations: int = 3000
    save_interval: int = 100
    log_interval: int = 1
    device: str = "cpu"
    init_noise_std: Optional[float] = 0.2
    std_trainable: bool = True
    std_lr: Optional[float] = 1e-3
    resume: str = ""


@dataclass
class FinetuneCfg:
    paths: PathsCfg = field(default_factory=PathsCfg)
    motion: MotionCfg = field(default_factory=MotionCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    lora: LoRACfg = field(default_factory=LoRACfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    ppo: PPOCfg = field(default_factory=PPOCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    def resolve(self) -> "FinetuneCfg":
        self.paths.resolve()
        return self

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ----------------------------------------------------------------------------- overrides
def _cast(value: str, current):
    """Cast a CLI string to the type of the field it replaces."""
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        return [type(current[0])(v) if current else v for v in value.split(",")] if value else []
    if value.lower() in ("none", "null"):
        return None
    return value


def apply_overrides(cfg: FinetuneCfg, items: List[str]) -> FinetuneCfg:
    """``section.field=value`` overrides (e.g. ``env.num_envs=4 lora.rank=2``)."""
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must look like section.field=value: {item}")
        key, value = item.split("=", 1)
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = getattr(node, p)
        name = parts[-1]
        if not hasattr(node, name):
            raise AttributeError(f"unknown cfg field: {key}")
        current = getattr(node, name)
        if current is None:
            # untyped None default: infer from the declared field type
            ftype = next(f.type for f in dataclasses.fields(node) if f.name == name)
            ftype = str(ftype)
            if "int" in ftype:
                current = 0
            elif "float" in ftype:
                current = 0.0
            elif "List" in ftype or "list" in ftype:
                current = [""]
            else:
                current = ""
        setattr(node, name, _cast(value, current))
    return cfg
