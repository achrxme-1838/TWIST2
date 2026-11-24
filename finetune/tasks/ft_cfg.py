from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from finetune._paths import TWIST2_ROOT


@dataclass
class PathsCfg:
    name: str = "prop5+GMT+diff-pos-b_hist25_futu1"
    student_pt: str = ""       # default: assets/pt/<name>.pt
    actor_spec: str = ""       # default: assets/pt/<name>.yaml, then assets/ckpts/<name>.yaml
    critic_spec: str = ""      # default by critic.source: specs/critic_default.yaml (scratch) / critic_teacher.yaml (teacher)
    motion_file: str = "~/isaaclab_ws/DEX_RL_LAB_PHUMA/data_processing/motion_retargeting/data/integrated_motions/PHUMA_rt-v30_ap0.99.pkl"
    xml: str = os.path.join(TWIST2_ROOT, "assets", "g1", "g1_sim2sim_29dof.xml")
    log_root: str = os.path.join(TWIST2_ROOT, "logs", "finetune")
    export_dir: str = os.path.join(TWIST2_ROOT, "assets", "ckpts", "ft")              # ONNX (+ spec) for the deploy servers
    export_pt_dir: str = os.path.join(TWIST2_ROOT, "assets", "pt_finetuned")    # rsl_rl .pt for IsaacLab play.py
    rsl_rl_root: Optional[str] = None   # DEX_RL_LAB_PHUMA dir; None -> auto (see _paths.py)

    def resolve(self, critic_source: str = "scratch"):
        pt_dir = os.path.join(TWIST2_ROOT, "assets", "pt")
        ckpt_dir = os.path.join(TWIST2_ROOT, "assets", "ckpts")
        if not self.critic_spec:
            fname = "critic_teacher.yaml" if critic_source == "teacher" else "critic_default.yaml"
            self.critic_spec = os.path.join(TWIST2_ROOT, "finetune", "specs", fname)
        if self.student_pt and not self.actor_spec:
            # An explicit student .pt names its own spec (<dir>/<stem>.yaml, then assets/ckpts/
            # <stem>.yaml). Never fall back to ``name`` here: a wrong spec with the same obs
            # width (e.g. diff_body_pos_b vs diff_body_pos_b_deploy) is not caught by the dim
            # check and silently breaks the policy.
            stem = os.path.splitext(os.path.basename(self.student_pt))[0]
            for cand in (os.path.join(os.path.dirname(os.path.abspath(self.student_pt)), f"{stem}.yaml"),
                         os.path.join(pt_dir, f"{stem}.yaml"), os.path.join(ckpt_dir, f"{stem}.yaml")):
                if os.path.isfile(cand):
                    self.actor_spec = cand
                    if self.name == PathsCfg.name:
                        self.name = stem
                    break
        if not self.student_pt:
            self.student_pt = os.path.join(pt_dir, f"{self.name}.pt")
        if not self.actor_spec:
            for cand in (os.path.join(pt_dir, f"{self.name}.yaml"), os.path.join(ckpt_dir, f"{self.name}.yaml")):
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
    num_envs: int = 4096
    control_hz: int = 50
    sim_dt: float = 0.001
    episode_length_s: float = 20.0
    action_scale: float = 0.25
    action_clip: float = 10.0
    kp_scale: float = 1.0
    kd_scale: float = 1.0
    reset_root_z_offset: float = 0.03
    realtime: bool = False
    backend: str = "gpu"                # cpu (env/mujoco_env.py) / gpu (mujoco_warp, env/batched_mujoco_env.py)
    contact_force_threshold: float = 1.0
    contact_body_exclude: List[str] = field(default_factory=lambda: [".*ankle.*", ".*wrist.*"])
    soft_joint_limit_factor: float = 0.9
    seed: int = 0


@dataclass
class RewardCfg:
    terms: Dict[str, float] = field(default_factory=lambda: {
        # -- task
        "root_orientation":             0.5,
        # "mimic_extended_body_pos_exp":  5.0,
        "mimic_extended_upper_body_pos_exp": 3.0,
        "mimic_extended_lower_body_pos_exp": 2.0,
        "mimic_extended_body_ori_exp":  1.0,
        "mimic_body_lin_vel_exp":       1.0,
        "mimic_body_ang_vel_exp":       1.0,
        # "feet_pos_exp":                40.0,
        # -- regularisation
        # "action_rate":                 -0.3,
        "action_rate":                 -0.4,
        "joint_pos_limits":           -10.0,
        "undesired_contacts":          -0.1,
    })

    root_orientation_sigma: float = 0.4 ** 2
    # body_pos_sigma: float = 0.3 ** 2
    body_pos_sigma: float = 0.01 ** 2
    upper_body_pos_sigma: float = 0.3 ** 2
    lower_body_pos_sigma: float = 0.3 ** 2
    body_ori_sigma: float = 0.4 ** 2
    body_lin_vel_sigma: float = 1.0 ** 2
    body_ang_vel_sigma: float = 3.14 ** 2
    feet_pos_sigma: float = 0.045 ** 2
    feet_body_names: List[str] = field(
        default_factory=lambda: ["left_ankle_roll_link", "right_ankle_roll_link"])
    upper_body_names: List[str] = field(default_factory=lambda: [
        "waist_yaw_link", "waist_roll_link", "torso_link",
        "left_shoulder_pitch_link", "left_shoulder_roll_link",
        "left_shoulder_yaw_link", "left_elbow_link",
        "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
        "right_shoulder_pitch_link", "right_shoulder_roll_link",
        "right_shoulder_yaw_link", "right_elbow_link",
        "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
        "left_hand_link_ext", "right_hand_link_ext", "head_link_ext",
    ])
    lower_body_names: List[str] = field(default_factory=lambda: [
        "pelvis",
        "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
        "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
        "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
        "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    ])

    scale_by_dt: bool = True


SIGMA_FIELDS: Dict[str, str] = {
    "root_orientation":            "root_orientation_sigma",
    "mimic_extended_body_pos_exp": "body_pos_sigma",
    "mimic_extended_body_ori_exp": "body_ori_sigma",
    "mimic_body_lin_vel_exp":      "body_lin_vel_sigma",
    "mimic_body_ang_vel_exp":      "body_ang_vel_sigma",
    "feet_pos_exp":                "feet_pos_sigma",
    "mimic_extended_upper_body_pos_exp": "upper_body_pos_sigma",
    "mimic_extended_lower_body_pos_exp": "lower_body_pos_sigma",
}


@dataclass
class UpdateSigmaCfg:
    enabled: bool = True
    # enabled: bool = False
    per_motion: bool = True     # load per-motion sigmas from ckpt
    alpha: float = 1e-3         # EMA coef
    min_sigma: float = 1e-4
    t_warmup: int = 200
    terms: Dict[str, Optional[float]] = field(default_factory=lambda: {
        # "mimic_extended_body_pos_exp": 0.3**2,
        "mimic_extended_upper_body_pos_exp": 0.3**2,
        "mimic_extended_lower_body_pos_exp": 0.3**2,
    })
    teacher_ckpt: str = ""


@dataclass
class OdomDelayCfg:
    enabled: bool = True
    update_period_steps: List[int] = field(default_factory=lambda: [4, 6])  # 12.5 ~ 8.3 Hz (10 Hz nominal)
    latency_steps: List[int] = field(default_factory=lambda: [0, 3])        # 0 ~ 60 ms

    # drift
    drift_enabled: bool = True
    drift_scale: float = 1.5             # DR
    drift_xy_dist: float = 0.01          # [m/sqrt(m)]
    drift_xy_time: float = 0.001         # [m/sqrt(s)]
    drift_z_dist: float = 0.005          # [m/sqrt(m)]
    drift_yaw_dist: float = 0.05         # [deg/sqrt(m)]
    drift_yaw_rot: float = 0.03          # [deg/sqrt(deg)]
    drift_yaw_time: float = 0.01         # [deg/sqrt(s)]
    drift_scale_bias_std: float = 0.5    # [%] per episode
    drift_yaw_bias_std: float = 0.01     # [deg/m] per episode

    max_xy_dev: float = 0.2              # [m]   error bound (est - truth)
    max_z_dev: float = 0.05              # [m]
    max_yaw_dev: float = 5.0             # [deg]

    init_xy_dev: float = 0.05            # [m]
    init_yaw_dev: float = 1.0            # [deg]


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
    source: str = "teacher"             # scratch / teacher
    teacher_ckpt: str = ""
    adapt: str = "lora"                 # teacher critic: lora / fft / frozen
    lora_rank: int = 1
    lora_alpha: float = 1.0
    lora_a_init_std: Optional[float] = None
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])   # scratch only
    activation: str = "elu"
    lr: float = 1e-3
    warmup_iters: int = 20             # critic-only iterations before the actor is unfrozen


@dataclass
class PPOCfg:
    num_steps_per_env: int = 24
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    use_clipped_value_loss: bool = True
    max_grad_norm: float = 1.0
    schedule: str = "adaptive"          # adaptive / fixed
    desired_kl: float = 0.01
    adaptive_groups: List[str] = field(default_factory=lambda: ["actor", "std"])
    normalize_advantage_per_mini_batch: bool = False


@dataclass
class TrainCfg:
    max_iterations: int = 3000
    save_interval: int = 500
    log_interval: int = 1
    device: str = "cuda:0"
    init_noise_std: Optional[float] = 0.2
    std_trainable: bool = False
    std_lr: Optional[float] = 1e-3
    resume: str = ""


@dataclass
class FinetuneCfg:
    paths: PathsCfg = field(default_factory=PathsCfg)
    motion: MotionCfg = field(default_factory=MotionCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    update_sigma: UpdateSigmaCfg = field(default_factory=UpdateSigmaCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    odom_delay: OdomDelayCfg = field(default_factory=OdomDelayCfg)
    lora: LoRACfg = field(default_factory=LoRACfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    ppo: PPOCfg = field(default_factory=PPOCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    def _expand_paths(self):
        """Expand ``~`` in every path field. bash does not do it after ``=``, so an override like
        ``--set critic.teacher_ckpt=~/x.pt`` arrives with a literal tilde."""
        for name in ("student_pt", "actor_spec", "critic_spec", "motion_file", "xml", "log_root",
                     "export_dir", "export_pt_dir", "rsl_rl_root"):
            v = getattr(self.paths, name)
            if isinstance(v, str) and v:
                setattr(self.paths, name, os.path.expanduser(v))
        if self.critic.teacher_ckpt:
            self.critic.teacher_ckpt = os.path.expanduser(self.critic.teacher_ckpt)
        if self.update_sigma.teacher_ckpt:
            self.update_sigma.teacher_ckpt = os.path.expanduser(self.update_sigma.teacher_ckpt)
        if self.train.resume:
            self.train.resume = os.path.expanduser(self.train.resume)

    def resolve(self) -> "FinetuneCfg":
        self._expand_paths()
        if self.critic.source not in ("scratch", "teacher"):
            raise ValueError(f"critic.source must be scratch or teacher, got {self.critic.source!r}")
        if self.critic.source == "teacher" and not self.critic.teacher_ckpt:
            raise ValueError("critic.source=teacher needs critic.teacher_ckpt (DEX_RL_LAB model_N.pt)")
        if self.critic.adapt not in ("lora", "fft", "frozen"):
            raise ValueError(f"critic.adapt must be lora, fft or frozen, got {self.critic.adapt!r}")
        if self.env.backend not in ("cpu", "gpu"):
            raise ValueError(f"env.backend must be cpu or gpu, got {self.env.backend!r}")
        if self.env.backend == "gpu" and not str(self.train.device).startswith("cuda"):
            raise ValueError(
                f"env.backend=gpu needs a CUDA train.device, got {self.train.device!r}. "
                "Add train.device=cuda:0 (select the physical GPU with CUDA_VISIBLE_DEVICES)."
            )
        if self.ppo.schedule not in ("adaptive", "fixed"):
            raise ValueError(f"ppo.schedule must be adaptive or fixed, got {self.ppo.schedule!r}")
        self._resolve_update_sigma()
        self._resolve_odom_delay()
        self.paths.resolve(self.critic.source)
        return self

    def _resolve_odom_delay(self):
        o = self.odom_delay
        if not o.enabled:
            return
        # gpu backend only: the cpu env (env/mujoco_env.py) ignores it (train.py warns)
        for name in ("update_period_steps", "latency_steps"):
            v = getattr(o, name)
            if len(v) != 2:
                raise ValueError(f"odom_delay.{name} must be [lo, hi], got {v!r}")
        (p_lo, p_hi), (l_lo, l_hi) = o.update_period_steps, o.latency_steps
        if not (1 <= p_lo <= p_hi) or not (0 <= l_lo <= l_hi):
            raise ValueError(f"invalid odom_delay ranges: update_period_steps={o.update_period_steps}, "
                             f"latency_steps={o.latency_steps}")
        if o.drift_enabled:
            for name in ("drift_scale", "drift_xy_dist", "drift_xy_time", "drift_z_dist", "drift_yaw_dist",
                         "drift_yaw_rot", "drift_yaw_time", "drift_scale_bias_std", "drift_yaw_bias_std",
                         "init_xy_dev", "init_yaw_dev"):
                if float(getattr(o, name)) < 0:
                    raise ValueError(f"odom_delay.{name} must be >= 0, got {getattr(o, name)!r}")

    def _resolve_update_sigma(self):
        """Validate ``update_sigma`` and write each sigma_init into ``reward``."""
        u = self.update_sigma
        if not u.enabled:
            return
        if self.env.backend != "gpu":
            raise ValueError(
                "update_sigma.enabled=True needs env.backend=gpu; the cpu env does not implement "
                "the adaptive tracking factor. Add env.backend=gpu or update_sigma.enabled=false."
            )
        if not 0.0 < u.alpha <= 1.0:
            raise ValueError(f"update_sigma.alpha must be in (0, 1], got {u.alpha}.")
        if u.min_sigma <= 0.0:
            raise ValueError(f"update_sigma.min_sigma must be positive, got {u.min_sigma}.")
        if not u.terms:
            raise ValueError("update_sigma.enabled=True but update_sigma.terms is empty.")
        if u.per_motion and u.t_warmup < 1:
            raise ValueError(f"update_sigma.t_warmup must be >= 1, got {u.t_warmup}.")
        if u.teacher_ckpt:
            if not u.per_motion:
                raise ValueError("update_sigma.teacher_ckpt holds per-motion factors; "
                                 "it needs update_sigma.per_motion=true.")
            if not os.path.isfile(u.teacher_ckpt):
                raise ValueError(f"update_sigma.teacher_ckpt not found: {u.teacher_ckpt}")
            import torch
            state = torch.load(u.teacher_ckpt, map_location="cpu", weights_only=False).get("adaptive_sigma")
            if not state or not state.get("per_motion", False):
                raise ValueError(
                    f"{u.teacher_ckpt} has no per-motion 'adaptive_sigma' entry. Train the teacher "
                    "with SigmaUpdateCfg.enabled=True and per_motion=True.")
            u.terms = {name: None for name in state["sigma"]}
        for term_name, sigma_init in u.terms.items():
            if term_name not in SIGMA_FIELDS:
                raise ValueError(
                    f"update_sigma.terms lists '{term_name}', which is not an exp(-x / sigma) "
                    f"tracking term. Adaptable terms: {list(SIGMA_FIELDS)}."
                )
            # A term missing from reward.terms is never computed, so its factor would never move.
            if not self.reward.terms.get(term_name):
                raise ValueError(
                    f"update_sigma.terms lists '{term_name}', which is not an active reward term "
                    "(absent from reward.terms or weight 0)."
                )
            if sigma_init is not None:
                setattr(self.reward, SIGMA_FIELDS[term_name], float(sigma_init))

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
            node = node[p] if isinstance(node, dict) else getattr(node, p)
        name = parts[-1]
        if isinstance(node, dict):
            if name not in node:
                raise KeyError(f"unknown key {name!r} in {'.'.join(parts[:-1])} "
                               f"(have: {', '.join(node)})")
            node[name] = _cast(value, node[name])
            continue
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
