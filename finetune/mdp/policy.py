"""Actor-critic for LoRA fine-tuning of a distilled student."""
from __future__ import annotations

import re
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from finetune.tasks.ft_cfg import CriticCfg, LoRACfg
from finetune.tasks.lora import LoRALinear, count_parameters, inject_lora, lora_parameters, merge_lora


_ACT = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU, "gelu": nn.GELU, "silu": nn.SiLU}


def build_mlp(in_dim: int, hidden: List[int], out_dim: int, activation: str = "elu") -> nn.Sequential:
    act = _ACT[activation.lower()]
    layers: List[nn.Module] = [nn.Linear(in_dim, hidden[0]), act()]
    for i in range(len(hidden)):
        if i == len(hidden) - 1:
            layers.append(nn.Linear(hidden[i], out_dim))
        else:
            layers.append(nn.Linear(hidden[i], hidden[i + 1]))
            layers.append(act())
    return nn.Sequential(*layers)


def load_student_state(path: str, map_location="cpu") -> Dict[str, torch.Tensor]:
    """Return {student.*, std|log_std} tensors from an rsl_rl checkpoint or a bare state dict."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    out = {k: v for k, v in sd.items() if k.startswith("student.") or k in ("std", "log_std")}
    if not any(k.startswith("student.") for k in out):
        raise KeyError(f"{path}: no 'student.*' keys (found e.g. {list(sd)[:5]}) -- not a distillation checkpoint?")
    return out


def mlp_from_state(state: Dict[str, torch.Tensor], prefix: str, activation: str = "elu") -> nn.Sequential:
    """Rebuild an nn.Sequential MLP ([Linear, act]*n + Linear) from the ``<prefix>.<i>.weight``
    shapes of a state dict and load its weights."""
    pat = re.compile(re.escape(prefix) + r"\.(\d+)\.weight$")
    idx = sorted({int(m.group(1)) for k in state if (m := pat.match(k))})
    if not idx:
        raise KeyError(f"no '{prefix}.<i>.weight' keys in state dict (found e.g. {list(state)[:5]})")
    dims = [state[f"{prefix}.{i}.weight"].shape for i in idx]
    hidden = [d[0] for d in dims[:-1]]
    seq = build_mlp(dims[0][1], hidden, dims[-1][0], activation)
    seq.load_state_dict({k[len(prefix) + 1:]: v for k, v in state.items() if k.startswith(prefix + ".")})
    return seq


def student_from_state(state: Dict[str, torch.Tensor], activation: str = "elu") -> nn.Sequential:
    """Rebuild the student nn.Sequential ([Linear, act]*n + Linear) from the state-dict shapes."""
    return mlp_from_state(state, "student", activation)


def load_teacher_critic_state(path: str, map_location="cpu") -> Dict[str, torch.Tensor]:
    """``critic.*`` tensors of a DEX_RL_LAB PPO (teacher) checkpoint."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    out = {k: v for k, v in sd.items() if k.startswith("critic.")}
    if not out:
        raise KeyError(f"{path}: no 'critic.*' keys (found e.g. {list(sd)[:5]}) -- not a PPO teacher checkpoint?")
    return out


class LoRAActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, student_state: Dict[str, torch.Tensor], num_critic_obs: int, num_actions: int,
                 lora_cfg: LoRACfg, critic_cfg: CriticCfg, init_noise_std: Optional[float] = 0.2,
                 std_trainable: bool = True, activation: str = "elu",
                 critic_state: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self.mode = lora_cfg.mode
        self.num_actions = num_actions
        self.critic_source = critic_cfg.source
        # scratch critics are fully trained; teacher critics follow critic_cfg.adapt
        self.critic_mode = "fft" if critic_cfg.source == "scratch" else critic_cfg.adapt

        # ---- actor: pre-trained student ----
        self.actor = student_from_state(student_state, activation)
        self.num_actor_obs = self.actor[0].in_features
        for p in self.actor.parameters():
            p.requires_grad_(self.mode == "fft")
        if self.mode == "lora":
            inject_lora(self.actor, lora_cfg.rank, lora_cfg.alpha, lora_cfg.a_init_std)
        elif self.mode not in ("fft", "frozen"):
            raise ValueError(f"unknown lora mode {self.mode}")

        # ---- exploration std (scalar per action, like the student checkpoint) ----
        if init_noise_std is None:
            if "std" in student_state:
                std0 = student_state["std"].clone().float()
            elif "log_std" in student_state:
                std0 = torch.exp(student_state["log_std"].clone().float())
            else:
                raise KeyError("checkpoint has no std; set train.init_noise_std")
        else:
            std0 = torch.full((num_actions,), float(init_noise_std))
        self.std = nn.Parameter(std0, requires_grad=std_trainable)
        self._std_trainable = std_trainable

        # ---- critic: from scratch, or the teacher's (frozen W0 + LoRA / fft / frozen) ----
        if critic_cfg.source == "scratch":
            self.critic = build_mlp(num_critic_obs, list(critic_cfg.hidden_dims), 1, critic_cfg.activation)
        else:
            if critic_state is None:
                raise ValueError("critic.source=teacher needs critic_state (load_teacher_critic_state)")
            self.critic = mlp_from_state(critic_state, "critic", critic_cfg.activation)
            if self.critic[0].in_features != num_critic_obs:
                raise ValueError(
                    f"teacher critic expects {self.critic[0].in_features} obs but the critic spec builds "
                    f"{num_critic_obs} -- the spec must reproduce the teacher's critic observation group "
                    f"(finetune/tasks/export_critic_spec.py)")
            for p in self.critic.parameters():
                p.requires_grad_(self.critic_mode == "fft")
            if self.critic_mode == "lora":
                inject_lora(self.critic, critic_cfg.lora_rank, critic_cfg.lora_alpha, critic_cfg.lora_a_init_std)
        self.distribution: Optional[Normal] = None
        Normal.set_default_validate_args(False)

    # ----- parameter groups -----
    def actor_trainable_parameters(self) -> Iterator[nn.Parameter]:
        if self.mode == "lora":
            yield from lora_parameters(self.actor)
        elif self.mode == "fft":
            yield from self.actor.parameters()

    def critic_parameters(self) -> Iterator[nn.Parameter]:
        """Trainable critic parameters (all / LoRA only / none)."""
        if self.critic_mode == "lora":
            yield from lora_parameters(self.critic)
        elif self.critic_mode == "fft":
            yield from self.critic.parameters()

    def set_actor_trainable(self, flag: bool):
        for p in self.actor_trainable_parameters():
            p.requires_grad_(flag)
        self.std.requires_grad_(flag and self._std_trainable)

    def summary(self) -> str:
        n_actor = count_parameters(self.actor.parameters())
        n_train = count_parameters(self.actor_trainable_parameters())
        n_critic = count_parameters(self.critic.parameters())
        n_critic_train = count_parameters(self.critic_parameters())
        return (f"actor: {n_actor} params, trainable {n_train} ({100.0 * n_train / max(n_actor, 1):.3f}%, mode={self.mode})"
                f" | critic ({self.critic_source}, {self.critic_mode}): {n_critic} params, trainable {n_critic_train}"
                f" | std init mean {self.std.mean().item():.3f}")

    # ----- rsl_rl policy API -----
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, self.std.expand_as(mean))

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    # ----- export -----
    def merged_actor(self) -> nn.Sequential:
        """Plain nn.Sequential with W0 + (alpha/r) B A folded in (deploy / ONNX)."""
        return merge_lora(self.actor) if self.mode == "lora" else __import__("copy").deepcopy(self.actor)

    def merged_critic(self) -> nn.Sequential:
        return merge_lora(self.critic) if self.critic_mode == "lora" else __import__("copy").deepcopy(self.critic)

    def lora_state(self) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in self.state_dict().items() if "lora_" in k}
