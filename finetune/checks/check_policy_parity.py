from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from finetune.checks._common import add_cfg_args, cfg_from_args  # noqa: E402


def main():
    p = add_cfg_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--onnx", default=None, help="default: assets/ckpts/<name>.onnx")
    args = p.parse_args()
    cfg = cfg_from_args(args, need_motion=False)
    from finetune._paths import TWIST2_ROOT
    from finetune.mdp.policy import LoRAActorCritic, load_student_state, student_from_state

    state = load_student_state(cfg.paths.student_pt)
    actor = student_from_state(state).eval()
    print(f"student: {actor}")
    print(f"std in checkpoint: {state['std'].mean().item() if 'std' in state else 'n/a'}")
    x = torch.randn(16, actor[0].in_features)
    with torch.no_grad():
        y_ref = actor(x)

    # LoRA policy at init == student
    pol = LoRAActorCritic(state, 8, actor[-1].out_features, cfg.lora, cfg.critic,
                          init_noise_std=cfg.train.init_noise_std).eval()
    with torch.no_grad():
        y_lora = pol.act_inference(x)
        y_merged = pol.merged_actor()(x)
    print(f"|lora(B=0) - student| max = {(y_lora - y_ref).abs().max().item():.2e}")
    print(f"|merged   - student| max = {(y_merged - y_ref).abs().max().item():.2e}")
    print(pol.summary())

    onnx_path = args.onnx or os.path.join(TWIST2_ROOT, "assets", "ckpts", f"{cfg.paths.name}.onnx")
    if os.path.isfile(onnx_path):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            y_onnx = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
            print(f"|onnx     - student| max = {np.abs(y_onnx - y_ref.numpy()).max():.2e}  ({onnx_path})")
        except ImportError:
            print("onnxruntime not installed; ONNX comparison skipped")
    else:
        print(f"no ONNX at {onnx_path}; skipped")


if __name__ == "__main__":
    main()
