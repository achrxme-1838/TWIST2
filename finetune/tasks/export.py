from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from finetune import _paths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def export_onnx(actor: torch.nn.Sequential, path: str):
    actor = actor.cpu().eval()
    dummy = torch.zeros(1, actor[0].in_features)
    kwargs = dict(input_names=["obs"], output_names=["actions"],
                  dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(actor, dummy, path, dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(actor, dummy, path, **kwargs)


def verify_onnx(actor: torch.nn.Sequential, path: str, n: int = 8) -> float:
    try:
        import onnxruntime as ort
    except ImportError:
        print("[export] onnxruntime not installed; skipping ONNX verification")
        return float("nan")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.randn(n, actor[0].in_features).astype(np.float32)
    with torch.no_grad():
        y_t = actor(torch.from_numpy(x)).numpy()
    y_o = sess.run(None, {"obs": x})[0]
    return float(np.abs(y_t - y_o).max())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-name", default=None, help="default: <student name>_ft<iter>")
    p.add_argument("--export-dir", default=None)
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    _paths.setup(cfg["paths"].get("rsl_rl_root"))
    from finetune.mdp.policy import build_mlp

    sd = ck["merged_actor_state_dict"]
    idx = sorted({int(k.split(".")[0]) for k in sd if k.endswith(".weight")})
    dims = [sd[f"{i}.weight"].shape for i in idx]
    actor = build_mlp(dims[0][1], [d[0] for d in dims[:-1]], dims[-1][0], "elu")
    actor.load_state_dict(sd)

    export_dir = args.export_dir or cfg["paths"]["export_dir"]
    os.makedirs(export_dir, exist_ok=True)
    name = args.out_name or f"{cfg['paths']['name']}_ft{int(ck.get('iter', 0)) + 1}"
    onnx_path = os.path.join(export_dir, f"{name}.onnx")
    export_onnx(actor, onnx_path)
    shutil.copyfile(cfg["paths"]["actor_spec"], os.path.join(export_dir, f"{name}.yaml"))
    err = verify_onnx(actor, onnx_path)
    print(f"[export] {onnx_path}\n[export] {os.path.join(export_dir, name + '.yaml')}\n[export] max |torch - onnx| = {err:.2e}")
    print(f"[export] run: bash sim2sim_odom_docker.sh {onnx_path}")


if __name__ == "__main__":
    main()
