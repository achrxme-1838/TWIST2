from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from finetune import _paths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402


# ----------------------------------------------------------------------------- common
def load_finetune_ckpt(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "merged_actor_state_dict" not in ck:
        raise SystemExit(f"{path}: not a finetune checkpoint (no merged_actor_state_dict)")
    return ck


def merged_actor_from_ckpt(ck) -> torch.nn.Sequential:
    from finetune.mdp.policy import build_mlp
    sd = ck["merged_actor_state_dict"]
    idx = sorted({int(k.split(".")[0]) for k in sd if k.endswith(".weight")})
    dims = [sd[f"{i}.weight"].shape for i in idx]
    actor = build_mlp(dims[0][1], [d[0] for d in dims[:-1]], dims[-1][0], "elu")
    actor.load_state_dict(sd)
    return actor.eval()


# ----------------------------------------------------------------------------- onnx
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


# ----------------------------------------------------------------------------- rsl_rl .pt
def build_student_pt(ck, student_pt: str) -> dict:
    src = torch.load(student_pt, map_location="cpu", weights_only=False)
    sd = dict(src["model_state_dict"])
    old = {k: v for k, v in sd.items() if k.startswith("student.")}
    new = {f"student.{k}": v.detach().cpu().clone() for k, v in ck["merged_actor_state_dict"].items()}
    if set(old) != set(new):
        raise SystemExit(f"student layout mismatch: source {sorted(old)[:4]}... vs fine-tuned {sorted(new)[:4]}...")
    for k, v in new.items():
        if old[k].shape != v.shape:
            raise SystemExit(f"{k}: shape {tuple(old[k].shape)} in source vs {tuple(v.shape)} fine-tuned")
    sd.update(new)
    if "std" in sd and "std" in ck:
        sd["std"] = ck["std"].detach().cpu().clone().to(sd["std"].dtype)
    out = dict(src)
    out["model_state_dict"] = sd
    out["finetune"] = {
        "source_student_pt": os.path.abspath(student_pt),
        "finetune_ckpt": None,          # filled by caller
        "finetune_iter": int(ck.get("iter", 0)) + 1,
        "cfg": ck["cfg"],
    }
    return out


def verify_student_pt(pt_path: str, actor: torch.nn.Sequential, n: int = 8) -> float:
    from finetune.mdp.policy import load_student_state, student_from_state
    student = student_from_state(load_student_state(pt_path)).eval()
    x = torch.randn(n, actor[0].in_features)
    with torch.no_grad():
        return float((student(x) - actor(x)).abs().max())


def resolve_run_dir(run: str | None, actor_spec: str, rsl_rl_root: str | None) -> str:
    """``--into-run`` target: an explicit directory, or a run name (default: the actor spec's
    ``source_run``) searched under <DEX>/scripts/logs/rsl_rl/*/."""
    if run and os.path.isdir(run):
        return run
    if not run:
        with open(actor_spec) as f:
            run = yaml.safe_load(f).get("source_run")
        if not run:
            raise SystemExit(f"--into-run needs a run name: {actor_spec} has no source_run")
    logs_root = os.path.join(_paths.resolve_rsl_rl_root(rsl_rl_root), "scripts", "logs", "rsl_rl")
    hits = glob.glob(os.path.join(logs_root, "*", run))
    if len(hits) != 1:
        raise SystemExit(f"run '{run}' -> {len(hits)} matches under {logs_root}: {hits}")
    return hits[0]


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="finetune checkpoint (logs/finetune/<run>/model_N.pt)")
    p.add_argument("--out-name", default=None, help="default: <student name>_ft<iter>")
    p.add_argument("--export-dir", default=None, help="ONNX + spec dir (default: cfg paths.export_dir)")
    p.add_argument("--pt-dir", default=None, help=".pt dir (default: cfg paths.export_pt_dir)")
    p.add_argument("--no-onnx", action="store_true")
    p.add_argument("--no-pt", action="store_true")
    p.add_argument("--into-run", nargs="?", const="", default=None, metavar="RUN",
                   help="also copy the .pt into a DEX run dir as model_ft<N>.pt for play.py "
                        "(RUN = run dir or name; default: the actor spec's source_run)")
    args = p.parse_args()

    ck = load_finetune_ckpt(args.ckpt)
    cfg = ck["cfg"]
    _paths.setup(cfg["paths"].get("rsl_rl_root"))
    actor = merged_actor_from_ckpt(ck)
    ft_iter = int(ck.get("iter", 0)) + 1
    name = args.out_name or f"{cfg['paths']['name']}_ft{ft_iter}"

    if not args.no_onnx:
        export_dir = args.export_dir or cfg["paths"]["export_dir"]
        os.makedirs(export_dir, exist_ok=True)
        onnx_path = os.path.join(export_dir, f"{name}.onnx")
        spec_path = os.path.join(export_dir, f"{name}.yaml")
        export_onnx(actor, onnx_path)
        shutil.copyfile(cfg["paths"]["actor_spec"], spec_path)
        err = verify_onnx(actor, onnx_path)
        print(f"[export] onnx : {onnx_path}\n[export] spec : {spec_path}\n[export]        max |torch - onnx| = {err:.2e}")
        print(f"[export]        run: bash sim2sim_batch.sh  (CKPT={onnx_path})")

    if not args.no_pt:
        pt_dir = args.pt_dir or cfg["paths"].get("export_pt_dir") or os.path.join(_paths.TWIST2_ROOT, "assets", "pt_finetuned")
        os.makedirs(pt_dir, exist_ok=True)
        pt_path = os.path.join(pt_dir, f"{name}.pt")
        out = build_student_pt(ck, cfg["paths"]["student_pt"])
        out["finetune"]["finetune_ckpt"] = os.path.abspath(args.ckpt)
        torch.save(out, pt_path)
        err = verify_student_pt(pt_path, actor)
        print(f"[export] pt   : {pt_path}\n[export]        max |student(pt) - merged actor| = {err:.2e}")
        if args.into_run is not None:
            run_dir = resolve_run_dir(args.into_run or None, cfg["paths"]["actor_spec"], cfg["paths"].get("rsl_rl_root"))
            dst = os.path.join(run_dir, f"model_ft{ft_iter}.pt")
            shutil.copyfile(pt_path, dst)
            print(f"[export] copy : {dst}\n[export]        run: python play.py --task <distill task> "
                  f"--load_run {os.path.basename(run_dir)} --checkpoint ft{ft_iter}")


if __name__ == "__main__":
    main()
