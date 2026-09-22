from __future__ import annotations

import argparse
import glob
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from finetune import _paths  # noqa: E402


def _dex_scripts():
    root = _paths.resolve_rsl_rl_root()
    sys.path.insert(0, os.path.join(root, "scripts"))
    import export_deploy_cfg as edc  # noqa: E402
    return root, edc


def _find_run(run: str, logs_root: str) -> str:
    if os.path.isdir(run):
        return run
    hits = glob.glob(os.path.join(logs_root, "*", run))
    if len(hits) != 1:
        raise SystemExit(f"run '{run}' -> {len(hits)} matches under {logs_root}: {hits}")
    return hits[0]


def build_critic_spec(env: dict, edc) -> dict:
    grp = env["observations"]["critic"]
    cmd = env["commands"]["reference_motions"]
    T = int(cmd.get("num_future_steps") or 1)
    interval = int(cmd.get("step_interval") or 1)
    group_hist = int(grp.get("history_length") or 0)

    terms = []
    for attr, term in grp.items():
        if not isinstance(term, dict) or "func" not in term:
            continue
        name = term["func"].split(":")[-1].split(".")[-1]
        entry = {"name": name}
        hist = int(term.get("history_length") or group_hist or 0)
        if hist > 1:
            entry["history"] = hist
        if term.get("scale") is not None:
            entry["scale"] = float(term["scale"])
        if term.get("clip") is not None:
            entry["clip"] = [float(v) for v in term["clip"]]
        params = edc._simple_params(term.get("params"))
        n_future = edc._future_num_steps(name, term.get("params"), T)
        if n_future is not None:
            params["num_steps"] = n_future
        if params:
            entry["params"] = params
        terms.append(entry)
    print(f"{len(terms)} critic terms: {[t['name'] for t in terms]}")
    return {"future_steps": T, "future_interval": interval, "terms": terms}


def critic_in_features(ckpt: str) -> int:
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    return int(sd["critic.0.weight"].shape[1])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="teacher run dir, or its name under <DEX>/scripts/logs/rsl_rl/*/")
    p.add_argument("--ckpt", default=None, help="teacher model_N.pt: records obs_dim = critic input width")
    p.add_argument("--out", default=os.path.join(here, "..", "specs", "critic_teacher.yaml"))
    args = p.parse_args()

    dex_root, edc = _dex_scripts()
    run_dir = _find_run(args.run, os.path.join(dex_root, "scripts", "logs", "rsl_rl"))
    with open(os.path.join(run_dir, "params", "env.yaml")) as f:
        env = yaml.load(f, Loader=edc._EnvYamlLoader)
    spec = build_critic_spec(env, edc)
    spec["source_run"] = os.path.basename(run_dir.rstrip("/"))
    if args.ckpt:
        spec["obs_dim"] = critic_in_features(args.ckpt)
        print(f"obs_dim = {spec['obs_dim']} (critic.0.weight of {args.ckpt})")

    # fail early if a term has no deploy implementation
    _paths.setup()
    import finetune.mdp.critic_terms  # noqa: F401  (registers priv_* / teacher terms)
    from obs_terms import REGISTRY
    missing = [t["name"] for t in spec["terms"] if t["name"] not in REGISTRY]
    if missing:
        print(f"WARNING: no deploy implementation for {missing}")

    out = os.path.abspath(args.out)
    with open(out, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=None)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
