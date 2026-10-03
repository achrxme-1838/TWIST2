"""Parity check: BatchedG1SimCore (mujoco_warp, GPU) vs G1SimCore (MuJoCo, CPU).

Resets both cores to the same state, drives them with the same actions and compares every
field of ``state()``. The GPU core is float32 and the CPU core float64, so differences grow
with the rollout; the point is that they start at float32 epsilon and stay small.

    python check_batched_core.py [--envs 4] [--steps 50] [--xml ...]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from finetune import _paths  # noqa: E402

_paths.setup()

import mujoco  # noqa: E402
import mujoco_warp as mjwarp  # noqa: E402
import warp as wp  # noqa: E402

from cfg import g1_29dof_cfg as cfg  # noqa: E402
from finetune._paths import TWIST2_ROOT  # noqa: E402
from finetune.env.batched_core import BatchedG1SimCore  # noqa: E402
from finetune.env.sim_core import G1SimCore  # noqa: E402

FIELDS = ["root_pos", "root_quat", "root_lin_vel_w", "root_ang_vel_b",
          "dof_pos_sdk", "dof_vel_sdk", "dof_pos_isaac", "dof_vel_isaac", "torque_sdk",
          "body_pos", "body_quat", "body_lin_vel", "body_ang_vel",
          "body_lin_vel_ext", "body_ang_vel_ext", "contact_force"]


def compare(cpu_state, gpu_state, env: int, tag: str):
    print(f"\n--- {tag} ---")
    worst = 0.0
    for name in FIELDS:
        a = np.asarray(getattr(cpu_state, name), dtype=np.float64)
        b = getattr(gpu_state, name)[env].detach().cpu().numpy().astype(np.float64)
        if a.shape != b.shape:
            print(f"  {name:20s} SHAPE MISMATCH cpu{a.shape} gpu{b.shape}")
            worst = float("inf")
            continue
        d = np.abs(a - b).max() if a.size else 0.0
        scale = max(np.abs(a).max(), 1e-9) if a.size else 1.0
        worst = max(worst, d / scale)
        print(f"  {name:20s} max|diff| = {d:.3e}   rel = {d / scale:.3e}   |cpu|max = {scale:.4f}")
    return worst


def check_contact_force_layout(xml: str):
    """mj_contactForce packs [force(3), torque(3)]; confirm mujoco_warp keeps that order."""
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    n = cfg.NUM_ACTIONS
    d.qpos[2] = 0.757
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:7 + n] = cfg.DEFAULT_DOF_POS
    for _ in range(200):
        mujoco.mj_step(m, d)
    mw = mjwarp.put_model(m)
    dw = mjwarp.put_data(m, d, nworld=1, njmax=256, naconmax=256)
    mjwarp.step(mw, dw)
    mujoco.mj_step(m, d)
    ids = wp.array(np.arange(dw.contact.worldid.shape[0], dtype=np.int32), dtype=wp.int32)
    force = wp.zeros(dw.contact.worldid.shape[0], dtype=wp.spatial_vectorf)
    mjwarp.contact_force(mw, dw, ids, True, force)
    wp.synchronize()
    fw = wp.to_torch(force).cpu().numpy()
    nacon = int(wp.to_torch(dw.nacon).cpu()[0])
    # CPU reference: world-frame contact forces, sorted by magnitude (contact order differs)
    cpu = []
    f6 = np.zeros(6)
    for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, f6)
        cpu.append(d.contact[i].frame.reshape(3, 3).T @ f6[:3])
    cpu = np.sort(np.linalg.norm(cpu, axis=-1)) if cpu else np.zeros(0)
    first3 = np.sort(np.linalg.norm(fw[:nacon, 0:3], axis=-1))
    last3 = np.sort(np.linalg.norm(fw[:nacon, 3:6], axis=-1))
    print(f"\n--- contact force layout (cpu ncon={d.ncon}, gpu nacon={nacon}) ---")
    print(f"  cpu |f| sorted      : {np.round(cpu, 3)}")
    print(f"  gpu |wrench[0:3]|   : {np.round(first3, 3)}")
    print(f"  gpu |wrench[3:6]|   : {np.round(last3, 3)}")
    if cpu.size and first3.size == cpu.size:
        print(f"  => [0:3] is the force: {np.abs(first3 - cpu).max() < 1e-2 * max(cpu.max(), 1.0)}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--xml", default=os.path.join(TWIST2_ROOT, "assets", "g1", "g1_sim2sim_29dof.xml"))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    J = cfg.NUM_ACTIONS

    cpu_core = G1SimCore(args.xml, future_steps=10, future_fk_steps=1)
    gpu_core = BatchedG1SimCore(args.xml, num_envs=args.envs, device=args.device)
    print(f"[core] decimation={gpu_core.decimation} contact_substeps={sorted(gpu_core.contact_substeps)} "
          f"(cpu: {cpu_core.decimation}, {sorted(cpu_core.contact_substeps)})")
    assert gpu_core.decimation == cpu_core.decimation
    assert gpu_core.contact_substeps == cpu_core.contact_substeps

    # identical initial state: standing on the default pose, slight tilt + motion
    root_pos = np.array([0.0, 0.0, 0.80])
    root_quat = np.array([0.9962, 0.0, 0.0872, 0.0])
    root_quat = root_quat / np.linalg.norm(root_quat)
    lin, ang = np.array([0.1, -0.05, 0.0]), np.array([0.0, 0.1, -0.05])
    dof_pos = cfg.DEFAULT_DOF_POS + rng.normal(scale=0.05, size=J)
    dof_vel = rng.normal(scale=0.1, size=J)

    cpu_core.reset_to(root_pos, root_quat, lin, ang, dof_pos, dof_vel)
    dev = gpu_core.device
    t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=dev).expand(args.envs, -1).contiguous()
    gpu_core.reset_to(torch.arange(args.envs, device=dev), t(root_pos), t(root_quat), t(lin), t(ang),
                      t(dof_pos), t(dof_vel))
    wp.synchronize()
    compare(cpu_core.state(), gpu_core.state(), 0, "after reset")

    actions = rng.normal(scale=0.3, size=(args.steps, J)).astype(np.float32)
    for k in range(args.steps):
        cpu_core.step(actions[k])
        gpu_core.step(torch.as_tensor(actions[k], device=dev).expand(args.envs, -1).contiguous())
        if k == 0:
            wp.synchronize()
            compare(cpu_core.state(), gpu_core.state(), 0, "after 1 control step")
    wp.synchronize()
    worst = compare(cpu_core.state(), gpu_core.state(), 0, f"after {args.steps} control steps")

    gs = gpu_core.state()
    spread = max(float((getattr(gs, f) - getattr(gs, f)[0:1]).abs().max()) for f in FIELDS)
    print(f"\nspread across the {args.envs} identical envs: {spread:.3e}")
    print(f"worst relative difference vs CPU: {worst:.3e}")

    check_contact_force_layout(args.xml)


if __name__ == "__main__":
    main()
