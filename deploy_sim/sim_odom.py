#!/usr/bin/env python
"""Sim2sim validation of the LIO odometry / global-origin pipeline in MuJoCo (LIO = FastLioEmulator,
launched with the L key)."""
import argparse
import json
import os
import time
from collections import deque

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio
import mujoco
import numpy as np

from cfg import g1_29dof_cfg as cfg
from odom_estimator import pose_inv, pose_mul, wrap, yaw_quat
from robot_control.config import Config
from robot_control.sim_g1_env import IMU_DEFAULTS, SimG1Env
from server_low_level_g1_real import (POLICY_STATES, STATES, RealTimePolicyController,
                                      add_controller_args, controller_kwargs)
from sim_lidar import DRIFT_DEFAULTS, FastLioEmulator
from utils.math import yaw_from_quat

LOG_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "sim2sim_odom"))


# --------------------------------------------------------------------------- GT monitor
class SimMonitor:
    """Tick hook: ground-truth comparison of the odometry estimate."""

    def __init__(self, env, lio, ctrl):
        self.env, self.lio, self.ctrl = env, lio, ctrl
        self._gt_hist = deque(maxlen=200)
        self._origin_seen = 0
        self.origin_errs = []
        self.last = {}

    def world_from_G(self):
        return pose_mul(self.lio.world_from_W, self.ctrl.est.T_W_G)

    def true_root_in_G(self):
        """GT pelvis in the G the controller set (perfect odometry, same origin)."""
        return pose_mul(pose_inv(self.world_from_G()), self.env.gt_pelvis())

    def _gt_at(self, t):
        ts = np.array([h[0] for h in self._gt_hist])
        return self._gt_hist[int(np.argmin(np.abs(ts - t)))][1]

    def __call__(self, ctrl):
        est, now = ctrl.est, self.env.now()
        gt = self.env.gt_pelvis()
        self._gt_hist.append((now, gt))
        x = {"gt_world_pos": gt[0], "gt_world_yaw": yaw_from_quat(gt[1])}
        if est.T_W_P is not None:
            gts = self._gt_at(est.stamp)
            Tw = pose_mul(self.lio.world_from_W, est.T_W_P)
            x["err_W_pos"] = Tw[0] - gts[0]
            x["err_W_yaw"] = wrap(yaw_from_quat(Tw[1]) - yaw_from_quat(gts[1]))
        if est.origin_count != self._origin_seen:
            self._origin_seen = est.origin_count
            X = ctrl.chain.point(gt, ctrl.origin_ref)
            ideal = (np.array([X[0][0], X[0][1], 0.0]), yaw_quat(yaw_from_quat(X[1])))
            d = pose_mul(pose_inv(ideal), self.world_from_G())
            e = np.r_[d[0], yaw_from_quat(d[1])]
            self.origin_errs.append(e)
            print(f"[monitor] origin #{est.origin_count} error vs GT: xy {np.linalg.norm(e[:2]) * 1e3:.1f} mm, "
                  f"z {e[2] * 1e3:+.1f} mm, yaw {np.rad2deg(e[3]):+.2f} deg")
        if est.has_origin:
            TwG = self.world_from_G()
            T_true = pose_mul(pose_inv(TwG), gt)
            x["gt_G_pos"], x["gt_G_yaw"] = T_true[0], yaw_from_quat(T_true[1])
            x["origin_err"] = self.origin_errs[-1]
            G_est = est.pelvis_in_G()
            if G_est is not None:
                gts = pose_mul(pose_inv(TwG), self._gt_at(est.stamp))
                x["err_G_pos"] = G_est[0] - gts[0]                         # at the odom stamp
                x["err_G_yaw"] = wrap(yaw_from_quat(G_est[1]) - yaw_from_quat(gts[1]))
                x["err_G_pos_now"] = G_est[0] - T_true[0]                  # incl. latency / hold
        ctrl.tick_extra.update(x)
        self.last = x


# --------------------------------------------------------------------------- recorder
def _add_geom(scn, gtype, size, pos, mat, rgba):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, dtype=np.float64), np.asarray(pos, dtype=np.float64),
                        np.asarray(mat, dtype=np.float64).reshape(9), np.asarray(rgba, dtype=np.float32))
    scn.ngeom += 1
    return g


def _add_arrow(scn, a, b, width, rgba):
    g = _add_geom(scn, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3), rgba)
    if g is not None:
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width,
                             np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def _add_axes(scn, T, length, width, alpha=1.0, gray=False):
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(T[1], dtype=np.float64))
    R = R.reshape(3, 3)
    for k in range(3):
        c = [0.6, 0.6, 0.6, alpha] if gray else [float(k == 0), float(k == 1), float(k == 2), alpha]
        _add_arrow(scn, T[0], T[0] + length * R[:, k], width, c)


class SimRecorder:
    """Tick hook: one frame per control tick -> mp4 (+ frames/markers, text overlay).

    Markers: W (LIO frame, thin axes), G (origin: axes + disc), GT pelvis (gray
    axes), estimated pelvis (RGB axes), reference bodies of the policy (green dots),
    estimated pelvis trail (yellow)."""

    def __init__(self, env, lio, monitor, path, size=(960, 540), every=1):
        self.env, self.lio, self.mon = env, lio, monitor
        w, h = size
        self.renderer = mujoco.Renderer(env.model, height=h, width=w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.2, 135.0, -20.0
        self.every = max(1, int(every))
        fps = 1.0 / (env.config.control_dt * self.every)
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1)
        self.path, self.n, self._k = path, 0, 0
        self.trail = deque(maxlen=400)
        self.scale = h / 540.0
        print(f"[video] {w}x{h} @ {fps:.0f} fps -> {path}")

    def __call__(self, ctrl):
        self._k += 1
        if (self._k - 1) % self.every:
            return
        gt = self.env.gt_pelvis()
        self.cam.lookat[:] = gt[0]
        r = self.renderer
        r.update_scene(self.env.data, camera=self.cam)
        scn = r.scene
        if self.lio.world_from_W is not None:
            _add_axes(scn, self.lio.world_from_W, 0.25, 0.008, 0.6)
        _add_axes(scn, gt, 0.3, 0.012, 1.0, gray=True)
        est = ctrl.est
        if est.has_origin:
            TwG = self.mon.world_from_G()
            _add_axes(scn, TwG, 0.4, 0.015)
            _add_geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.08, 0.002, 0], TwG[0], np.eye(3), [1, 1, 0, 0.6])
            G = est.pelvis_in_G()
            if G is not None:
                Tw = pose_mul(TwG, G)
                _add_axes(scn, Tw, 0.3, 0.02)
                if self._k % 5 == 0:
                    self.trail.append(Tw[0].copy())
        # Policy reference bodies (green), drawn in G.
        ref = ctrl.obs_pipe.ref_data
        if ctrl.state in POLICY_STATES and ref is not None and ctrl.last_root is not None:
            T = self.mon.world_from_G() if est.has_origin else None
            if T is not None:
                for bid in ctrl.tracked_body_ids:
                    p = pose_mul(T, (ref.xpos[bid], np.array([1.0, 0, 0, 0])))[0]
                    _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0, 0], p, np.eye(3), [0.1, 1, 0.1, 0.7])
        for p in self.trail:
            _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], p, np.eye(3), [1, 0.9, 0, 0.9])
        img = np.ascontiguousarray(r.render())
        self._overlay(img, ctrl)
        self.writer.append_data(img)
        self.n += 1

    def _overlay(self, img, ctrl):
        m = self.mon.last
        now = self.env.now()
        age = ctrl.est.age(now)
        lines = [f"t={now:6.2f}s  {ctrl.state}/{ctrl.phase}  gain {ctrl.safety.kp_scale * 100:.0f}%  "
                 f"origin #{ctrl.est.origin_count} ({ctrl.origin_ref})",
                 f"odom: {'-' if not np.isfinite(age) else f'{age * 1e3:.0f} ms'} old, {ctrl.est.n_msgs} msgs"]
        if "err_G_pos" in m:
            e, y = m["err_G_pos"], m["err_G_yaw"]
            lines.append(f"est-GT (G): xy {np.linalg.norm(e[:2]) * 100:.2f} cm  z {e[2] * 100:+.2f} cm  "
                         f"yaw {np.rad2deg(y):+.2f} deg")
        if "origin_err" in m:
            o = m["origin_err"]
            lines.append(f"origin-GT: xy {np.linalg.norm(o[:2]) * 100:.2f} cm  z {o[2] * 100:+.2f} cm  "
                         f"yaw {np.rad2deg(o[3]):+.2f} deg")
        if ctrl.status:
            lines.append(ctrl.status[:90])
        s = 0.55 * self.scale
        for i, t in enumerate(lines):
            y = int((24 + 22 * i) * self.scale)
            cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)

    def close(self):
        self.writer.close()
        self.renderer.close()
        print(f"[video] {self.n} frames saved to {self.path}")


# --------------------------------------------------------------------------- summary
def summarize(ctrl, monitor, lio, stop_reason, path):
    L = {k: np.asarray(v) for k, v in ctrl._log.items()}
    pol = np.isin(L["state"], [STATES.index(s) for s in POLICY_STATES])   # odometry in use
    out = dict(stop_reason=stop_reason, ticks=int(len(L["t"])),
               policy_ticks=int(pol.sum()),
               motion_ticks=int((L["state"] == STATES.index("MOTION")).sum()),
               odom_msgs=int(ctrl.est.n_msgs), sync_miss=int(ctrl.est.sync_miss),
               lio=dict(scans=lio.n_scans, dropped=lio.n_dropped, jumps=lio.n_jumps),
               origin_err=[dict(xy_mm=float(np.linalg.norm(e[:2]) * 1e3), z_mm=float(e[2] * 1e3),
                                yaw_deg=float(np.rad2deg(e[3]))) for e in monitor.origin_errs])

    def stats(key, sel):
        if key not in L:
            return None
        v = L[key][sel]
        v = v[np.all(np.isfinite(v.reshape(len(v), -1)), axis=1)] if len(v) else v
        if not len(v):
            return None
        if v.ndim == 1:
            return dict(rms_deg=float(np.rad2deg(np.sqrt(np.mean(v ** 2)))), max_deg=float(np.rad2deg(np.abs(v).max())))
        xy, z = np.linalg.norm(v[:, :2], axis=1), v[:, 2]
        return dict(xy_rms_mm=float(np.sqrt(np.mean(xy ** 2)) * 1e3), xy_max_mm=float(xy.max() * 1e3),
                    z_rms_mm=float(np.sqrt(np.mean(z ** 2)) * 1e3), z_max_mm=float(np.abs(z).max() * 1e3))

    allt = np.ones(len(L["t"]), dtype=bool)
    out["err_W_all"] = dict(pos=stats("err_W_pos", allt), yaw=stats("err_W_yaw", allt))
    out["err_G_policy"] = dict(pos=stats("err_G_pos", pol), yaw=stats("err_G_yaw", pol),
                               pos_now=stats("err_G_pos_now", pol))
    if "obs_err" in L:
        oe = L["obs_err"][pol]
        oe = oe[np.all(np.isfinite(oe), axis=1)]
        if len(oe):
            out["obs_err_max"] = {n: float(v) for n, v in zip(ctrl.obs_pipe.builder.slices().keys(), oe.max(axis=0))}
    if "gt_world_pos" in L:
        out["fell"] = bool(np.nanmin(L["gt_world_pos"][:, 2]) < 0.45)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[summary] {path}\n{json.dumps(out, indent=2)}")
    return out


def main():
    p = add_controller_args(argparse.ArgumentParser(description=__doc__,
                                                    formatter_class=argparse.RawDescriptionHelpFormatter))
    g = p.add_argument_group("sim")
    g.add_argument("--keys", type=str, default="", help='Scripted keys "s@0.5,l@3,a@4,b@11" (sim s); '
                                                        "given -> unattended, as fast as possible")
    g.add_argument("--max_time", type=float, default=None, help="Scripted runs: press C at this sim time")
    g.add_argument("--realtime", action="store_true", help="Pace scripted runs in real time")
    g.add_argument("--root_source", choices=["odom", "gt"], default="odom",
                   help="Policy root: odometry estimate, or GT in the same G (baseline)")
    g.add_argument("--no_shadow_gt", action="store_true", help="Skip the GT-root obs comparison")
    g.add_argument("--launch_waist", type=float, nargs=3, default=[15.0, 5.0, 5.0], metavar=("YAW", "ROLL", "PITCH"),
                   help="Waist angles [deg] of the launch pose (before START)")
    g.add_argument("--init_yaw", type=float, default=30.0, help="Robot heading in the world [deg]")
    g.add_argument("--settle", type=float, default=1.0, help="Sim seconds settled before the controller starts")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--push_force", type=float, default=150.0)
    g.add_argument("--hoist", choices=["key", "always", "never"], default="key",
                   help="Virtual hoist: attached until A (lowered at A, H toggles) (key); kept; or none")
    # LIO emulator (launched with the L key)
    g.add_argument("--lio_site", choices=["livox_imu", "mid360"], default="livox_imu",
                   help="Frame the emulator reports (match --odom_frame: mid360 <-> lidar)")
    g.add_argument("--lio_rate", type=float, default=10.0)
    g.add_argument("--lio_grav", type=int, default=1, help="1: gravity-aligned odom frame (yaw only)")
    g.add_argument("--lio_latency", type=float, default=0.03)
    g.add_argument("--lio_latency_std", type=float, default=0.008)
    g.add_argument("--lio_pos_std", type=float, default=0.003, help="[m]")
    g.add_argument("--lio_rot_std", type=float, default=0.1, help="[deg]")
    # LIO drift (random walk on scan-to-scan increments + per-launch bias); defaults ~ real FAST-LIO
    D = DRIFT_DEFAULTS
    g.add_argument("--lio_drift_xy_dist", type=float, default=D["xy_dist"], help="[m/sqrt(m)]")
    g.add_argument("--lio_drift_xy_time", type=float, default=D["xy_time"], help="[m/sqrt(s)]")
    g.add_argument("--lio_drift_z_dist", type=float, default=D["z_dist"], help="[m/sqrt(m)]")
    g.add_argument("--lio_drift_yaw_dist", type=float, default=D["yaw_dist"], help="[deg/sqrt(m)]")
    g.add_argument("--lio_drift_yaw_rot", type=float, default=D["yaw_rot"], help="[deg/sqrt(deg)]")
    g.add_argument("--lio_drift_yaw_time", type=float, default=D["yaw_time"], help="[deg/sqrt(s)]")
    g.add_argument("--lio_scale_bias_std", type=float, default=D["scale_bias"], help="[%%] per launch")
    g.add_argument("--lio_yaw_bias_std", type=float, default=D["yaw_bias"], help="[deg/m] per launch")
    g.add_argument("--lio_no_drift", action="store_true", help="Disable all drift terms")
    g.add_argument("--lio_jump_prob", type=float, default=0.0)
    g.add_argument("--lio_jump_mag", type=float, default=0.0, help="[m]")
    g.add_argument("--lio_dropouts", type=str, default="", help='"t0:dur,..." sim seconds')
    # IMU
    # IMU (on-board AHRS) error model; defaults ~ MEMS-grade IMU
    I = IMU_DEFAULTS
    g.add_argument("--imu_gyro_noise", type=float, default=I["gyro_noise"], help="[rad/s] white, per sample")
    g.add_argument("--imu_gyro_bias", type=float, default=I["gyro_bias"], help="[rad/s] std of the turn-on bias")
    g.add_argument("--imu_gyro_bias_rw", type=float, default=I["gyro_bias_rw"], help="[rad/s/sqrt(s)]")
    g.add_argument("--imu_ahrs_tau", type=float, default=I["ahrs_tau"], help="[s] roll/pitch gravity correction")
    g.add_argument("--imu_accel_tilt", type=float, default=I["accel_tilt"], help="[-] accel-induced tilt gain")
    g.add_argument("--imu_misalign", type=float, default=I["misalign"], help="[deg] roll/pitch mounting std")
    g.add_argument("--imu_no_drift", action="store_true", help="Disable all IMU error terms (yaw offset kept)")
    g.add_argument("--imu_yaw_offset", type=float, default=40.0, help="[deg] arbitrary IMU yaw (must not matter)")
    # output
    g.add_argument("--run_name", type=str, default=None)
    g.add_argument("--no_video", action="store_true")
    g.add_argument("--video_size", type=int, nargs=2, default=[960, 540], metavar=("W", "H"))
    g.add_argument("--video_every", type=int, default=1, help="Record every N-th control tick")
    args = p.parse_args()

    scripted = bool(args.keys)
    keys = args.keys
    if args.max_time is not None:
        keys = ",".join(filter(None, [keys, f"c@{args.max_time}"]))
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(LOG_ROOT, run_name) if args.log_dir == p.get_default("log_dir") else args.log_dir
    os.makedirs(run_dir, exist_ok=True)

    config = Config(args.config)
    config.default_angles = cfg.DEFAULT_DOF_POS.astype(np.float32).copy()
    config.kps = (cfg.STIFFNESS.astype(np.float32) * args.kp_scale).tolist()
    config.kds = (cfg.DAMPING.astype(np.float32) * args.kd_scale).tolist()
    env = SimG1Env(args.xml, config, launch_waist=np.deg2rad(args.launch_waist), init_yaw=np.deg2rad(args.init_yaw),
                   imu_yaw_offset=np.deg2rad(args.imu_yaw_offset),
                   **({} if args.imu_no_drift else dict(
                       imu_gyro_noise=args.imu_gyro_noise, imu_gyro_bias=args.imu_gyro_bias,
                       imu_gyro_bias_rw=args.imu_gyro_bias_rw, imu_ahrs_tau=args.imu_ahrs_tau,
                       imu_accel_tilt=args.imu_accel_tilt, imu_misalign=np.deg2rad(args.imu_misalign))),
                   key_script=keys, push_force=args.push_force, hoist=args.hoist, seed=args.seed)
    lio = FastLioEmulator(env, site=args.lio_site, rate=args.lio_rate, grav=bool(args.lio_grav),
                          latency=args.lio_latency, latency_std=args.lio_latency_std, pos_std=args.lio_pos_std,
                          rot_std=np.deg2rad(args.lio_rot_std), jump_prob=args.lio_jump_prob,
                          jump_mag=args.lio_jump_mag, dropouts=args.lio_dropouts, seed=args.seed + 1,
                          **({} if args.lio_no_drift else dict(
                              drift_xy_dist=args.lio_drift_xy_dist, drift_xy_time=args.lio_drift_xy_time,
                              drift_z_dist=args.lio_drift_z_dist, drift_yaw_dist=np.deg2rad(args.lio_drift_yaw_dist),
                              # deg/sqrt(deg) -> rad/sqrt(rad)
                              drift_yaw_rot=args.lio_drift_yaw_rot * np.sqrt(np.deg2rad(1.0)),
                              drift_yaw_time=np.deg2rad(args.lio_drift_yaw_time),
                              scale_bias_std=args.lio_scale_bias_std / 100.0,
                              yaw_bias_std=np.deg2rad(args.lio_yaw_bias_std))))
    print(lio.describe())
    env.step_callbacks.append(lio.update)
    env.settle(args.settle)

    kw = controller_kwargs(args)
    kw["log_dir"] = run_dir
    ctrl = RealTimePolicyController(env=env, odom_source=lio, realtime=(not scripted) or args.realtime,
                                    keyboard=not scripted, run_name=run_name, **kw)
    mon = SimMonitor(env, lio, ctrl)
    ctrl.tick_hooks.append(mon)
    if args.root_source == "gt":
        ctrl.root_override = mon.true_root_in_G
    if not args.no_shadow_gt:
        ctrl.shadow_root_fn = mon.true_root_in_G
    rec = None
    if not args.no_video:
        rec = SimRecorder(env, lio, mon, os.path.join(run_dir, "video.mp4"), tuple(args.video_size), args.video_every)
        ctrl.tick_hooks.append(rec)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    stop_reason = None
    try:
        stop_reason = ctrl.run()
    finally:
        if rec is not None:
            rec.close()
        summarize(ctrl, mon, lio, stop_reason, os.path.join(run_dir, "summary.json"))
        try:
            from plot_odom_log import plot
            plot(os.path.join(run_dir, f"odom_{run_name}.npz"))
        except Exception as e:  # plotting is best effort
            print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()
