#!/usr/bin/env python
"""Plot an odometry log (odom_<run>.npz from the controller / sim_odom.py).

    python plot_odom_log.py logs/sim2sim_odom/<run>/odom_<run>.npz   -> <same name>.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STATE_COLORS = {"PASSIVE": "#dddddd", "MOVING": "#ffe9b0", "HOLD": "#d6ecff",
                "STAND": "#e3f5dc", "MOTION": "#b9e8a8"}


def _shade(ax, t, state, states):
    if not len(t):
        return
    edges = np.flatnonzero(np.diff(state)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(t) - 1]
    for s, e in zip(starts, ends):
        ax.axvspan(t[s], t[e], color=STATE_COLORS.get(states[state[s]], "white"), lw=0, zorder=0)


def plot(npz_path, out=None):
    d = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    t, state, states = d["t"], d["state"], meta["states"]
    has = lambda k: k in d.files  # noqa: E731
    rows = [("G x / y [m]", "pos_xy"), ("G z [m]", "pos_z"), ("G yaw [deg]", "yaw"),
            ("est - GT (G, at stamp)", "err"), ("odom age [ms] / msgs", "age")]
    if has("obs_err"):
        rows.append(("obs err (max |est-GT|)", "obs"))
    fig, axes = plt.subplots(len(rows), 1, figsize=(13, 2.4 * len(rows)), sharex=True)
    for ax, (label, kind) in zip(axes, rows):
        _shade(ax, t, state, states)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
        if kind == "pos_xy":
            for i, c in ((0, "tab:red"), (1, "tab:green")):
                ax.plot(t, d["est_G_pos"][:, i], c, lw=1.2, label="xy"[i] + " est")
                if has("gt_G_pos"):
                    ax.plot(t, d["gt_G_pos"][:, i], c, lw=0.8, ls="--", label="xy"[i] + " GT")
            ax.legend(fontsize=7, ncol=4, loc="upper left")
        elif kind == "pos_z":
            ax.plot(t, d["est_G_pos"][:, 2], "tab:blue", lw=1.2, label="est")
            if has("gt_G_pos"):
                ax.plot(t, d["gt_G_pos"][:, 2], "k", lw=0.8, ls="--", label="GT")
            ax.legend(fontsize=7, loc="upper left")
        elif kind == "yaw":
            ax.plot(t, np.rad2deg(d["est_G_yaw"]), "tab:purple", lw=1.2, label="est")
            if has("gt_G_yaw"):
                ax.plot(t, np.rad2deg(d["gt_G_yaw"]), "k", lw=0.8, ls="--", label="GT")
            ax.legend(fontsize=7, loc="upper left")
        elif kind == "err":
            if has("err_G_pos"):
                e = d["err_G_pos"]
                ax.plot(t, np.linalg.norm(e[:, :2], axis=1) * 100, "tab:red", lw=1, label="xy [cm]")
                ax.plot(t, e[:, 2] * 100, "tab:blue", lw=1, label="z [cm]")
                ax.plot(t, np.rad2deg(d["err_G_yaw"]), "tab:purple", lw=1, label="yaw [deg]")
            if has("err_G_pos_now"):
                ax.plot(t, np.linalg.norm(d["err_G_pos_now"][:, :2], axis=1) * 100, "tab:orange", lw=0.8,
                        ls=":", label="xy now [cm]")
            ax.legend(fontsize=7, ncol=4, loc="upper left")
        elif kind == "age":
            ax.plot(t, np.clip(d["odom_age"], 0, 1) * 1e3, "tab:gray", lw=1, label="age [ms]")
            ax2 = ax.twinx()
            ax2.plot(t, d["odom_msgs"], "tab:cyan", lw=0.8)
            ax2.set_ylabel("msgs", fontsize=8)
        elif kind == "obs":
            names = meta.get("obs_terms", [])
            oe = d["obs_err"]
            for i, n in enumerate(names):
                if np.nanmax(np.nan_to_num(oe[:, i], nan=0.0)) > 1e-6:
                    ax.plot(t, oe[:, i], lw=0.9, label=n)
            ax.legend(fontsize=7, ncol=3, loc="upper left")
    for ev_t, ev, msg in meta.get("events", []):
        if ev in ("s", "a", "l", "b", "c", "p", "h", "stop"):
            for ax in axes:
                ax.axvline(ev_t, color="k", lw=0.6, alpha=0.5)
            axes[0].text(ev_t, axes[0].get_ylim()[1], ev.upper(), fontsize=8, va="bottom")
    axes[-1].set_xlabel("t [s]")
    fig.suptitle(f"{npz_path}  (origin_ref={meta.get('origin_ref')}, odom_frame={meta.get('odom_frame')})",
                 fontsize=9)
    fig.tight_layout()
    out = out or npz_path.replace(".npz", ".png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[plot] {out}")
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        plot(p)
