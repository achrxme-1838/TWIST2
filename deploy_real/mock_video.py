#!/usr/bin/env python
"""Run server_low_level_g1_real.py in --mock mode and save an mp4 of the MuJoCo robot.

The real controller is left untouched: this wrapper swaps MockG1Env for a subclass
that renders one offscreen frame per control step (send_robot_action), then calls
the controller's main(). Every other flag is forwarded to server_low_level_g1_real.py.

    python mock_video.py --policy X.onnx --motion_npz X.npz --root_source odom \
        [--video_path out.mp4] [--video_size 640 480] [--video_ref]
"""
import argparse
import os
import sys
import time

# Offscreen rendering; MuJoCo picks its GL backend at import time.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio
import mujoco
import numpy as np

import motion_player
import server_low_level_g1_real as real
from observations import _drive_ref_data
from robot_control import mock_g1_env

# Latest LocalMotionPlayer sample, for the phase overlay and the reference pane.
_last_sample = None
_orig_step = motion_player.LocalMotionPlayer.step


def _step_and_keep(self):
    global _last_sample
    _last_sample = _orig_step(self)
    return _last_sample


def _stamp(img, text):
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _free_cam():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.0
    cam.azimuth = 90.0
    cam.elevation = -20.0
    return cam


def make_recording_env(video_path, video_size, video_ref):
    class RecordingMockG1Env(mock_g1_env.MockG1Env):
        """MockG1Env + one rendered frame per control step (50 fps for control_dt 0.02)."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            w, h = video_size
            self.renderer = mujoco.Renderer(self.model, height=h, width=w)
            self.cam = _free_cam()
            self.ref_data = mujoco.MjData(self.model) if video_ref else None
            self.ref_cam = _free_cam() if video_ref else None
            self.writer = imageio.get_writer(video_path, fps=1.0 / self.config.control_dt)
            self.n_frames = 0
            print(f"[mock_video] recording {w * (2 if video_ref else 1)}x{h} "
                  f"@ {1.0 / self.config.control_dt:.0f} fps -> {video_path}")

        def send_robot_action(self, target_dof_pos, kp_scale=1.0, kd_scale=1.0):
            super().send_robot_action(target_dof_pos, kp_scale, kd_scale)
            self._record(kp_scale)

        def _record(self, kp_scale):
            pelvis = self.data.xpos[self._pelvis_id]
            self.cam.lookat[:] = pelvis
            self.renderer.update_scene(self.data, camera=self.cam)
            phase = _last_sample.phase if _last_sample is not None else "-"
            status = "HOIST" if self.hoist else phase
            img = _stamp(np.ascontiguousarray(self.renderer.render()),
                         f"t={self.sim_time:6.2f}s  {status}  gain {kp_scale:.2f}")
            if self.ref_data is not None:
                ref = np.ascontiguousarray(np.zeros_like(img))
                if _last_sample is not None:
                    _drive_ref_data(self.model, self.ref_data, _last_sample.action_mimic,
                                    self.config.num_actions, ref_root_xy_w=self.data.qpos[:2])
                    self.ref_cam.lookat[:] = self.ref_data.xpos[self._pelvis_id]
                    self.renderer.update_scene(self.ref_data, camera=self.ref_cam)
                    ref = np.ascontiguousarray(self.renderer.render())
                img = np.concatenate([img, _stamp(ref, "REF")], axis=1)
            self.writer.append_data(img)
            self.n_frames += 1

        def close(self):
            self.writer.close()
            self.renderer.close()
            print(f"[mock_video] {self.n_frames} frames ({self.n_frames * self.config.control_dt:.1f}s) "
                  f"saved to {self.writer_path}")
            super().close()

    RecordingMockG1Env.writer_path = video_path
    return RecordingMockG1Env


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--video_path", type=str, default=None,
                        help="Output mp4 (default: <repo>/logs/real/mock_<npz>_<time>.mp4)")
    parser.add_argument("--video_size", type=int, nargs=2, default=(640, 480), metavar=("W", "H"))
    parser.add_argument("--video_ref", action="store_true",
                        help="Add a second pane with the reference pose (mimic target)")
    parser.add_argument("--record_video", action="store_true", help="Accepted for sim2real.sh; always on")
    args, rest = parser.parse_known_args()

    video_path = args.video_path
    if video_path is None:
        npz = rest[rest.index("--motion_npz") + 1] if "--motion_npz" in rest else "motion"
        name = os.path.splitext(os.path.basename(npz))[0]
        video_path = os.path.join(real.DEFAULT_LOG_DIR, f"mock_{name}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    video_path = os.path.abspath(video_path)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    motion_player.LocalMotionPlayer.step = _step_and_keep
    # server_low_level_g1_real.main() imports MockG1Env from the module at call time.
    mock_g1_env.MockG1Env = make_recording_env(video_path, tuple(args.video_size), args.video_ref)
    if "--mock" not in rest:
        rest = ["--mock"] + rest
    sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_low_level_g1_real.py")] + rest
    real.main()


if __name__ == "__main__":
    main()
