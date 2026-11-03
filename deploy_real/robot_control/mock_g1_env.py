"""MuJoCo stand-in for G1RealWorldEnv (+ a FAST-LIO-like odometry source).

Lets server_low_level_g1_real.py run end to end without the robot / unitree_interface /
ROS2: same method surface as G1RealWorldEnv, physics advanced inside send_robot_action
(the motor PD runs at 1 kHz for one control_dt per command, like the real joint
controllers), and the remote-controller buttons are scripted on sim time. A virtual
hoist (spring-damper on the pelvis) holds the robot until A is pressed, like the
real start-up procedure.
"""
import time
from types import SimpleNamespace

import mujoco
import numpy as np
from rich import print

from cfg import g1_29dof_cfg as cfg
from utils.math import quat_apply, quat_mul, rpy_to_quat

# Same bit masks as g1_wrapper.ContollerMapping (that module imports unitree_interface).
ContollerMapping = {
    "A": 0x0100, "B": 0x0200, "X": 0x0400, "Y": 0x0800,
    "R1": 0x0001, "L1": 0x0002, "start": 0x0004, "select": 0x0008,
    "R2": 0x0010, "L2": 0x0020, "F1": 0x0040, "F2": 0x0080,
    "up": 0x1000, "right": 0x2000, "down": 0x4000, "left": 0x8000,
}

SIM_DT = 0.001
BUTTON_HOLD = 0.1   # [s] how long a scripted button stays pressed
HOIST_KP, HOIST_KD = 2000.0, 200.0        # pelvis position spring [N/m, N s/m]
HOIST_ROT_KP, HOIST_ROT_KD = 300.0, 30.0  # pelvis orientation spring [N m/rad, N m s/rad]


def parse_button_script(script: str):
    """"start@0,A@3,B@5" -> [(0.0, "start"), (3.0, "A"), (5.0, "B")] (sim seconds)."""
    events = []
    for item in filter(None, (s.strip() for s in script.split(","))):
        name, t = item.split("@")
        if name not in ContollerMapping:
            raise ValueError(f"unknown button '{name}' in --mock_buttons")
        events.append((float(t), name))
    return sorted(events)


class MockG1Env:
    def __init__(self, xml_file, config, button_script="start@0,A@3,B@5", realtime=False):
        self.config = config
        self.realtime = realtime
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:3] = [0.0, 0.0, 0.757]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7:7 + config.num_actions] = config.default_angles
        mujoco.mj_forward(self.model, self.data)

        self.decimation = int(round(config.control_dt / SIM_DT))
        self.kps = np.asarray(config.kps, dtype=np.float64)
        self.kds = np.asarray(config.kds, dtype=np.float64)
        self.torque_limits = cfg.TORQUE_LIMITS
        self.controller_mapping = ContollerMapping
        self._buttons = parse_button_script(button_script)
        self._pelvis_id = self.model.body("pelvis").id
        self._hoist_pos = self.data.qpos[:3].copy()
        self.hoist = True   # released on the first A press
        print(f"[mock] MuJoCo G1 ({xml_file}); scripted buttons {self._buttons}; hoist until A")

    @property
    def sim_time(self) -> float:
        return float(self.data.time)

    def _sleep(self, dt):
        if self.realtime:
            time.sleep(dt)

    def read_controller_input(self):
        keys = 0
        for t, name in self._buttons:
            if t <= self.sim_time < t + BUTTON_HOLD:
                keys = self.controller_mapping[name]
        # Waiting loops that do not command the robot (move_to_default_pos) would never
        # advance sim time; let a button scripted at t<=0 through immediately.
        if keys == 0 and self.sim_time == 0.0:
            keys = next((self.controller_mapping[n] for t, n in self._buttons if t <= 0.0), 0)
        if self.hoist and keys == self.controller_mapping["A"]:
            self.hoist = False
            print(f"[mock] hoist released at t={self.sim_time:.2f}s")
        return SimpleNamespace(keys=keys, lx=0.0, ly=0.0, rx=0.0, ry=0.0)

    # ---- same sequence as G1RealWorldEnv ----
    def move_to_default_pos(self):
        print("[green]Waiting for the start signal to move to default pos...[/green]")
        while self.read_controller_input().keys != self.controller_mapping["start"]:
            self._sleep(self.config.control_dt)
        print("[green]Moving to default pos.[/green]")
        num_step = int(2 / self.config.control_dt)
        qpos = self.get_robot_state()[0]
        tgt_qpos = self.config.default_angles.copy()
        for i in range(num_step):
            alpha = i / num_step
            self.send_robot_action(qpos * (1 - alpha) + tgt_qpos * alpha)
            self._sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state. Waiting for the Button A signal...")
        while self.read_controller_input().keys != self.controller_mapping["A"]:
            self.send_robot_action(self.config.default_angles.copy())
            self._sleep(self.config.control_dt)

    def get_robot_state(self):
        n = self.config.num_actions
        dof_pos = self.data.qpos[7:7 + n].astype(np.float32).copy()
        dof_vel = self.data.qvel[6:6 + n].astype(np.float32).copy()
        quat = self.data.qpos[3:7].copy()                                # IMU quat, wxyz
        ang_vel = self.data.qvel[3:6].astype(np.float32).copy()          # body-frame gyro
        dof_tau = self.data.ctrl[:n].astype(np.float32).copy()
        return (dof_pos, dof_vel, quat, ang_vel,
                np.zeros((n, 2), dtype=np.float32), dof_tau, np.zeros(n, dtype=np.float32))

    def send_robot_action(self, target_dof_pos, kp_scale=1.0, kd_scale=1.0):
        n = self.config.num_actions
        target = np.asarray(target_dof_pos, dtype=np.float64)
        for _ in range(self.decimation):
            q = self.data.qpos[7:7 + n]
            dq = self.data.qvel[6:6 + n]
            tau = (target - q) * self.kps * kp_scale - dq * self.kds * kd_scale
            self.data.ctrl[:n] = np.clip(tau, -self.torque_limits, self.torque_limits)
            self._apply_hoist()
            mujoco.mj_step(self.model, self.data)

    def _apply_hoist(self):
        f = self.data.xfrc_applied[self._pelvis_id]
        if not self.hoist:
            f[:] = 0.0
            return
        pos, quat = self.data.qpos[:3], self.data.qpos[3:7]
        lin_vel = self.data.qvel[:3]
        ang_vel_w = quat_apply(quat, self.data.qvel[3:6])   # free-joint qvel[3:6] is body frame
        f[:3] = HOIST_KP * (self._hoist_pos - pos) - HOIST_KD * lin_vel
        # small-angle orientation error towards upright (identity), world frame
        q = quat if quat[0] >= 0 else -quat
        f[3:] = -HOIST_ROT_KP * 2.0 * q[1:] - HOIST_ROT_KD * ang_vel_w

    def gt_root(self):
        """Ground-truth pelvis pose (pos, quat wxyz) -- only the mock has this."""
        return self.data.qpos[:3].copy(), self.data.qpos[3:7].copy()

    def close(self):
        pass


class MockOdom:
    """FAST-LIO-like odometry from the mock sim: the pose of a sensor frame rigidly
    attached to ``body`` (offset xyz + rpy), expressed in an arbitrary fixed odom frame
    (FAST-LIO's frame is the sensor's start pose, not the robot world), sampled at
    ``rate`` Hz. Same interface as ros_odom.OdomSubscriber."""

    def __init__(self, env: MockG1Env, body="torso_link", offset=(0, 0, 0, 0, 0, 0), rate=50.0,
                 frame_xyz=(1.0, -2.0, 0.3), frame_rpy=(0.03, -0.05, 0.7)):
        self.env = env
        self.body_id = env.model.body(body).id
        self.off_pos = np.asarray(offset[:3], dtype=np.float64)
        self.off_quat = rpy_to_quat(*offset[3:])
        self.period = 1.0 / rate
        self.frame_pos = np.asarray(frame_xyz, dtype=np.float64)
        self.frame_quat = rpy_to_quat(*frame_rpy)
        self._last = None

    def _sample(self):
        d = self.env.data
        b_pos, b_quat = d.xpos[self.body_id].copy(), d.xquat[self.body_id].copy()
        s_pos = b_pos + quat_apply(b_quat, self.off_pos)
        s_quat = quat_mul(b_quat, self.off_quat)
        return (self.frame_pos + quat_apply(self.frame_quat, s_pos),
                quat_mul(self.frame_quat, s_quat), self.env.sim_time)

    def get_pose_with_age(self):
        t = self.env.sim_time
        if self._last is None or t - self._last[2] >= self.period:
            self._last = self._sample()
        pos, quat, t_s = self._last
        return pos.copy(), quat.copy(), t - t_s

    def get_pose(self):
        p = self.get_pose_with_age()
        return p[:2]

    def close(self):
        pass
