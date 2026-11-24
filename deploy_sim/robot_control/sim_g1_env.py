"""MuJoCo stand-in for G1RealWorldEnv (sim2sim odometry validation)."""
import numpy as np
import mujoco

from cfg import g1_29dof_cfg as cfg
from odom_estimator import LIVOX_IMU_IN_MID360, MID360_IN_TORSO, pose, pose_mul, yaw_quat
from utils.math import quat_mul

SIM_DT = 0.001
HOIST_KP_Z, HOIST_KD_Z, HOIST_SLACK = 3000.0, 300.0, 0.01   # [N/m, N s/m, m]
HOIST_KP_XY, HOIST_KD_XY = 300.0, 60.0
HOIST_KP_ROT, HOIST_KD_ROT = 300.0, 30.0                    # [N m/rad, N m s/rad]
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
GRAVITY = 9.81
IMU_RATE = 500.0                                            # [Hz] internal AHRS gyro integration rate


IMU_DEFAULTS = dict(
    gyro_noise=0.003,     # [rad/s]       gyro white noise, per sample
    gyro_bias=0.002,      # [rad/s]       std of the turn-on gyro bias (sampled once, ~0.11 deg/s)
    gyro_bias_rw=2e-5,    # [rad/s/sqrt(s)] gyro bias random walk (bias instability)
    ahrs_tau=1.0,         # [s]           AHRS gravity-correction time constant (roll/pitch only)
    accel_tilt=1.0,       # [-]           gain of the linear-acceleration induced tilt error (0: off)
    misalign=0.3,         # [deg]         std of the roll/pitch mounting misalignment (sampled once)
)


def _rotvec_quat(v):
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[np.cos(0.5 * a)], np.sin(0.5 * a) * np.asarray(v) / a])


def parse_key_script(script):
    """"s@1,a@6,b@8,c@40" -> [(1.0, 's'), ...] sorted (sim seconds)."""
    events = []
    for item in filter(None, (s.strip() for s in (script or "").split(","))):
        k, t = item.split("@")
        events.append((float(t), k.strip().lower()))
    return sorted(events)


def build_model(xml_file, mid360_in_torso=MID360_IN_TORSO, livox_imu_in_mid360=LIVOX_IMU_IN_MID360):
    spec = mujoco.MjSpec.from_file(xml_file)
    torso = spec.body("torso_link")
    T_T_L = pose(mid360_in_torso)
    T_T_LI = pose_mul(T_T_L, pose(livox_imu_in_mid360))
    torso.add_site(name="mid360", pos=T_T_L[0], quat=T_T_L[1], size=[0.02, 0, 0], rgba=[1, 0.5, 0, 1])
    torso.add_site(name="livox_imu", pos=T_T_LI[0], quat=T_T_LI[1], size=[0.01, 0, 0], rgba=[0, 1, 1, 1])
    spec.visual.global_.offwidth = 1920       # offscreen recording up to 1920x1080
    spec.visual.global_.offheight = 1080
    return spec.compile()


class SimG1Env:
    def __init__(self, xml_file, config, launch_waist=(0.0, 0.0, 0.0), init_yaw=0.0, init_xy=(0.0, 0.0),
                 imu_gyro_noise=0.0, imu_gyro_bias=0.0, imu_gyro_bias_rw=0.0, imu_ahrs_tau=1.0,
                 imu_accel_tilt=0.0, imu_misalign=0.0, imu_yaw_offset=0.0,
                 key_script="", push_force=150.0, push_time=0.1, hoist="key", seed=0):
        self.config = config
        self.model = build_model(xml_file)
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)
        self.n = config.num_actions
        self.decimation = int(round(config.control_dt / SIM_DT))
        self.kps = np.asarray(config.kps, dtype=np.float64)
        self.kds = np.asarray(config.kds, dtype=np.float64)
        self.torque_limits = np.asarray(cfg.TORQUE_LIMITS, dtype=np.float64)
        self.pelvis_id = self.model.body("pelvis").id
        self.rng = np.random.default_rng(seed)
        self.imu_rng = np.random.default_rng(seed + 2)        # separate stream: pushes stay reproducible
        self.imu_gyro_noise, self.imu_gyro_bias_rw = float(imu_gyro_noise), float(imu_gyro_bias_rw)
        self.imu_ahrs_tau, self.imu_accel_tilt = float(imu_ahrs_tau), float(imu_accel_tilt)
        self.imu_gyro_bias0 = self.imu_rng.normal(0.0, imu_gyro_bias, 3) if imu_gyro_bias > 0 else np.zeros(3)
        self.imu_gyro_bias = self.imu_gyro_bias0.copy()
        m = self.imu_rng.normal(0.0, imu_misalign, 2) if imu_misalign > 0 else np.zeros(2)
        self.imu_misalign_q = _rotvec_quat([m[0], m[1], 0.0])
        self.imu_dth = np.zeros(3)                             # AHRS attitude error (world-frame rotvec)
        self._imu_prev_vel = None
        self.imu_yaw_q = yaw_quat(imu_yaw_offset)
        self._keys = parse_key_script(key_script)
        self.push_force, self.push_steps = float(push_force), int(round(push_time / SIM_DT))
        self._push_left, self._push_f = 0, np.zeros(3)
        self.step_callbacks = []          # called after every physics advance (LIO emulator)
        if hoist not in ("key", "always", "never"):
            raise ValueError("hoist must be key / always / never")
        self.hoist_mode = hoist
        self.hoist = hoist != "never"

        # launch pose: default + waist offsets, standing on the floor
        self.launch_pose = np.asarray(config.default_angles, dtype=np.float64).copy()
        for name, v in zip(WAIST_JOINTS, launch_waist):
            self.launch_pose[cfg.SDK_JOINT_NAMES.index(name)] += float(v)
        d = self.data
        d.qpos[:2] = init_xy
        d.qpos[2] = 0.0
        d.qpos[3:7] = yaw_quat(init_yaw)
        d.qpos[7:7 + self.n] = self.launch_pose
        mujoco.mj_kinematics(self.model, d)
        feet = [i for i in range(self.model.ngeom)
                if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
                and self.model.body(self.model.geom_bodyid[i]).name.endswith("ankle_roll_link")]
        d.qpos[2] = -float(np.min(d.geom_xpos[feet, 2] - self.model.geom_size[feet, 0])) + 0.002
        mujoco.mj_forward(self.model, d)
        self._hoist_pos = d.qpos[:3].copy()

    # ----- clock / scripted input -----
    def now(self):
        return float(self.data.time)

    def poll_events(self):
        out = []
        while self._keys and self._keys[0][0] <= self.now() + 1e-9:
            out.append(self._keys.pop(0)[1])
        return out

    # ----- physics -----
    def _advance(self, target, kp_scale, kd_scale):
        m, d, n = self.model, self.data, self.n
        target = np.asarray(target, dtype=np.float64)
        for _ in range(self.decimation):
            tau = (target - d.qpos[7:7 + n]) * self.kps * kp_scale - d.qvel[6:6 + n] * self.kds * kd_scale
            d.ctrl[:n] = np.clip(tau, -self.torque_limits, self.torque_limits)
            f = self._hoist_wrench()
            if self._push_left > 0:
                f[:3] += self._push_f
                self._push_left -= 1
            d.xfrc_applied[self.pelvis_id] = f
            mujoco.mj_step(m, d)
        mujoco.mj_kinematics(m, d)   # xpos/site poses of the *current* qpos (mj_step leaves them 1 substep old)
        self._imu_update(self.decimation * SIM_DT)
        for cb in self.step_callbacks:
            cb()

    def _hoist_wrench(self):
        f = np.zeros(6)
        if not self.hoist:
            return f
        d = self.data
        pos, vel = d.qpos[:3], d.qvel[:3]
        dz = self._hoist_pos[2] - HOIST_SLACK - pos[2]
        if dz > 0:
            f[2] = HOIST_KP_Z * dz - HOIST_KD_Z * min(vel[2], 0.0)
        f[:2] = HOIST_KP_XY * (self._hoist_pos[:2] - pos[:2]) - HOIST_KD_XY * vel[:2]
        R = d.xmat[self.pelvis_id].reshape(3, 3)
        w = R @ d.qvel[3:6]                                    # world angular velocity
        tilt = np.cross(R[:, 2], np.array([0.0, 0.0, 1.0]))    # rotate body z towards world z
        f[3:5] = HOIST_KP_ROT * tilt[:2] - HOIST_KD_ROT * w[:2]
        return f

    def toggle_hoist(self):
        """H key (hoist="key"): lower / re-attach the virtual hoist (re-attached above the current xy).
        It is also lowered automatically at A (on_policy_start)."""
        if self.hoist_mode != "key":
            return f"fixed ({self.hoist_mode})"
        self.hoist = not self.hoist
        if self.hoist:
            self._hoist_pos[:2] = self.data.qpos[:2]
        return f"{'attached' if self.hoist else 'released'} at t={self.now():.2f}s"

    def on_policy_start(self):
        """A (hoist="key"): the hoist holds the robot from the start up to A, then it is lowered."""
        if self.hoist_mode == "key" and self.hoist:
            return self.toggle_hoist()
        return None

    def send_robot_action(self, target_dof_pos, kp_scale=1.0, kd_scale=1.0):
        self._advance(target_dof_pos, kp_scale, kd_scale)

    def passive_step(self):
        self._advance(self.launch_pose, 1.0, 1.0)

    def settle(self, seconds):
        for _ in range(int(round(seconds / self.config.control_dt))):
            self.passive_step()

    def push(self):
        a = self.rng.uniform(0, 2 * np.pi)
        self._push_f = self.push_force * np.array([np.cos(a), np.sin(a), 0.0])
        self._push_left = self.push_steps
        return f"{self.push_force:.0f} N towards {np.rad2deg(a):.0f} deg for {self.push_steps * SIM_DT:.2f}s"

    # ----- sensors -----
    def _imu_update(self, dt):
        """Advance the gyro bias and the AHRS attitude error by one control tick."""
        d, r = self.data, self.imu_rng
        if self.imu_gyro_bias_rw > 0:
            self.imu_gyro_bias = self.imu_gyro_bias + r.normal(0.0, self.imu_gyro_bias_rw * np.sqrt(dt), 3)
        R = d.xmat[self.pelvis_id].reshape(3, 3)
        # integrated gyro error: bias + white noise sampled at IMU_RATE (angle random walk)
        w_err = self.imu_gyro_bias * dt
        if self.imu_gyro_noise > 0:
            w_err = w_err + r.normal(0.0, self.imu_gyro_noise * np.sqrt(dt / IMU_RATE), 3)
        self.imu_dth = self.imu_dth + R @ w_err
        # accelerometer tilt reference: specific force f = a + g_up; AHRS aligns f with "up"
        vel = d.qvel[:3].copy()                                # free joint: world-frame linear velocity
        acc = np.zeros(3) if self._imu_prev_vel is None else (vel - self._imu_prev_vel) / dt
        self._imu_prev_vel = vel
        acc = acc * self.imu_accel_tilt
        fz = GRAVITY + acc[2]
        ref = np.array([np.arctan2(acc[1], fz), -np.arctan2(acc[0], fz)])
        if self.imu_ahrs_tau > 0:                               # roll/pitch corrected, yaw drifts
            self.imu_dth[:2] += (1.0 - np.exp(-dt / self.imu_ahrs_tau)) * (ref - self.imu_dth[:2])

    def imu_error(self):
        """(world-frame attitude error rotvec [rad], current gyro bias [rad/s]) for logging."""
        return self.imu_dth.copy(), self.imu_gyro_bias.copy()

    def get_robot_state(self):
        d, n = self.data, self.n
        dof_pos = d.qpos[7:7 + n].astype(np.float64).copy()
        dof_vel = d.qvel[6:6 + n].astype(np.float64).copy()
        q = quat_mul(_rotvec_quat(self.imu_dth), quat_mul(d.xquat[self.pelvis_id], self.imu_misalign_q))
        q = quat_mul(self.imu_yaw_q, q)
        gyro = d.qvel[3:6].copy() + self.imu_gyro_bias            # free joint: body-frame angular velocity
        if self.imu_gyro_noise > 0:
            gyro = gyro + self.imu_rng.normal(0.0, self.imu_gyro_noise, 3)
        tau = d.ctrl[:n].copy()
        return (dof_pos, dof_vel, q / np.linalg.norm(q), gyro.astype(np.float32),
                np.zeros((n, 2), dtype=np.float32), tau, np.zeros(n, dtype=np.float32))

    def gt_pelvis(self):
        return self.data.xpos[self.pelvis_id].copy(), self.data.xquat[self.pelvis_id].copy()

    def close(self):
        pass
