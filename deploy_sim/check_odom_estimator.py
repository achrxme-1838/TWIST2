"""Offline frame-chain and origin checks; no ROS or robot connection.

Run from any directory: python deploy_sim/check_odom_estimator.py
"""
import unittest
from pathlib import Path

import mujoco
import numpy as np

from cfg import g1_29dof_cfg as cfg
from odom_estimator import (MID360_IN_TORSO, FrameChain, OdomEstimator,
                            gravity_error, pose_mul)
from ros_odom import OdomMsg
from utils.math import rpy_to_quat


class OdomFrameChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        xml = Path(__file__).resolve().parents[1] / 'assets/g1/g1_sim2sim_29dof_deploy.xml'
        cls.model = mujoco.MjModel.from_xml_path(str(xml))
        cls.dof = cfg.DEFAULT_DOF_POS.copy()

    def feed(self, est, sensor_pose, start=1):
        for seq in range(start, start + 5):
            t = seq * 0.02
            est.push_joints(t, self.dof)
            est.update(OdomMsg(seq, t, *sensor_pose, t))
        return t

    def test_gravity_ignores_yaw_and_quaternion_sign(self):
        a = rpy_to_quat(.1, -.2, -2.0)
        b = rpy_to_quat(.1, -.2, 1.2)
        self.assertLess(gravity_error(a, -b), 1e-7)
        self.assertAlmostEqual(gravity_error(a, rpy_to_quat(.1 + np.pi, .2, 1.2)), np.pi)

    def test_reported_pose_refuses_origin(self):
        est = OdomEstimator(FrameChain(self.model))
        sensor = (np.array([.043, .267, .037]), rpy_to_quat(*np.deg2rad([178.7, 3.4, -9.4])))
        now = self.feed(est, sensor)
        ok, msg = est.reset_origin(now, self.dof, rpy_to_quat(*np.deg2rad([1.55, 4.75, -12.06])))
        self.assertFalse(ok, msg)
        self.assertIn('mounting extrinsics', msg)
        self.assertFalse(est.has_origin)
        self.assertEqual(est.origin_count, 0)

    def test_correct_upside_down_mount_and_lift(self):
        mount = list(MID360_IN_TORSO)
        mount[3] = np.pi
        chain = FrameChain(self.model, mid360_in_torso=mount)
        est = OdomEstimator(chain)
        pelvis = (np.array([.2, -.3, .755]), rpy_to_quat(.02, -.03, .4))
        sensor = pose_mul(pelvis, chain.sensor_in_pelvis(self.dof))
        now = self.feed(est, sensor)
        np.testing.assert_allclose(est.T_W_P_raw[0], pelvis[0], atol=1e-12)
        # IMU and odometry have independent yaw origins.
        imu = rpy_to_quat(.02, -.03, -1.4)
        ok, msg = est.reset_origin(now, self.dof, imu)
        self.assertTrue(ok, msg)
        before = est.root_in_G(imu)[0]
        lifted = (sensor[0] + [0, 0, .1], sensor[1])
        self.feed(est, lifted, start=6)
        np.testing.assert_allclose(est.root_in_G(imu)[0] - before, [0, 0, .1], atol=1e-12)

    def test_recorded_g1_mount_matches_pelvis_gravity(self):
        # Read-only G1 capture: gravity-aligned Livox pose, pelvis IMU and waist.
        # The two IMUs have unrelated yaw origins, so compare gravity directions.
        sensor_q = np.array([.0104440235, .9996691580, .0012008307, -.0234745589])
        pelvis_q = np.array([-.9942950010, -.0109481132, -.0396261327, -.0984261259])
        joints = self.dof.copy()
        joints[12:15] = [-.0829493999, -.0233227145, -.0810217485]
        sensor_pose = (np.zeros(3), sensor_q)
        upright = FrameChain(self.model).pelvis_from_sensor(sensor_pose, joints)
        self.assertGreater(gravity_error(upright[1], pelvis_q), np.deg2rad(170))
        mount = list(MID360_IN_TORSO)
        mount[3] = np.pi
        corrected = FrameChain(self.model, mid360_in_torso=mount).pelvis_from_sensor(sensor_pose, joints)
        self.assertLess(gravity_error(corrected[1], pelvis_q), np.deg2rad(2))

    def test_refused_reset_preserves_previous_origin(self):
        est = OdomEstimator(FrameChain(self.model, odom_frame='pelvis'))
        pose = (np.array([0., 0., .755]), rpy_to_quat(0, 0, .4))
        now = self.feed(est, pose)
        ok, msg = est.reset_origin(now, self.dof, rpy_to_quat(0, 0, 0))
        self.assertTrue(ok, msg)
        origin = est.T_W_G
        bad = (pose[0], rpy_to_quat(np.pi, 0, .4))
        now = self.feed(est, bad, start=6)
        ok, _ = est.reset_origin(now, self.dof, rpy_to_quat(0, 0, 0))
        self.assertFalse(ok)
        self.assertIs(est.T_W_G, origin)
        self.assertEqual(est.origin_count, 1)

    def test_small_mismatch_warns_but_stale_refuses(self):
        est = OdomEstimator(FrameChain(self.model, odom_frame='pelvis'))
        now = self.feed(est, (np.array([0., 0., .755]), rpy_to_quat(.1, 0, 0)))
        ok, msg = est.reset_origin(now, self.dof, rpy_to_quat(0, 0, 0))
        self.assertTrue(ok, msg)
        self.assertIn('[WARN]', msg)
        ok, msg = est.reset_origin(now + 1, self.dof, rpy_to_quat(0, 0, 0))
        self.assertFalse(ok)
        self.assertIn('not fresh', msg)

    def test_invalid_imu_refuses_origin(self):
        est = OdomEstimator(FrameChain(self.model, odom_frame='pelvis'))
        now = self.feed(est, (np.array([0., 0., .755]), rpy_to_quat(0, 0, 0)))
        for imu in (np.zeros(4), np.full(4, np.nan)):
            ok, _ = est.reset_origin(now, self.dof, imu)
            self.assertFalse(ok)
            self.assertFalse(est.has_origin)


if __name__ == '__main__':
    unittest.main()
