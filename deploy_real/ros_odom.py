"""Optional ROS2 odometry I/O for the deployed G1 controllers.

Groundwork for feeding the robot's *world* root pose into the ``diff_body_*``
observation terms when ``--update_robot_w_odom`` is set:

  * real: subscribe to a FAST-LIO style ``nav_msgs/Odometry`` topic.
  * sim:  publish the MuJoCo GT root as ``nav_msgs/Odometry`` (so the same
    topic-based pipeline is exercised) and/or subscribe to it.

``rclpy`` is imported lazily so the default (non-odom) path keeps working in the
py3.8 ``twist2`` conda env, where ROS2 Humble's rclpy (built for py3.10) is not
importable. To actually use odom, launch the controller from a sourced ROS2
(Humble / py3.10) environment.

Quaternion convention here is wxyz to match utils.math / MuJoCo; ROS messages use
xyzw, so the conversion happens at the message boundary.
"""
from __future__ import annotations

import re
import threading

import numpy as np

# Lazily-populated rclpy handles (see _ensure_ros).
_ROS_OK: bool | None = None
_IMPORT_ERR: Exception | None = None
_rclpy = None
_Odometry = None

_init_lock = threading.Lock()
_initialized = False


def _ensure_ros() -> bool:
    """Try to import rclpy + nav_msgs once. Returns True on success."""
    global _ROS_OK, _IMPORT_ERR, _rclpy, _Odometry
    if _ROS_OK is None:
        try:
            import rclpy  # noqa: F401
            from nav_msgs.msg import Odometry

            _rclpy = rclpy
            _Odometry = Odometry
            _ROS_OK = True
        except Exception as e:  # ModuleNotFoundError in py3.8 twist2 env, etc.
            _IMPORT_ERR = e
            _ROS_OK = False
    return _ROS_OK


def _require_ros():
    if not _ensure_ros():
        raise RuntimeError(
            "rclpy / nav_msgs are not importable, so --update_robot_w_odom cannot "
            "be used in this environment. Launch the controller from a sourced "
            f"ROS2 (Humble) env with a matching Python. Import error: {_IMPORT_ERR!r}"
        )


def _init_rclpy():
    global _initialized
    with _init_lock:
        if not _initialized:
            _rclpy.init(args=None)
            _initialized = True


def _safe_node_name(prefix: str, topic: str) -> str:
    suffix = re.sub(r"[^0-9a-zA-Z_]", "_", topic).strip("_") or "topic"
    return f"{prefix}_{suffix}"


class OdomSubscriber:
    """Background subscriber to a ``nav_msgs/Odometry`` topic (e.g. FAST-LIO).

    Spins rclpy in a daemon thread and caches the latest pose. ``get_pose``
    returns ``(pos_xyz, quat_wxyz)`` numpy arrays, or ``None`` until the first
    message arrives.
    """

    def __init__(self, topic: str, qos_depth: int = 10):
        _require_ros()
        _init_rclpy()
        self.topic = topic
        self._lock = threading.Lock()
        self._pos: np.ndarray | None = None
        self._quat_wxyz: np.ndarray | None = None
        self._stamp: float | None = None

        self._node = _rclpy.create_node(_safe_node_name("twist2_odom_sub", topic))
        self._sub = self._node.create_subscription(
            _Odometry, topic, self._cb, qos_depth
        )
        self._thread = threading.Thread(target=_rclpy.spin, args=(self._node,), daemon=True)
        self._thread.start()

    def _cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation  # ROS xyzw
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._lock:
            self._pos = np.array([p.x, p.y, p.z], dtype=np.float64)
            self._quat_wxyz = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)
            self._stamp = t

    def get_pose(self):
        """Return (pos_xyz, quat_wxyz) copies, or None if no message yet."""
        with self._lock:
            if self._pos is None:
                return None
            return self._pos.copy(), self._quat_wxyz.copy()

    def close(self):
        try:
            self._node.destroy_node()
        except Exception:
            pass


class OdomPublisher:
    """Publish a root pose as ``nav_msgs/Odometry`` (used by the sim to expose GT)."""

    def __init__(self, topic: str, frame_id: str = "odom",
                 child_frame_id: str = "base_link", qos_depth: int = 10):
        _require_ros()
        _init_rclpy()
        self.topic = topic
        self._frame_id = frame_id
        self._child_frame_id = child_frame_id
        self._node = _rclpy.create_node(_safe_node_name("twist2_odom_pub", topic))
        self._pub = self._node.create_publisher(_Odometry, topic, qos_depth)

    def publish(self, pos_xyz, quat_wxyz):
        msg = _Odometry()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.child_frame_id = self._child_frame_id
        msg.pose.pose.position.x = float(pos_xyz[0])
        msg.pose.pose.position.y = float(pos_xyz[1])
        msg.pose.pose.position.z = float(pos_xyz[2])
        # wxyz -> ROS xyzw
        msg.pose.pose.orientation.w = float(quat_wxyz[0])
        msg.pose.pose.orientation.x = float(quat_wxyz[1])
        msg.pose.pose.orientation.y = float(quat_wxyz[2])
        msg.pose.pose.orientation.z = float(quat_wxyz[3])
        self._pub.publish(msg)

    def close(self):
        try:
            self._node.destroy_node()
        except Exception:
            pass
