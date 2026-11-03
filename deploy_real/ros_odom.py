"""Optional ROS odometry I/O for the deployed G1 controllers (ROS2 rclpy or ROS1 rospy).

Groundwork for feeding the robot's *world* root pose into the ``diff_body_*``
observation terms when ``--update_robot_w_odom`` is set:

  * real: subscribe to a FAST-LIO style ``nav_msgs/Odometry`` topic.
  * sim:  publish the MuJoCo GT root as ``nav_msgs/Odometry`` (so the same
    topic-based pipeline is exercised) and/or subscribe to it.

Backend: ``rclpy`` (ROS2) is tried first; if it is not importable, ``rospy`` (ROS1
Noetic, py3.8 -- same Python as the ``twist2`` conda env) is used. Both are imported
lazily so the default (non-odom) path works without any ROS. For ROS1, source the
catkin / noetic ``setup.bash`` and set ``ROS_MASTER_URI`` / ``ROS_IP`` before launching
(the conda env also needs ``pip install rospkg catkin_pkg``).

Quaternion convention here is wxyz to match utils.math / MuJoCo; ROS messages use
xyzw, so the conversion happens at the message boundary.
"""
from __future__ import annotations

import re
import threading
import time

import numpy as np

# Lazily-populated ROS handles (see _ensure_ros).
_ROS_OK: bool | None = None
_IMPORT_ERR: str | None = None
_BACKEND: str | None = None     # "ros2" | "ros1"
_rclpy = None
_rospy = None
_Odometry = None

_init_lock = threading.Lock()
_initialized = False


def _ensure_ros() -> bool:
    """Try rclpy (ROS2), then rospy (ROS1), + nav_msgs once. Returns True on success."""
    global _ROS_OK, _IMPORT_ERR, _BACKEND, _rclpy, _rospy, _Odometry
    if _ROS_OK is None:
        errs = []
        try:
            import rclpy  # noqa: F401
            from nav_msgs.msg import Odometry

            _rclpy, _Odometry, _BACKEND = rclpy, Odometry, "ros2"
        except Exception as e:  # ModuleNotFoundError in py3.8 twist2 env, etc.
            errs.append(f"rclpy: {e!r}")
            try:
                import rospy  # noqa: F401
                from nav_msgs.msg import Odometry

                _rospy, _Odometry, _BACKEND = rospy, Odometry, "ros1"
            except Exception as e1:
                errs.append(f"rospy: {e1!r}")
        _ROS_OK = _BACKEND is not None
        if _ROS_OK:
            print(f"[ros_odom] backend: {_BACKEND}")
        else:
            _IMPORT_ERR = "; ".join(errs)
    return _ROS_OK


def _require_ros():
    if not _ensure_ros():
        raise RuntimeError(
            "neither rclpy (ROS2) nor rospy (ROS1) + nav_msgs is importable, so odometry "
            "cannot be used in this environment. Launch the controller from a sourced ROS "
            "env (ROS1: devel/setup.bash + ROS_MASTER_URI; ROS2: Humble w/ matching Python). "
            f"Import errors: {_IMPORT_ERR}"
        )


def _init_ros(name: str):
    global _initialized
    with _init_lock:
        if not _initialized:
            if _BACKEND == "ros2":
                _rclpy.init(args=None)
            else:
                # ROS1 allows one node per process; SIGINT stays with the controller.
                _rospy.init_node(name, anonymous=True, disable_signals=True)
            _initialized = True


def _safe_node_name(prefix: str, topic: str) -> str:
    suffix = re.sub(r"[^0-9a-zA-Z_]", "_", topic).strip("_") or "topic"
    return f"{prefix}_{suffix}"


def _stamp_sec(stamp) -> float:
    if hasattr(stamp, "nanosec"):            # ROS2 builtin_interfaces/Time
        return stamp.sec + stamp.nanosec * 1e-9
    return stamp.secs + stamp.nsecs * 1e-9   # ROS1 rospy.Time


class OdomSubscriber:
    """Background subscriber to a ``nav_msgs/Odometry`` topic (e.g. FAST-LIO).

    ROS2: spins rclpy in a daemon thread. ROS1: rospy runs callbacks on its own
    threads. Caches the latest pose; ``get_pose`` returns ``(pos_xyz, quat_wxyz)``
    numpy arrays, or ``None`` until the first message arrives.
    """

    def __init__(self, topic: str, qos_depth: int = 10):
        _require_ros()
        _init_ros(_safe_node_name("twist2_odom_sub", topic))
        self.topic = topic
        self._lock = threading.Lock()
        self._pos: np.ndarray | None = None
        self._quat_wxyz: np.ndarray | None = None
        self._stamp: float | None = None
        self._recv_time: float | None = None   # local wall clock at receipt (staleness check)

        self._node = None
        if _BACKEND == "ros2":
            self._node = _rclpy.create_node(_safe_node_name("twist2_odom_sub", topic))
            self._sub = self._node.create_subscription(
                _Odometry, topic, self._cb, qos_depth
            )
            self._thread = threading.Thread(target=_rclpy.spin, args=(self._node,), daemon=True)
            self._thread.start()
        else:
            self._sub = _rospy.Subscriber(topic, _Odometry, self._cb,
                                          queue_size=1, tcp_nodelay=True)

    def _cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation  # ROS xyzw
        t = _stamp_sec(msg.header.stamp)
        with self._lock:
            self._pos = np.array([p.x, p.y, p.z], dtype=np.float64)
            self._quat_wxyz = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)
            self._stamp = t
            self._recv_time = time.time()

    def get_pose(self):
        """Return (pos_xyz, quat_wxyz) copies, or None if no message yet."""
        with self._lock:
            if self._pos is None:
                return None
            return self._pos.copy(), self._quat_wxyz.copy()

    def get_pose_with_age(self):
        """Return (pos_xyz, quat_wxyz, age_s) where age_s is the local time since the
        last message arrived, or None if no message yet."""
        with self._lock:
            if self._pos is None:
                return None
            return self._pos.copy(), self._quat_wxyz.copy(), time.time() - self._recv_time

    def close(self):
        try:
            if self._node is not None:
                self._node.destroy_node()
            else:
                self._sub.unregister()
        except Exception:
            pass


class OdomPublisher:
    """Publish a root pose as ``nav_msgs/Odometry`` (used by the sim to expose GT)."""

    def __init__(self, topic: str, frame_id: str = "odom",
                 child_frame_id: str = "base_link", qos_depth: int = 10):
        _require_ros()
        _init_ros(_safe_node_name("twist2_odom_pub", topic))
        self.topic = topic
        self._frame_id = frame_id
        self._child_frame_id = child_frame_id
        self._node = None
        if _BACKEND == "ros2":
            self._node = _rclpy.create_node(_safe_node_name("twist2_odom_pub", topic))
            self._pub = self._node.create_publisher(_Odometry, topic, qos_depth)
        else:
            self._pub = _rospy.Publisher(topic, _Odometry, queue_size=qos_depth)

    def publish(self, pos_xyz, quat_wxyz):
        msg = _Odometry()
        if self._node is not None:
            msg.header.stamp = self._node.get_clock().now().to_msg()
        else:
            msg.header.stamp = _rospy.Time.now()
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
            if self._node is not None:
                self._node.destroy_node()
            else:
                self._pub.unregister()
        except Exception:
            pass
