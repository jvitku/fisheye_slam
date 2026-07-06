"""
Odometry aggregator: combines Pegasus ROS2Backend pose + twist into nav_msgs/Odometry.

MRS expects ground truth as nav_msgs/msg/Odometry, but Pegasus publishes separate
PoseStamped and TwistStamped messages. This node subscribes to both and republishes
a combined Odometry message.

Runs inside the Isaac Sim container using Isaac Sim's internal rclpy.

IMPORTANT: Uses a **dedicated rclpy Context** so that OdometryBridge gets its own
isolated wait set.  Isaac Sim's bundled rclpy has a bug in wait set index management
(qos_event.py:90) — creating subscriptions on the shared global context corrupts the
wait set and crashes with ``IndexError: wait set index too big``, which then kills
ALL ROS2 publishing in the process (lidar, IMU, cameras, state — all 0 Hz).

BEST_EFFORT QoS is also used for subscriptions to match Pegasus publishers.
"""

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry

# QoS matching Pegasus ROS2Backend publishers (BEST_EFFORT)
_SENSOR_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)


class OdometryBridge(Node):
    def __init__(self, namespace: str):
        # Dedicated context — isolates our wait set from Isaac Sim's internal rclpy
        self._ctx = Context()
        self._ctx.init()

        super().__init__("odom_bridge", context=self._ctx)

        self._latest_pose = None
        self._latest_twist = None

        self.create_subscription(
            PoseStamped,
            f"/{namespace}/state/pose",
            self._pose_cb,
            _SENSOR_QOS,
        )
        self.create_subscription(
            TwistStamped,
            f"/{namespace}/state/twist",
            self._twist_cb,
            _SENSOR_QOS,
        )
        self._odom_pub = self.create_publisher(
            Odometry,
            f"/{namespace}/ground_truth",
            10,
        )

        # Own executor spinning on our dedicated context
        self._executor = SingleThreadedExecutor(context=self._ctx)
        self._executor.add_node(self)

    def _pose_cb(self, msg: PoseStamped):
        self._latest_pose = msg
        self._publish_odom()

    def _twist_cb(self, msg: TwistStamped):
        self._latest_twist = msg
        self._publish_odom()

    def _publish_odom(self):
        if self._latest_pose is None or self._latest_twist is None:
            return

        odom = Odometry()
        odom.header = self._latest_pose.header
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link"
        odom.pose.pose = self._latest_pose.pose
        odom.twist.twist = self._latest_twist.twist
        self._odom_pub.publish(odom)

    def spin(self):
        self._executor.spin()

    def destroy_node(self):
        self._executor.shutdown()
        super().destroy_node()
        self._ctx.shutdown()
