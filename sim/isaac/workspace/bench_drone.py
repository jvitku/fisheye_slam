"""3d_fisheye benchmark drone: N-fisheye-camera + IMU rig in Isaac/Pegasus.

Derived from px4_drone.py (swarm_stack tools/isaac) with the lidar stack
removed and the camera rig driven by a rig yaml (RIG_CONFIG env, default
/rigs/rig_3cam.yaml). Everything else — PX4 SITL backend, ROS2 publishing,
ground-truth odometry bridge, reset-on-SIGUSR1 — matches the original so the
same MRS tooling can fly the trajectories.

Published (ROS2, bridged to ROS1 by the ros1-bridge container):
  /uav1/<cam_name>/color/image_raw   per rig camera (perfectly synced sim time;
                                     unsync variants are generated OFFLINE by
                                     bench/skew_bag.py from the recorded bag)
  /uav1/imu                          body-center IMU in sensor-local FLU
  /uav1/ground_truth                 nav_msgs/Odometry (mrs_bridge aggregator)
"""

import os
import signal
import sys
import threading

_headless = "--headless" in sys.argv

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": _headless})

import omni.kit.app
manager = omni.kit.app.get_app().get_extension_manager()

if _headless:
    manager.set_extension_enabled_immediate("omni.services.transport.server.http", True)
    manager.set_extension_enabled_immediate("omni.kit.livestream.webrtc", True)
    manager.set_extension_enabled_immediate("omni.services.livestream.nvcf", True)

    import carb.settings
    settings = carb.settings.get_settings()
    settings.set("/exts/omni.services.transport.server.http/host", "0.0.0.0")
    public_ip = os.environ.get("STREAM_PUBLIC_IP", "")
    if public_ip:
        settings.set("/app/livestream/publicEndpointAddress", public_ip)

# Enable ROS2 bridge extension (must happen before ROS2Backend import)
manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

import omni.timeline
import omni.usd
from omni.isaac.core.world import World
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend, PX4MavlinkBackendConfig
)
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from scipy.spatial.transform import Rotation
import carb

from livox_imu import LivoxIMU
from fisheye_rig import load_rig, make_cameras, apply_fisheye_projections

UAV_NAME = os.environ.get("UAV_NAME", "uav1")
RIG_CONFIG = os.environ.get("RIG_CONFIG", "/rigs/rig_3cam.yaml")


class BenchROS2Backend(ROS2Backend):
    """ROS2Backend + clean body-center IMU on /uavN/imu.

    Reuses the LivoxIMU sensor (sensor-local FLU output, no FRD/NED
    conversion — see swarm_stack tools/isaac CLAUDE.md) mounted at the body
    center, so the VIO candidates get a conventional FLU IMU that shares the
    camera clock.
    """

    def initialize_publishers(self, config):
        super().initialize_publishers(config)
        import rclpy
        from sensor_msgs.msg import Imu
        self.bench_imu_pub = self.node.create_publisher(
            Imu,
            self._namespace + str(self._id) + "/imu",
            rclpy.qos.qos_profile_sensor_data,
        )

    def update_sensor(self, sensor_type, data):
        if sensor_type == "LivoxIMU":
            self._update_bench_imu(data)
        else:
            super().update_sensor(sensor_type, data)

    def _update_bench_imu(self, data):
        from sensor_msgs.msg import Imu
        msg = Imu()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self._namespace + str(self._id) + "/base_link"
        msg.angular_velocity.x = float(data["angular_velocity"][0])
        msg.angular_velocity.y = float(data["angular_velocity"][1])
        msg.angular_velocity.z = float(data["angular_velocity"][2])
        msg.linear_acceleration.x = float(data["linear_acceleration"][0])
        msg.linear_acceleration.y = float(data["linear_acceleration"][1])
        msg.linear_acceleration.z = float(data["linear_acceleration"][2])
        self.bench_imu_pub.publish(msg)


class BenchApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self._ros2_backend = None
        self.rig = load_rig(RIG_CONFIG)
        carb.log_warn(
            f"bench_drone: rig '{self.rig.get('name', RIG_CONFIG)}' with "
            f"{len(self.rig['cameras'])} fisheye cameras"
        )

        self._load_environment()
        self._setup_vehicle()

    def _load_environment(self):
        env_name = os.environ.get("SIM_ENVIRONMENT", "Curved Gridroom")
        if os.path.exists(env_name) or env_name.startswith(("http://", "https://", "omniverse://")):
            self.pg.load_environment(env_name)
        else:
            self.pg.load_environment(SIMULATION_ENVIRONMENTS[env_name])

    def _setup_vehicle(self):
        cameras = make_cameras(self.rig)

        # Body-center IMU for the VIO candidates (sensor-local FLU).
        body_imu = LivoxIMU(config={
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        })

        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": "/opt/PX4-Autopilot",
            "px4_vehicle_model": "gazebo-classic_iris",
        })

        ros2_backend = BenchROS2Backend(vehicle_id=1, num_rotors=4, config={
            "namespace": "uav",
            "pub_sensors": True,
            "pub_graphical_sensors": True,
            "pub_state": True,
            "pub_tf": False,
        })
        self._ros2_backend = ros2_backend

        config = MultirotorConfig()
        config.sensors.append(body_imu)
        config.backends = [PX4MavlinkBackend(mavlink_config), ros2_backend]
        config.graphical_sensors = cameras

        Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config,
        )

        self.world.reset()

        # Rewrite camera prims to f-theta fisheye projection.
        stage = omni.usd.get_context().get_stage()
        apply_fisheye_projections(stage, "/World/quadrotor/body", self.rig)

        self._start_odom_bridge()

    def _start_odom_bridge(self):
        """Ground-truth Odometry aggregator on /uavN/ground_truth."""
        from mrs_bridge import OdometryBridge
        self._odom_bridge = OdometryBridge(UAV_NAME)
        self._odom_thread = threading.Thread(target=self._odom_bridge.spin, daemon=True)
        self._odom_thread.start()

    def _teardown(self):
        carb.log_warn("Tearing down simulation...")
        self.timeline.stop()
        self._odom_bridge.destroy_node()
        self._odom_thread.join(timeout=5.0)
        if self._ros2_backend is not None:
            for name, writers in self._ros2_backend.graphical_sensors_writers.items():
                for writer in writers:
                    try:
                        writer.detach()
                    except Exception as e:
                        carb.log_warn(f"Failed to detach writer {name}: {e}")
            self._ros2_backend.graphical_sensors_writers.clear()
        self.pg.clear_scene()
        self.world = self.pg.world

    def reset_simulation(self):
        carb.log_warn("Resetting simulation...")
        self._teardown()
        self._load_environment()
        for _ in range(10):
            simulation_app.update()
        self._setup_vehicle()
        self.timeline.play()
        carb.log_warn("Simulation reset complete.")

    def run(self):
        self._reset_requested = False

        def _on_sigusr1(signum, frame):
            self._reset_requested = True

        signal.signal(signal.SIGUSR1, _on_sigusr1)

        render_every_n = int(os.environ.get("RENDER_EVERY_N", "8"))
        self.timeline.play()
        step = 0
        while simulation_app.is_running():
            if self._reset_requested:
                self._reset_requested = False
                self.reset_simulation()
                step = 0
                continue
            step += 1
            self.world.step(render=(step % render_every_n == 0))
        self.timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    BenchApp().run()
