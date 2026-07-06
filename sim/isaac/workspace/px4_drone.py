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

    # Bind HTTP server to all interfaces and advertise public IP for remote WebRTC
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
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from livox_imu import LivoxIMU


class WallClockROS2Backend(ROS2Backend):
    """ROS2Backend that publishes lidar with wall-clock timestamps to match IMU.

    Pegasus's default ROS2Backend uses the sim-time lidar writer
    (RtxLidarROS2PublishPointCloud) whose timestamps come from Isaac's
    IsaacReadSimulationTime annotator (secs ~371), while the IMU uses
    wall-clock time (secs ~1.7 billion).  FAST-LIO's sync_packages()
    can never correlate them.  This subclass overrides the lidar writer
    to use the SystemTime variants so both sensors share the same clock.
    """

    def initialize_publishers(self, config):
        super().initialize_publishers(config)
        import rclpy
        from sensor_msgs.msg import Imu, CompressedImage
        self.livox_imu_pub = self.node.create_publisher(
            Imu,
            self._namespace + str(self._id) + "/livox/imu",
            rclpy.qos.qos_profile_sensor_data,
        )
        self._compressed_camera_pubs = {}
        for cam_name in ("camera_front", "camera_down"):
            self._compressed_camera_pubs[cam_name] = self.node.create_publisher(
                CompressedImage,
                self._namespace + str(self._id) + "/" + cam_name + "/image_raw/compressed",
                rclpy.qos.qos_profile_sensor_data,
            )

    def add_monocular_camera_writter(self, data):
        super().add_monocular_camera_writter(data)
        cam_name = data["camera_name"]
        if cam_name in self._compressed_camera_pubs:
            import rclpy
            from sensor_msgs.msg import Image
            pub = self._compressed_camera_pubs[cam_name]
            self.node.create_subscription(
                Image,
                self._namespace + str(self._id) + "/" + cam_name + "/color/image_raw",
                lambda msg, p=pub: self._compress_and_publish_camera(msg, p),
                rclpy.qos.qos_profile_sensor_data,
            )

    def _compress_and_publish_camera(self, msg, publisher):
        """Convert raw Image to CompressedImage (JPEG) using Pillow."""
        import numpy as np
        from PIL import Image as PILImage
        import io
        from sensor_msgs.msg import CompressedImage

        channels = len(msg.data) // (msg.height * msg.width)
        img_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, channels
        )
        # Handle RGBA -> RGB for JPEG
        if channels == 4:
            img_array = img_array[:, :, :3]
        pil_img = PILImage.fromarray(img_array)

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=80)

        compressed_msg = CompressedImage()
        compressed_msg.header = msg.header
        compressed_msg.format = "jpeg"
        compressed_msg.data = buf.getvalue()
        publisher.publish(compressed_msg)

    def update_sensor(self, sensor_type, data):
        if sensor_type == "LivoxIMU":
            self._update_livox_imu(data)
        else:
            super().update_sensor(sensor_type, data)

    def _update_livox_imu(self, data):
        from sensor_msgs.msg import Imu
        msg = Imu()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self._namespace + "_" + "base_link_frd"

        msg.angular_velocity.x = float(data["angular_velocity"][0])
        msg.angular_velocity.y = float(data["angular_velocity"][1])
        msg.angular_velocity.z = float(data["angular_velocity"][2])

        msg.linear_acceleration.x = float(data["linear_acceleration"][0])
        msg.linear_acceleration.y = float(data["linear_acceleration"][1])
        msg.linear_acceleration.z = float(data["linear_acceleration"][2])

        self.livox_imu_pub.publish(msg)

    def add_lidar_writter(self, data):
        import omni.replicator.core as rep

        render_prod_path = rep.create.render_product(
            data["stage_prim_path"], [1, 1], name=data["lidar_name"],
        )

        writer = rep.writers.get("RtxLidarROS2SystemTimePublishPointCloud")
        writer.initialize(
            nodeNamespace=self._namespace + str(self._id),
            topicName=data["lidar_name"] + "/pointcloud",
            frameId=self._namespace + str(self._id) + "/livox",
        )
        writer.attach([render_prod_path])
        self.graphical_sensors_writers[data["lidar_name"]] = [writer]


from pegasus.simulator.logic.graphical_sensors.lidar import Lidar
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from scipy.spatial.transform import Rotation
import carb
import numpy as np

UAV_NAME = os.environ.get("UAV_NAME", "uav1")

def apply_mid360_fov(prim):
    """Override Example_Rotary emitter angles to approximate Livox Mid-360
    non-repetitive scanning pattern.

    Keeps 128 emitters (required by Example_Rotary model) with shuffled
    elevations spanning -7° to +52°. Reduces reportRate to 100 Hz so each
    revolution has only 10 firings × 128 emitters ≈ 1,280 points/frame,
    matching Gazebo Livox plugin density (~1,250 pts/frame).
    """
    from pxr import Vt

    def _set(name, val):
        attr = prim.GetAttribute(name)
        if attr and attr.IsValid():
            attr.Set(val)
        else:
            carb.log_warn(f"Attribute {name} not found on lidar prim")

    n = 128  # Must match Example_Rotary numberOfEmitters
    rng = np.random.default_rng(seed=42)

    # Elevations: uniform coverage of -7° to +52° FOV, shuffled to break row ordering
    elevations = np.linspace(-7.0, 52.0, n)
    rng.shuffle(elevations)

    # Azimuths: spread across full 360° with random jitter
    # Each firing instant covers all azimuths, eliminating visible rotary sweep
    azimuths = np.linspace(0.0, 360.0 * (1 - 1/n), n)
    rng.shuffle(azimuths)
    azimuths += rng.uniform(-1.0, 1.0, size=n)

    _set("omni:sensor:Core:emitterState:s001:elevationDeg",
         Vt.FloatArray([round(float(e), 2) for e in elevations]))
    _set("omni:sensor:Core:emitterState:s001:azimuthDeg",
         Vt.FloatArray([round(float(a), 2) for a in azimuths]))

    # Mid-360 range: 40m (Example_Rotary default is 200m)
    _set("omni:sensor:Core:farRangeM", 40.0)

    # Match expected 10Hz scan rate for FAST-LIO
    _set("omni:sensor:Core:scanRateBaseHz", 10.0)
    # ~1,280 points/frame: 10 firings/rev × 128 emitters
    _set("omni:sensor:Core:reportRateBaseHz", 100.0)  # 10 firings/rev × 10 Hz

    carb.log_info(f"Applied Mid-360 non-repetitive pattern: {n} emitters, "
                  f"elevation [{min(elevations):.1f}..{max(elevations):.1f}] deg, "
                  f"azimuth [0..360] deg, range 40m, ~1280 pts/frame")


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self._ros2_backend = None

        self._load_environment()
        self._setup_vehicle()

    def _load_environment(self):
        env_name = os.environ.get("SIM_ENVIRONMENT", "Curved Gridroom")
        if os.path.exists(env_name) or env_name.startswith(("http://", "https://", "omniverse://")):
            self.pg.load_environment(env_name)
        else:
            self.pg.load_environment(SIMULATION_ENVIRONMENTS[env_name])

    def _setup_vehicle(self):
        # -- Sensors --
        front_camera = MonocularCamera("camera_front", config={
            "position": [0.10, 0.0, 0.0],
            "orientation": [0.0, 0.0, 180.0],  # Forward-facing (ZYX Euler, degrees)
            "resolution": (640, 480),
            "frequency": 5,
            "diagonal_fov": 90.0,
            "depth": False,
        })

        down_camera = MonocularCamera("camera_down", config={
            "position": [0.0, 0.0, -0.05],
            "orientation": [0.0, 90.0, 180.0],  # Pitched down 90 deg (ZYX Euler)
            "resolution": (640, 480),
            "frequency": 5,
            "diagonal_fov": 90.0,
            "depth": False,
        })

        lidar = Lidar("livox_mid360", config={
            "position": [0.086, 0.0, 0.068],
            "orientation": [0.0, 25.0, 0.0],       # 25° pitch to match goodai_3 mount (ZYX Euler, deg)
            "frequency": 10.0,
            "sensor_configuration": {"sensor_configuration": "Example_Rotary"},
            "show_render": False,
        })

        livox_imu = LivoxIMU(config={
            "position": [0.086, 0.0, 0.068],
            "orientation": [0.0, 25.0, 0.0],    # 25° pitch matching LiDAR mount (ZYX Euler, deg)
        })

        # -- Backends --
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": "/opt/PX4-Autopilot",
            "px4_vehicle_model": "gazebo-classic_iris",
        })

        ros2_backend = WallClockROS2Backend(vehicle_id=1, num_rotors=4, config={
            "namespace": "uav",
            "pub_sensors": True,
            "pub_graphical_sensors": True,
            "pub_state": True,
            "pub_tf": False,
        })
        self._ros2_backend = ros2_backend

        config = MultirotorConfig()
        config.sensors.append(livox_imu)
        config.backends = [PX4MavlinkBackend(mavlink_config), ros2_backend]
        # config.graphical_sensors = [front_camera, down_camera, lidar]
        config.graphical_sensors = [front_camera, lidar]

        Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config,
        )

        self.world.reset()

        # Override Example_Rotary elevation/range to match Mid-360 FOV
        stage = omni.usd.get_context().get_stage()
        lidar_prim = stage.GetPrimAtPath("/World/quadrotor/body/livox_mid360")
        if lidar_prim and lidar_prim.IsValid():
            apply_mid360_fov(lidar_prim)

        # Start odometry aggregator (pose+twist -> Odometry) in background thread
        self._start_odom_bridge()

    def _start_odom_bridge(self):
        """Launch the MRS odometry bridge as a background thread."""
        from mrs_bridge import OdometryBridge
        self._odom_bridge = OdometryBridge(UAV_NAME)
        self._odom_thread = threading.Thread(target=self._odom_bridge.spin, daemon=True)
        self._odom_thread.start()

    def _teardown(self):
        carb.log_warn("Tearing down simulation...")
        self.timeline.stop()
        self._odom_bridge.destroy_node()
        self._odom_thread.join(timeout=5.0)

        # Detach Replicator writers before clearing scene to prevent stale references
        if self._ros2_backend is not None:
            import omni.replicator.core as rep
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
        # Pump Kit event loop to let async init from clear_scene() settle
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
    PegasusApp().run()
