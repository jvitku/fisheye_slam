"""PodIMU — the sensor pod's own IMU (PX4 flight controller inside the pod).

The real sensor unit contains a standard PX4 FC used purely as a rigidly
mounted IMU source (cheap, well-characterized MEMS IMU, hardware timestamping;
streamed out via MAVLink HIGHRES_IMU). In sim we model it as a LivoxIMU-style
offset IMU at the pod origin: same rigid-body offset physics
(a_body + alpha x r + omega x (omega x r)), noise densities from the rig yaml
(rig_math.imu_sensor_config), published on its own topic (config["topic"],
/uavN/sensor_pod/imu for a single pod rig, /uavN/<ns>/imu per member of a
composite rig) so candidates consume the POD IMU while the drone's body IMU
(/uavN/imu) stays available as reference.

Composite rigs (rigs/pod3_oakdpro.yaml) carry several pod IMUs — e.g. the
pod's PX4 FC and the OAK-D Pro's BMI270 — so the Pegasus sensor type string
encodes the topic ("PodIMU:/uav1/oakd/imu") and BenchROS2Backend routes on it.
"""

__all__ = ["PodIMU"]

from livox_imu import LivoxIMU

_TYPE_PREFIX = "PodIMU:"


class PodIMU(LivoxIMU):
    def __init__(self, config={}):
        super().__init__(config)
        # Rebrand the Pegasus sensor type so the ROS2 backend routes each pod
        # IMU stream to its own topic. Pegasus's Sensor base stores the type
        # string set in Sensor.__init__; override it here.
        # VERIFY-IN-SIM: if Pegasus renames the backing attribute, backend
        # routing for "PodIMU:*" fails visibly at bring-up.
        self._sensor_type = _TYPE_PREFIX + config.get("topic", "/uav1/sensor_pod/imu")

    @staticmethod
    def is_pod_type(sensor_type) -> bool:
        return isinstance(sensor_type, str) and sensor_type.startswith(_TYPE_PREFIX)

    @staticmethod
    def topic_of(sensor_type: str) -> str:
        return sensor_type[len(_TYPE_PREFIX):]
