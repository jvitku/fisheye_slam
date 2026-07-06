"""PodIMU — the sensor pod's own IMU (PX4 flight controller inside the pod).

The real sensor unit contains a standard PX4 FC used purely as a rigidly
mounted IMU source (cheap, well-characterized MEMS IMU, hardware timestamping;
streamed out via MAVLink HIGHRES_IMU). In sim we model it as a LivoxIMU-style
offset IMU at the pod origin: same rigid-body offset physics
(a_body + alpha x r + omega x (omega x r)), same Pegasus-default noise model,
published on its own topic (/uavN/sensor_pod/imu) so candidates consume the
POD IMU while the drone's body IMU (/uavN/imu) stays available as reference.
"""

__all__ = ["PodIMU"]

from livox_imu import LivoxIMU


class PodIMU(LivoxIMU):
    def __init__(self, config={}):
        super().__init__(config)
        # Rebrand the Pegasus sensor type so the ROS2 backend routes pod and
        # body IMU streams to different topics. Pegasus's Sensor base stores
        # the type string set in Sensor.__init__; override it here.
        # VERIFY-IN-SIM: if Pegasus renames the backing attribute, backend
        # routing for "PodIMU" fails visibly at bring-up.
        self._sensor_type = "PodIMU"
