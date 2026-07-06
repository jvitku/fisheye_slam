"""
LivoxIMU sensor — computes physically correct IMU readings at the LiDAR
offset position on the rigid body.

The real Livox Mid-360 has an IMU co-located with the LiDAR.  Pegasus's
built-in IMU is at the drone body center.  During rotation, the offset
causes centripetal/tangential acceleration terms that must be modelled
so FAST-LIO sees the same physics as with real hardware.

Rigid-body kinematics at offset r from body center:
    a_offset = a_body + alpha x r + omega x (omega x r)
where alpha = d(omega)/dt, and omega is the same everywhere on the body.
"""
__all__ = ["LivoxIMU"]

import numpy as np
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.sensors import Sensor
from pegasus.simulator.logic.sensors.geo_mag_utils import GRAVITY_VECTOR


class LivoxIMU(Sensor):
    """IMU sensor positioned at an offset from the body center (e.g. at the LiDAR)."""

    def __init__(self, config={}):
        super().__init__(sensor_type="LivoxIMU", update_rate=config.get("update_rate", 250.0))

        # Offset from body center in body FLU frame [x_forward, y_left, z_up]
        self._offset = np.array(config.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)

        # Sensor orientation relative to body FLU (ZYX Euler degrees, same as Lidar config)
        ori = config.get("orientation", [0.0, 0.0, 0.0])
        self._R_body_sensor = Rotation.from_euler("ZYX", ori, degrees=True)
        self._R_sensor_body = self._R_body_sensor.inv()

        # --- Noise parameters (same defaults as Pegasus IMU) ---
        gyro = config.get("gyroscope", {})
        self._gyro_noise_density = gyro.get("noise_density", 0.0003393695767766752)
        self._gyro_random_walk = gyro.get("random_walk", 3.878509448876288e-05)
        self._gyro_bias_corr_time = gyro.get("bias_correlation_time", 1.0e3)
        self._gyro_turn_on_bias_sigma = gyro.get("turn_on_bias_sigma", 0.008726646259971648)
        self._gyro_bias = np.zeros(3)

        accel = config.get("accelerometer", {})
        self._accel_noise_density = accel.get("noise_density", 0.004)
        self._accel_random_walk = accel.get("random_walk", 0.006)
        self._accel_bias_corr_time = accel.get("bias_correlation_time", 300.0)
        self._accel_turn_on_bias_sigma = accel.get("turn_on_bias_sigma", 0.196)
        self._accel_bias = np.zeros(3)

        # Previous-step state for finite differences
        self._prev_linear_velocity = np.zeros(3)
        self._prev_angular_velocity = np.zeros(3)

        self._state = {
            "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
            "angular_velocity": np.array([0.0, 0.0, 0.0]),
            "linear_acceleration": np.array([0.0, 0.0, 0.0]),
        }

    @property
    def state(self):
        return self._state

    @Sensor.update_at_rate
    def update(self, state: State, dt: float):
        # --- Gyroscope noise (identical to Pegasus IMU) ---
        tau_g = self._gyro_bias_corr_time
        sigma_g_d = 1.0 / np.sqrt(dt) * self._gyro_noise_density
        sigma_b_g = self._gyro_random_walk
        sigma_b_g_d = np.sqrt(-sigma_b_g**2 * tau_g / 2.0 * (np.exp(-2.0 * dt / tau_g) - 1.0))
        phi_g_d = np.exp(-1.0 / tau_g * dt)

        angular_velocity = np.zeros(3)
        for i in range(3):
            self._gyro_bias[i] = phi_g_d * self._gyro_bias[i] + sigma_b_g_d * np.random.randn()
            angular_velocity[i] = state.angular_velocity[i] + sigma_g_d * np.random.randn() + self._gyro_bias[i]

        # --- Body-center linear acceleration (same as Pegasus IMU) ---
        tau_a = self._accel_bias_corr_time
        sigma_a_d = 1.0 / np.sqrt(dt) * self._accel_noise_density
        sigma_b_a = self._accel_random_walk
        sigma_b_a_d = np.sqrt(-sigma_b_a**2 * tau_a / 2.0 * (np.exp(-2.0 * dt / tau_a) - 1.0))
        phi_a_d = np.exp(-1.0 / tau_a * dt)

        # Inertial-frame acceleration from finite-differencing velocity, minus gravity
        linear_accel_inertial = (state.linear_velocity - self._prev_linear_velocity) / dt - GRAVITY_VECTOR
        self._prev_linear_velocity = state.linear_velocity.copy()

        # Rotate into body FLU frame
        body_accel = Rotation.from_quat(state.attitude).inv().apply(linear_accel_inertial)

        # --- Rigid-body offset correction ---
        # Angular acceleration (body FLU) from finite-differencing angular velocity
        omega = state.angular_velocity  # body FLU
        alpha = (omega - self._prev_angular_velocity) / dt
        self._prev_angular_velocity = omega.copy()

        r = self._offset
        # a_offset = a_body + alpha x r + omega x (omega x r)
        offset_accel = body_accel + np.cross(alpha, r) + np.cross(omega, np.cross(omega, r))

        # --- Accelerometer noise ---
        for i in range(3):
            self._accel_bias[i] = phi_a_d * self._accel_bias[i] + sigma_b_a_d * np.random.randn()
            offset_accel[i] += sigma_a_d * np.random.randn()

        # --- Rotate from body FLU into sensor-local FLU frame ---
        # The real Livox Mid-360 IMU is co-oriented with the LiDAR (e.g. 25° pitch).
        # Express measurements in the sensor's own FLU frame.
        sensor_accel = self._R_sensor_body.apply(offset_accel)
        sensor_omega = self._R_sensor_body.apply(angular_velocity)

        # Output in sensor FLU / ENU convention (NO FRD/NED conversion).
        # The Pegasus body-center IMU converts to FRD/NED for PX4, but FAST-LIO
        # expects FLU — matching the real Livox Mid-360 driver output.
        # At rest & level: accel ≈ [0, 0, +9.81] before sensor pitch rotation.
        attitude_sensor_enu = Rotation.from_quat(state.attitude) * self._R_body_sensor

        self._state = {
            "orientation": attitude_sensor_enu.as_quat(),
            "angular_velocity": sensor_omega,
            "linear_acceleration": sensor_accel,
        }
        return self._state
