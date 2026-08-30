"""Rig loading: sim-style mounts and Kalibr-style T_cam_imu resolve to the same conventions."""
from pathlib import Path

import numpy as np

from podslam.rig import R_MOUNT_OPTICAL, load_rig

ROOT = Path(__file__).resolve().parents[2]


def test_sim_rig_extrinsics():
    rig = load_rig(str(ROOT / "rigs" / "oakdpro.yaml"))
    assert [c.name for c in rig.cameras] == ["cam0", "cam1"]
    c0, c1 = rig.cameras
    np.testing.assert_allclose(c0.T_imu_cam[:3, 3], [0.0, 0.0375, 0.0], atol=1e-12)
    np.testing.assert_allclose(c0.T_imu_cam[:3, :3], R_MOUNT_OPTICAL, atol=1e-12)
    # cam1 seen from cam0 (optical): 7.5 cm along optical +x (right)
    np.testing.assert_allclose(rig.T_cam_cam(0, 1)[:3, 3], [0.075, 0.0, 0.0], atol=1e-12)
    assert c0.topic == "/uav1/cam0/color/image_raw" and c0.depth_topic == "/uav1/cam0/depth/image_raw"
    assert rig.imu.topic == "/uav1/sensor_pod/imu" and rig.imu.rate_hz == 200
    assert c0.circle_mask() is None                      # pinhole: no image circle


def test_real_rig_extrinsics_and_mask():
    rig = load_rig(str(ROOT / "rigs" / "tumvi_room1.yaml"))
    c0 = rig.cameras[0]
    T_cam_imu = np.array([[-0.9995250378696743, 0.029615343885863205, -0.008522328211654736, 0.04727988224914392],
                          [0.0075019185074052044, -0.03439736061393144, -0.9993800792498829, -0.047443232143367084],
                          [-0.02989013031643309, -0.998969345370175, 0.03415885127385616, -0.0681999605066297],
                          [0, 0, 0, 1]])
    np.testing.assert_allclose(c0.T_cam_imu, T_cam_imu, atol=1e-12)
    assert c0.topic == "/cam0/image_raw" and rig.imu.topic == "/imu0"
    m = c0.circle_mask()
    assert m.shape == (512, 512) and m[256, 256] == 255 and m[2, 2] == 0
    b, ok = c0.model.unproject(np.array([[256.0, 256.0]]))
    np.testing.assert_allclose(b[0], [0, 0, 1], atol=1e-2)
