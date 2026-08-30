"""podslam — the in-house, permissively licensed multi-camera visual-inertial odometry.

Product lane of fisheye_slam (docs/REPORT.md §2b, "Track F2"): everything in
this package is ours or BSD/Apache — OpenCV (Apache-2.0) for classical image
ops, GTSAM (BSD) for the factor-graph optimisation and IMU preintegration,
PyTorch (BSD) for the learned plug-ins. No GPL code, no closed cores.

Architecture (one frame in, one pose out — see docs/podslam.md):

    images ─► conditioning ─► Frontend ─► Tracker (keyframes, landmarks) ─► Backend ─► pose
                 ▲              ▲                                             ▲
             plug-in        plug-in (KLT | XFeat | yours)                GTSAM fixed-lag smoother
    imu ───────────────────► IMU (static init, preintegration, KLT prediction)

Extension points (all keyed by config strings, all replaceable in Python):
    podslam.frontend.Frontend     feature detection / tracking / stereo matching
    podslam.conditioning          per-frame image conditioning (CLAHE, norm, learned enhancer)
    podslam.conditioning.masks    static + per-frame masks (image circle, saturation, learned)
    podslam.backend.Backend       the estimator (fixed-lag smoother today)
Outputs and inputs follow the benchmark contract (bench/README.md): same bag
topics, same rig yaml, TUM trajectory in the IMU frame, per-frame stats csv.
"""

__version__ = "0.1.0"
__license__ = "Apache-2.0"
