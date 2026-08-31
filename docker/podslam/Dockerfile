# podslam — the in-house, permissively licensed VIO/SLAM (Track F2 product
# lane): Python front-end (OpenCV KLT / multiKLT or learned XFeat) + GTSAM
# (BSD) factor graph back-end, ML extension points throughout. GPU optional
# (learned front-end / image conditioning / minisim GPU raycaster); the
# classical path is CPU-only.
#
# One image serves the whole containerized pipeline (rule: host stays clean):
# minisim render (torch raycaster) -> flightbag/frames2bag (rosbags) ->
# podslam.track_bag (gtsam + cv2) -> bench.evaluate / map_eval.
# Deps mirror pyproject.toml — keep the two in sync.
#
# The default base is the plain PyTorch CUDA runtime. For the cuVSLAM
# side-by-side lane (machines with the closed baseline stack) build with
#   docker build --build-arg BASE=3dfe/cuvslam-ml -t fisheye/podslam-cuvslam .
ARG BASE=pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
FROM ${BASE}
RUN pip3 install --no-cache-dir \
        "gtsam==4.3a2" \
        "numpy>=1.24" \
        "opencv-python-headless>=5.0.0.93" \
        "pillow>=12.3.0" \
        "pymavlink>=2.4.49" \
        "pyyaml>=6.0" \
        "rosbags>=0.9.22" \
        "scipy>=1.15.3" \
        "pytest>=8.0" && \
    python3 -c "import gtsam, cv2, torch, numpy, rosbags, scipy; \
print('gtsam ok, cv2', cv2.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
WORKDIR /repo
CMD ["bash"]
