# OAK-D Pro Option 2 / Track F1 — PyCuVSLAM (ROS-free cuVSLAM Python API).
# Needs an NVIDIA GPU at run time (--gpus all) — Orin or x86 + CUDA.
# License: NVIDIA Isaac, free to use, closed source (product-path lane,
# docs/REPORT.md §2b).
#
# PyCuVSLAM ships pre-built wheels on the cuVSLAM GitHub releases page
# (https://github.com/nvidia-isaac/cuVSLAM/releases): per Ubuntu / Python /
# CUDA-major / arch. This image = Ubuntu 22.04 + Python 3.10 + cu12 wheel
# (manylinux_2_35 = glibc 2.35 = 22.04). The wheel needs a CUDA 12 toolkit
# runtime, so the base is the 12.6.3 runtime image (the release's reference
# CUDA). Jetson Orin: same tag family with the aarch64 wheel (JetPack 6).
ARG CUVSLAM_VERSION=17.0.0
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04
ARG CUVSLAM_VERSION

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir rosbags numpy opencv-python-headless pyyaml

# Pinned wheel from the release page (fails loudly if the asset moves).
RUN set -e; W="cuvslam-${CUVSLAM_VERSION}+cu12-cp310-cp310-manylinux_2_35_x86_64.whl"; \
    curl -fL --retry 3 -o "/tmp/$W" \
      "https://github.com/nvidia-isaac/cuVSLAM/releases/download/v${CUVSLAM_VERSION}/$W" && \
    pip3 install --no-cache-dir "/tmp/$W" && rm "/tmp/$W" && \
    python3 -c "import cuvslam; print('cuvslam', getattr(cuvslam, '__version__', '?'))"

# Upstream examples / API reference for the runner (experiments/08_oakdpro_slam/cuvslam_track.py).
RUN git clone --depth 1 https://github.com/nvidia-isaac/PyCuVSLAM.git /opt/pycuvslam

WORKDIR /scripts
CMD ["bash"]
