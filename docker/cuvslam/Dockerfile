# OAK-D Pro Option 2 / Track F1 — PyCuVSLAM (ROS-free cuVSLAM Python API).
# Exploratory: written from upstream docs, not yet built. Needs an NVIDIA GPU
# (--gpus all) — Orin or x86+CUDA. License: NVIDIA Isaac, free to use,
# closed source (product-path lane, docs/REPORT.md §2b).
#
# PyCuVSLAM distributes per-platform wheels from the GitHub repo (no PyPI as
# of 2026-07): https://github.com/NVlabs/PyCuVSLAM — PIN the wheel/commit on
# first build (docs/oak_d_pro_slam.md §6 open item).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir rosbags numpy opencv-python-headless pyyaml

# Upstream install path (verify against the repo README on first build):
# wheels live in the repo per python/CUDA version.
RUN git clone --depth 1 https://github.com/NVlabs/PyCuVSLAM.git /opt/pycuvslam && \
    pip3 install /opt/pycuvslam || \
    echo "WARNING: pycuvslam install needs the platform wheel — see repo README"

WORKDIR /scripts
CMD ["bash"]
