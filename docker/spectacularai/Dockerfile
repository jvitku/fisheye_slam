# OAK-D Pro Option 1 — Spectacular AI SDK replay (docs/oak_d_pro_slam.md §2).
# Exploratory: written from upstream docs, not yet built.
#
# LICENSE: the spectacularAI PyPI wheel is free for NON-COMMERCIAL use on
# x86_64 Linux/Windows only. Benchmark yardstick lane — a product deployment
# needs a commercial license (Track F matrix). Jetson builds are distributed
# separately by Spectacular AI (not via PyPI).
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
# ffmpeg: bag2sai.py writes the SAI recording's per-camera lossless video.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    spectacularAI \
    rosbags \
    numpy \
    scipy \
    opencv-python-headless \
    pyyaml

# Runners mount experiments/08_oakdpro_slam at /scripts, bag at /data,
# output at /out (see run_spectacularai.sh).
WORKDIR /scripts
CMD ["bash"]
