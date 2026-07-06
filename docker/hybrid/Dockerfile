# Track D — learned components: XFeat, LightGlue, DBA-Fusion (desktop GPU).
# Exploratory: not yet built. For CPU-only XFeat experiments, a plain
# `uv run --with torch ...` on the host is fine (see experiments/04 README).
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-pip python3-dev build-essential cmake \
    libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir \
    opencv-python-headless kornia numpy scipy matplotlib tqdm evo

WORKDIR /src
# XFeat + LightGlue (pip-installable from source)
RUN git clone --depth 1 https://github.com/verlab/accelerated_features.git && \
    git clone --depth 1 https://github.com/cvg/LightGlue.git && \
    pip3 install --no-cache-dir -e LightGlue

# DBA-Fusion needs GTSAM + DROID-SLAM CUDA extensions; its own README drives
# the rest of the setup (kept manual while exploratory):
RUN git clone --depth 1 https://github.com/GREAT-WHU/DBA-Fusion.git

CMD ["bash"]
