# Track E / OAK-D Pro Option 2 mapping stage — nvblox core (GPU TSDF/ESDF).
# Exploratory: written from upstream docs, not yet built. Needs NVIDIA GPU;
# the Orin deployment is Phase 3 (experiments/06_dense_mapping). CPU lane
# stays voxblox (docker/voxblox).
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential \
    libgoogle-glog-dev libgflags-dev libgtest-dev libbenchmark-dev \
    libsqlite3-dev python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/nvidia-isaac/nvblox.git /opt/nvblox
WORKDIR /opt/nvblox/nvblox
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j4 --target nvblox_lib fuse_3dmatch

# fuse_3dmatch consumes depth + pose sequences — the runner converts the bag
# contract (depth 32FC1 + TUM poses) into its input layout.
RUN pip3 install --no-cache-dir rosbags numpy opencv-python-headless

WORKDIR /scripts
CMD ["bash"]
