# Track E / OAK-D Pro Option 2 mapping stage — nvblox core (GPU TSDF/ESDF).
# Needs an NVIDIA GPU at run time (--gpus all); the Orin deployment is Phase 3
# (experiments/06_dense_mapping). CPU lane stays voxblox (docker/voxblox).
#
# Upstream is built from the REPO ROOT (the cmake helpers live in cmake/);
# we build only the core library + the `fuse_3dmatch` dataset fuser that
# experiments/08_oakdpro_slam/nvblox_fuse.py drives: renderer (Vulkan),
# tests, benchmarks and the PyTorch wrapper are switched off. Third-party
# deps (glog/gflags/stdgpu/sqlite) are fetched by CMake.
# NVBLOX_CUDA_ARCH: 89 = RTX 40xx (dev laptop); add 87 for Jetson Orin.
# NVBLOX_JOBS=2: the compile is what pushes a laptop CPU to its thermal
# limit (bench/guard.sh killed a -j4 build at 100 C); raise on a workstation.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ARG NVBLOX_CUDA_ARCH=89
ARG NVBLOX_JOBS=2

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential ca-certificates \
    libeigen3-dev libsqlite3-dev python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/nvidia-isaac/nvblox.git /opt/nvblox
WORKDIR /opt/nvblox
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${NVBLOX_CUDA_ARCH}" \
        -DBUILD_RENDERER=OFF -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=OFF \
        -DBUILD_PYTORCH_WRAPPER=OFF -DBUILD_EXPERIMENTS=OFF && \
    cmake --build build -j"${NVBLOX_JOBS}" --target fuse_3dmatch && \
    ls -la build/executables/fuse_3dmatch

# fuse_3dmatch consumes depth + pose sequences — the runner converts the bag
# contract (depth 32FC1 + TUM poses) into its input layout.
RUN pip3 install --no-cache-dir rosbags numpy opencv-python-headless pyyaml

WORKDIR /scripts
CMD ["bash"]
