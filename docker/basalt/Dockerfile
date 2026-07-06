# Track B — Basalt (VIO + calibration toolchain, no ROS needed at runtime).
# Exploratory: written from upstream docs, not yet built (Phase 1, step 1).
# Pinned to 24acebe (2024-08-31, "Updates to support GCC13") — the last
# pre-vcpkg state of master. Everything after f252308 (2026-03-17) builds via
# vcpkg CONFIG packages (opengv/CLI11/magic_enum/Boost-format) that apt can't
# satisfy; this commit builds classically with vendored submodules.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential ca-certificates \
    libtbb-dev libeigen3-dev libglew-dev libfmt-dev \
    libjpeg-dev libpng-dev libbz2-dev liblz4-dev libzstd-dev \
    libopencv-dev libgtest-dev libgl1-mesa-dev libwayland-dev libxkbcommon-dev \
    libepoxy-dev \
    libboost-dev libboost-filesystem-dev libboost-date-time-dev \
    libboost-program-options-dev libboost-regex-dev libboost-serialization-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone https://gitlab.com/VladyslavUsenko/basalt.git && \
    cd basalt && git checkout 24acebe && \
    git submodule update --init --recursive --depth 1
WORKDIR /src/basalt
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j8 && cmake --install build

# Ship the TUM-VI calib/config JSONs where run.sh expects them.
RUN mkdir -p /usr/local/etc/basalt && \
    cp data/tumvi_512*.json /usr/local/etc/basalt/ || true

CMD ["bash"]
