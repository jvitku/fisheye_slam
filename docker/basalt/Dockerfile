# Track B — Basalt (VIO + calibration toolchain, no ROS needed at runtime).
# Exploratory: written from upstream docs, not yet built (Phase 1, step 1).
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential ca-certificates \
    libtbb-dev libeigen3-dev libglew-dev libfmt-dev \
    libjpeg-dev libpng-dev libbz2-dev liblz4-dev libzstd-dev \
    libopencv-dev libgtest-dev libgl1-mesa-dev libwayland-dev libxkbcommon-dev \
    libboost-dev libboost-filesystem-dev libboost-date-time-dev \
    libboost-program-options-dev libboost-regex-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --recursive https://gitlab.com/VladyslavUsenko/basalt.git
WORKDIR /src/basalt
# Upstream provides an install_deps script for anything the list above misses.
RUN ./scripts/install_deps.sh || true
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j4 && cmake --install build

# Ship the TUM-VI calib/config JSONs where run.sh expects them.
RUN mkdir -p /usr/local/etc/basalt && \
    cp data/tumvi_512*.json /usr/local/etc/basalt/ || true

CMD ["bash"]
