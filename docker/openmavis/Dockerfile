# Track C — OpenMAVIS (multi-camera VI-SLAM, ORB-SLAM3 lineage).
# Exploratory: written from ORB-SLAM3-family build conventions, not yet built.
# Expect the most friction of the three C++ tracks — see experiments/03 README.
FROM ros:noetic-perception

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential ca-certificates \
    libeigen3-dev libboost-all-dev libssl-dev \
    libgl1-mesa-dev libglew-dev libepoxy-dev \
    libopencv-dev python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

# Pangolin (ORB-SLAM3 viewer dependency)
WORKDIR /src
RUN git clone --depth 1 --branch v0.8 https://github.com/stevenlovegrove/Pangolin.git && \
    cmake -S Pangolin -B Pangolin/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build Pangolin/build -j4 && cmake --install Pangolin/build

RUN git clone --depth 1 https://github.com/MAVIS-SLAM/OpenMAVIS.git
WORKDIR /src/OpenMAVIS
# Upstream follows the ORB-SLAM3 build script convention (vendored DBoW2/g2o).
RUN chmod +x build.sh && ./build.sh

CMD ["bash"]
