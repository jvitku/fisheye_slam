# Track E — voxblox TSDF/ESDF mapping (CPU-only; the Pi-5-class mapper).
# Exploratory: written from upstream docs, not yet built.
FROM ros:noetic-perception

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3-catkin-tools python3-wstool build-essential \
    libgoogle-glog-dev libgflags-dev protobuf-compiler libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /catkin_ws/src
RUN git clone --depth 1 https://github.com/ethz-asl/voxblox.git && \
    wstool init . ./voxblox/voxblox_ssh.rosinstall 2>/dev/null || \
    wstool init . ./voxblox/voxblox_https.rosinstall

WORKDIR /catkin_ws
RUN . /opt/ros/noetic/setup.sh && \
    catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    catkin build voxblox_ros -j4 --no-status

CMD ["bash"]
