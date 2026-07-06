# Track A — OpenVINS (ROS1 Noetic build for TUM-VI rosbag replay).
# Exploratory: written from upstream docs, not yet built (Phase 1, step 1).
FROM ros:noetic-perception

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3-catkin-tools build-essential \
    libeigen3-dev libboost-all-dev libceres-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /catkin_ws/src
RUN git clone --depth 1 https://github.com/rpng/open_vins.git

WORKDIR /catkin_ws
RUN . /opt/ros/noetic/setup.sh && catkin build -j4 --no-status

CMD ["bash"]
