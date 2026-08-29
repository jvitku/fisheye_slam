# OAK-D Pro Option 3 — RTAB-Map mapping backend (OctoMap + dense cloud out).
# Exploratory: written from upstream docs, not yet built.
# Open-source lane: odometry from RTAB-Map stereo odom or external OpenVINS
# (3dfe/openvins on a shared roscore) — see
# experiments/08_oakdpro_slam/run_rtabmap.sh.
FROM ros:noetic-perception

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-rtabmap-ros \
    ros-noetic-octomap ros-noetic-octomap-msgs ros-noetic-octomap-server \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# rosbags: TUM-trajectory extraction from the rtabmap output in the runner.
RUN pip3 install --no-cache-dir rosbags numpy

WORKDIR /scripts
CMD ["bash"]
