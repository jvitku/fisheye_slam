#!/usr/bin/env bash
# Compile + run the KLT front-end parity test (needs OpenCV C++ from fisheye/gtsam-dev).
set -euo pipefail
cd "$(dirname "$0")/.."
docker run --rm ${GUARD_DOCKER_ARGS:-} -v "$PWD:/work" -v /tmp/claude-1001:/tmp/claude-1001 -w /work fisheye/gtsam-dev bash -c '
  set -e
  g++ -O2 -std=c++17 -Wall -Ipodslam-cpp/include -I/usr/include/opencv4 \
      podslam-cpp/src/test_frontend.cpp \
      -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_calib3d \
      -o podslam-cpp/build/test_frontend
  podslam-cpp/build/test_frontend "$@"' -- "$@"
