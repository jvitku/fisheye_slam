#!/usr/bin/env bash
# Compile + run the KLT front-end parity test (needs OpenCV C++ from fisheye/gtsam-dev).
set -euo pipefail
cd "$(dirname "$0")/.."
docker run --rm ${GUARD_DOCKER_ARGS:-} -v "$PWD:/work" -v /tmp/claude-1001:/tmp/claude-1001 -w /work fisheye/gtsam-dev bash -c '
  set -e
  # pinned OpenCV 5.0.0 (+IPP), the same version/config as the Python cv2 wheel
  g++ -O2 -std=c++17 -Wall -Ipodslam-cpp/include -Ipodslam-cpp/opencv-install/include/opencv5 \
      podslam-cpp/src/test_frontend.cpp \
      -Lpodslam-cpp/opencv-install/lib \
      -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_geometry -lopencv_flann \
      -Wl,-rpath,/work/podslam-cpp/opencv-install/lib \
      -o podslam-cpp/build/test_frontend
  LD_LIBRARY_PATH=/work/podslam-cpp/opencv-install/lib podslam-cpp/build/test_frontend "$@"' -- "$@"
