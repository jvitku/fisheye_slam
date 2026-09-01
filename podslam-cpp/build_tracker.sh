#!/usr/bin/env bash
# Compile + run the end-to-end C++ tracker (frontend + window) on the frame dump.
set -euo pipefail
cd "$(dirname "$0")/.."
docker run --rm ${GUARD_DOCKER_ARGS:-} ${PODSLAM_CPU_QUOTA:+--cpus=$PODSLAM_CPU_QUOTA} -e OPENCV_CPU_DISABLE="${OPENCV_CPU_DISABLE:-}" -e PODSLAM_SKIP_BUILD="${PODSLAM_SKIP_BUILD:-}" -e PODSLAM_DW="${PODSLAM_DW:-}" -e PODSLAM_DENSE="${PODSLAM_DENSE:-}" -v "$PWD:/work" -v /tmp/claude-1001:/tmp/claude-1001 -w /work fisheye/gtsam-dev bash -c '
  set -e
  [ -n "$PODSLAM_SKIP_BUILD" ] && [ -x podslam-cpp/build/test_tracker ] || \
  g++ -O2 -std=c++17 -Wall -Ipodslam-cpp/include \
      -Ipodslam-cpp/opencv-install/include/opencv5 -Ipodslam-cpp/gtsam-install/include \
      -I/usr/include/eigen3 podslam-cpp/src/test_tracker.cpp \
      -Lpodslam-cpp/opencv-install/lib -Lpodslam-cpp/gtsam-install/lib \
      -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_geometry -lopencv_features -lopencv_flann \
      -lgtsam -ltbb \
      -Wl,-rpath,/work/podslam-cpp/opencv-install/lib -Wl,-rpath,/work/podslam-cpp/gtsam-install/lib \
      -o podslam-cpp/build/test_tracker
  LD_LIBRARY_PATH=/work/podslam-cpp/opencv-install/lib:/work/podslam-cpp/gtsam-install/lib \
      nice -n 10 podslam-cpp/build/test_tracker "$@"' -- "$@"
