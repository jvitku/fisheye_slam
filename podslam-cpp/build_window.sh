#!/usr/bin/env bash
# Compile + run the window parity test against the mounted GTSAM install, inside the
# fisheye/gtsam-dev toolchain image (host has no boost/eigen).
set -euo pipefail
cd "$(dirname "$0")/.."
docker run --rm ${GUARD_DOCKER_ARGS:-} -v "$PWD:/work" -w /work fisheye/gtsam-dev bash -c '
  set -e
  g++ -O2 -std=c++17 -Wall -Ipodslam-cpp/include -Ipodslam-cpp/gtsam-install/include \
      -I/usr/include/eigen3 podslam-cpp/src/test_window.cpp \
      -Lpodslam-cpp/gtsam-install/lib -lgtsam -ltbb \
      -Wl,-rpath,/work/podslam-cpp/gtsam-install/lib -o podslam-cpp/build/test_window
  LD_LIBRARY_PATH=/work/podslam-cpp/gtsam-install/lib podslam-cpp/build/test_window "$@"' -- "$@"
