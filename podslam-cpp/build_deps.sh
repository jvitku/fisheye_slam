#!/usr/bin/env bash
# Build podslam-cpp's pinned dependencies from source into
#   podslam-cpp/gtsam-install    (GTSAM 4.3a2 — the tag the pypi wheel tracks)
#   podslam-cpp/opencv-install   (OpenCV 5.0.0 + IPP — the cv2 wheel is 5.0.0/IPP;
#                                 KLT parity is sensitive to the IPP code paths)
# Everything runs in the fisheye/gtsam-dev toolchain container (host stays clean);
# sources are cloned into podslam-cpp/deps-src (kept for incremental rebuilds,
# delete when disk is tight). ~40 min on 16 threads. Run under bench/guard.sh.
#
#   bash bench/guard.sh --cpus 16 --mem 20G --timeout 2h -- podslam-cpp/build_deps.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p podslam-cpp/deps-src podslam-cpp/gtsam-install podslam-cpp/opencv-install

docker run --rm ${GUARD_DOCKER_ARGS:-} --user "$(id -u):$(id -g)" -e HOME=/tmp -e DEPS_J \
    -v "$PWD:/work" -w /work fisheye/gtsam-dev bash -ec '
J=${DEPS_J:-16}
S=podslam-cpp/deps-src

[ -d "$S/gtsam/.git" ] || git clone --depth 1 --branch 4.3a2 https://github.com/borglab/gtsam.git "$S/gtsam"
cmake -S "$S/gtsam" -B "$S/gtsam/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/work/podslam-cpp/gtsam-install \
    -DGTSAM_BUILD_PYTHON=OFF -DGTSAM_BUILD_TESTS=OFF \
    -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_UNSTABLE=OFF \
    -DGTSAM_WITH_TBB=ON -DGTSAM_USE_SYSTEM_EIGEN=ON \
    -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF
cmake --build "$S/gtsam/build" -j"$J" --target install

[ -d "$S/opencv/.git" ] || git clone --depth 1 --branch 5.0.0 https://github.com/opencv/opencv.git "$S/opencv"
cmake -S "$S/opencv" -B "$S/opencv/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/work/podslam-cpp/opencv-install \
    -DBUILD_LIST=core,imgproc,video,features,flann,geometry \
    -DBUILD_SHARED_LIBS=ON -DWITH_IPP=ON \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF -DWITH_CUDA=OFF -DWITH_GTK=OFF -DWITH_QT=OFF \
    -DWITH_FFMPEG=OFF -DWITH_V4L=OFF -DWITH_PROTOBUF=OFF
cmake --build "$S/opencv/build" -j"$J" --target install
echo DEPS-OK
'
