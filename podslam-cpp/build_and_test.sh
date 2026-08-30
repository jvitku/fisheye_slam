#!/usr/bin/env bash
# Step-1 build: no dependencies beyond a C++17 compiler.  Golden data comes from
# the Python reference (see docs/CPP_PORT_PLAN.md).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p podslam-cpp/build
g++ -O2 -std=c++17 -Wall -Ipodslam-cpp/include podslam-cpp/tests/test_kb4.cpp -o podslam-cpp/build/test_kb4
g++ -O2 -std=c++17 -Wall -Ipodslam-cpp/include podslam-cpp/tests/test_ds_eucm.cpp -o podslam-cpp/build/test_ds_eucm
g++ -O2 -std=c++17 -Wall -Ipodslam-cpp/include podslam-cpp/tests/test_stereo.cpp -o podslam-cpp/build/test_stereo
podslam-cpp/build/test_stereo
podslam-cpp/build/test_ds_eucm
g++ -O2 -std=c++17 -Wall -Ipodslam-cpp/include podslam-cpp/tests/test_stereo.cpp -o podslam-cpp/build/test_stereo
podslam-cpp/build/test_stereo
podslam-cpp/build/test_kb4
g++ -O2 -std=c++17 -Wall -Ipodslam-cpp/include podslam-cpp/tests/test_ds_eucm.cpp -o podslam-cpp/build/test_ds_eucm
g++ -O2 -std=c++17 -Wall -Ipodslam-cpp/include podslam-cpp/tests/test_stereo.cpp -o podslam-cpp/build/test_stereo
podslam-cpp/build/test_stereo
podslam-cpp/build/test_ds_eucm
g++ -O2 -std=c++17 -Wall -Ipodslam-cpp/include podslam-cpp/tests/test_stereo.cpp -o podslam-cpp/build/test_stereo
podslam-cpp/build/test_stereo
