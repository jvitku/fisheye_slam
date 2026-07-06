#!/usr/bin/env bash
# Shallow-clone all candidate systems into candidates/.
# Clones are throwaway working copies, not submodules (they are .gitignored);
# pin exact commits in docker/ builds once a track graduates from exploration.
set -euo pipefail
cd "$(dirname "$0")"

clone() {
    local url="$1" dir="$2"
    if [ -d "$dir/.git" ]; then
        echo "== $dir already cloned, skipping"
    else
        echo "== cloning $url -> $dir"
        git clone --depth 1 --recursive "$url" "$dir"
    fi
}

# Track A — primary VIO candidate
clone https://github.com/rpng/open_vins open_vins
# Track B — calibration backbone + stereo baseline
clone https://gitlab.com/VladyslavUsenko/basalt.git basalt
# Track C — multi-camera SLAM
clone https://github.com/MAVIS-SLAM/OpenMAVIS OpenMAVIS
# Track D — learned components
clone https://github.com/verlab/accelerated_features accelerated_features
clone https://github.com/cvg/LightGlue LightGlue
clone https://github.com/GREAT-WHU/DBA-Fusion DBA-Fusion
# Track F — hybrid SLAM exploiting 3-cam+IMU on Orin (see experiments/07)
clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam isaac_ros_visual_slam
clone https://github.com/sair-lab/AirSLAM AirSLAM
clone https://github.com/MIT-SPARK/Kimera-VIO Kimera-VIO
# Deferred references (NOT in the comparison matrix):
#   MASt3R-SLAM — CC BY-NC weights + monocular (removed 2026-07-06)
#   DPVO — MIT but monocular; doesn't exploit the 3-cam+IMU rig
clone https://github.com/rmurai0610/MASt3R-SLAM MASt3R-SLAM
clone https://github.com/princeton-vl/DPVO DPVO
# Fallback (Jetson stereo, kept for reference only)
clone https://github.com/HKUST-Aerial-Robotics/VINS-Fisheye VINS-Fisheye

echo
echo "Done. Sizes:"
du -sh -- */ 2>/dev/null | sort -h
