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
# Fallback (Jetson stereo, kept for reference only)
clone https://github.com/HKUST-Aerial-Robotics/VINS-Fisheye VINS-Fisheye

echo
echo "Done. Sizes:"
du -sh -- */ 2>/dev/null | sort -h
