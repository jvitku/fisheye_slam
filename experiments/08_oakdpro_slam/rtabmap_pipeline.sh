#!/bin/bash
# In-container (3dfe/rtabmap) pipeline: bag -> rtabmap.db + octomap.bt +
# cloud.ply + est.tum. Invoked by run_rtabmap.sh — not directly.
# Args: <bag> <out_dir> <odom: rtabmap|external>
set -euo pipefail
source /opt/ros/noetic/setup.bash
BAG="$1"; OUT="$2"; ODOM="${3:-rtabmap}"
# Every background node (roscore, camera_info publisher, odometry, rtabmap)
# inherits this shell's stdout; unless they are killed the container's output
# pipe never closes and run_rtabmap.sh hangs after "done".
trap 'kill $(jobs -p) 2>/dev/null; sleep 1; kill -9 $(jobs -p) 2>/dev/null; true' EXIT


roscore &
sleep 3
rosparam set use_sim_time true

# CameraInfo is not in the bag contract — synthesize it from the rig yaml
# (sim GT calibration) or pass CALIB_YAML for HW bags.
python3 /scripts/publish_camera_info.py \
    --rig "${CALIB_YAML:-/rigs/oakdpro.yaml}" ${CALIB_YAML:+--calib} &
sleep 1

RGBD_REMAPS=(rgb/image:=/uav1/cam0/color/image_raw
             depth/image:=/uav1/cam0/depth/image_raw
             rgb/camera_info:=/uav1/cam0/camera_info)

if [ "$ODOM" = "rtabmap" ]; then
    rosrun rtabmap_odom rgbd_odometry "${RGBD_REMAPS[@]}" \
        _frame_id:=cam0 _approx_sync:=true _queue_size:=30 \
        _wait_imu_to_init:=false &
    ODOM_TOPIC=/odom
else
    # External odometry (open-source lane headline: OpenVINS). Start the
    # 3dfe/openvins live container on the host network publishing
    # nav_msgs/Odometry; remap its topic to /uav1/odom (see README).
    ODOM_TOPIC=/uav1/odom
fi

rosrun rtabmap_slam rtabmap --delete_db_on_start \
    "${RGBD_REMAPS[@]}" odom:="$ODOM_TOPIC" \
    _frame_id:=cam0 _subscribe_depth:=true _approx_sync:=true _queue_size:=30 \
    _database_path:="$OUT/rtabmap.db" \
    --Rtabmap/DetectionRate 2 --Grid/FromDepth true --Grid/3D true \
    --RGBD/CreateOccupancyGrid true &
RTABMAP_PID=$!
sleep 3

rosbag play --clock -d 2 "$BAG"
sleep 5

# OctoMap out: rtabmap (built with octomap) serves /rtabmap/octomap_binary.
# octomap_saver <file.bt> uses the octomap_binary service (its -f flag means
# FULL map via /octomap_full, not "file"). Bounded: never let an export hang.
timeout 60 rosrun octomap_server octomap_saver "$OUT/octomap.bt" \
    octomap_binary:=/rtabmap/octomap_binary || echo "WARN: octomap export failed"

kill -INT "$RTABMAP_PID"; wait "$RTABMAP_PID" || true

# Dense cloud + poses from the database. poses_format 1 = RGBD-SLAM/TUM
# (timestamp x y z qx qy qz qw). VERIFY-ON-FIRST-RUN: rtabmap-export flags.
timeout 300 rtabmap-export --cloud --poses --poses_format 1 \
    --output_dir "$OUT" "$OUT/rtabmap.db" || echo "WARN: rtabmap-export failed"
[ -f "$OUT/rtabmap_poses.txt" ] && mv "$OUT/rtabmap_poses.txt" "$OUT/est.tum"

echo "done: $OUT"
