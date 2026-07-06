# Isaac Sim + Pegasus Simulator Docker Setup

> **Provenance:** copied from `swarm_stack/tools/isaac` at commit
> `2a5de34a4e92cd2aa7f0317be8dfd7341d7b09cc` (2026-07-06) into the `3d_fisheye`
> project. 3d_fisheye additions: `workspace/bench_drone.py` (default
> `SIM_SCRIPT`, N-fisheye benchmark rig from `RIG_CONFIG`, lidar removed),
> `workspace/fisheye_rig.py` (rig yaml -> Pegasus cameras + f-theta projection
> override), compose mounts `../../rigs:/rigs:ro`. The original `px4_drone.py`
> is kept unmodified as reference (`SIM_SCRIPT=px4_drone.py`). Upstream fixes
> in swarm_stack should be ported here manually. Sections below describing the
> lidar/FAST-LIO pipeline apply to the original, not to `bench_drone.py`.

## Goal

MRS stack drone simulation using Isaac Sim + Pegasus Simulator in Docker with ROS2,
connected to MRS (ROS1/Noetic) via ROS1-ROS2 bridge. In 3d_fisheye this hosts
the **unified multi-fisheye benchmark** (see `bench/README.md`): identical
world + trajectory, camera rigs of 2/3/6 fisheyes defined in `rigs/*.yaml`.

## Architecture

```
Host (Ubuntu, NVIDIA GPU)
  |
  +-- Docker Container (isaac-pegasus)
  |     +-- Isaac Sim 5.1.0
  |     +-- Pegasus Simulator v5.1.0
  |     +-- ROS2 Humble (Isaac Sim internal libs)
  |     +-- PX4MavlinkBackend + ROS2Backend
  |     +-- Sensors: 2x camera, lidar
  |     +-- X11 forwarding for GUI
  |
  +-- Docker Container (ros1-bridge)
  |     +-- ROS1 Noetic + ROS2 Humble + ros1_bridge
  |     +-- Bridges ROS2 topics -> ROS1
  |
  +-- Host MRS Stack (ROS1 Noetic)
        +-- Subscribes to /uav1/ground_truth, cameras, lidar
```

## Versions

| Component          | Version       |
|--------------------|---------------|
| Isaac Sim          | 5.1.0         |
| Pegasus Simulator  | v5.1.0        |
| ROS2               | Humble        |
| ROS1 (host)        | Noetic        |
| Display            | X11 forwarding|

## Host Prerequisites

- NVIDIA GPU with 24GB+ VRAM (RTX 3090 or better)
- NVIDIA Driver **580.65.06+** (required for Isaac Sim 5.1.0)
- Docker 28+ with Compose v2
- NVIDIA Container Toolkit

## Usage

```bash
# Build the image (NGC login may be required first)
docker compose build

# Verify GPU access
docker compose run --rm isaac-sim nvidia-smi

# Launch sim + bridge (X11 GUI, foreground)
./start_all.sh

# Launch in background
./start_all.sh -d

# Headless mode (WebRTC streaming)
HEADLESS=true ./start_all.sh

# Vanilla Isaac Sim GUI (no Pegasus)
INTERACTIVE=true docker compose run --rm isaac-sim

# Stop everything
docker compose down

# View logs
docker compose logs -f              # all services
docker compose logs -f isaac-sim    # sim only
docker compose logs -f ros1-bridge  # bridge only
```

## NGC Authentication

If the build fails pulling the base image, authenticate with NGC:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>
```

## Workflow Rules

- **Always build after Dockerfile changes:** When modifying the Dockerfile or docker-compose.yml, always run `docker compose build` to verify the image builds without errors before considering the change done.

## Notes

- **Pegasus source is ONLY inside the Docker container** at `/opt/pegasus/` — it is NOT on the host filesystem. Do not search for Pegasus source code on the host. The host `workspace/` directory is mounted at `/workspace/` inside the container, but Pegasus itself is installed only in the image. To inspect Pegasus code, run commands inside the container (e.g., `docker compose run --rm isaac-sim cat /opt/pegasus/...`).
- **Lidar uses wall-clock timestamps**: `WallClockROS2Backend` in `px4_drone.py` overrides the default lidar writer to use `RtxLidarROS2SystemTimePublishPointCloud` (wall-clock) instead of `RtxLidarROS2PublishPointCloud` (sim-time). This ensures lidar and IMU timestamps share the same clock, which FAST-LIO requires for sync.
- **ros1_bridge auto-reconnects on roscore restart** — the watchdog loop in `bridge_entrypoint.sh` detects when roscore goes away, kills the stale bridge, waits for roscore to return, and restarts the bridge automatically. No manual intervention needed.
- **ros1_bridge 2to1 health check** — after `restart_sim.sh`, stale DDS endpoints can prevent the bridge from creating ROS2→ROS1 (2to1) bridges. The watchdog checks for "created 2to1 bridge" in the log after 30s; if none found, it restarts the bridge process (up to 3 retries). A fresh bridge process gets a new DDS participant that discovers Isaac's recreated publishers.
- **Isaac Sim 5.1.0 lidar configs use USDA assets** in `SUPPORTED_LIDAR_CONFIGS` (not raw JSON). Custom configs require USDA conversion via `/isaac-sim/tools/isaacsim.sensors.rtx/convert_lidar_json_to_usda.py` (needs Isaac runtime). `Example_Rotary` works out of the box.
- **RTX lidar outputs in FLU convention** (X-forward, Y-left, Z-up) in the sensor local frame. No coordinate transform is applied between sensor output and the ROS2 PointCloud2 message.
- **Frame conventions: Pegasus IMU = FRD/NED (for PX4), FAST-LIO expects FLU.** The body-center Pegasus IMU applies `rot_FLU_to_FRD` and `rot_ENU_to_NED`. Custom sensors for FAST-LIO must NOT apply this conversion — output in sensor-local FLU directly.
- **Custom Pegasus sensors**: Subclass `Sensor`, set a unique `sensor_type` string (e.g. `"LivoxIMU"`), use `@Sensor.update_at_rate` decorator. Override `update_sensor()` in the `ROS2Backend` subclass to route by `sensor_type`. Sensor is added to `MultirotorConfig.sensors` list (default: `[Barometer, IMU, Magnetometer, GPS]`).
- **LivoxIMU sensor** (`workspace/livox_imu.py`): Simulates IMU co-located and co-oriented with the LiDAR (matching real Livox Mid-360). Takes `position` and `orientation` config matching the Lidar mount. Computes rigid-body offset acceleration (`a_body + α×r + ω×(ω×r)`), rotates to sensor-local FLU frame, outputs without FRD/NED conversion.

## Debugging

Use the `/ros-debug` skill for debugging. Key commands:

```bash
# Capture tmux pane output from running MRS session
.claude/skills/ros-debug/scripts/find-and-capture-tmux.sh slam 0 300

# Read ROS logs from host filesystem
.claude/skills/ros-debug/scripts/read-ros-logs.sh errors
.claude/skills/ros-debug/scripts/read-ros-logs.sh tmux-logs "No point"

# Check Isaac Sim log
docker compose logs --tail 50 isaac-sim

# Quick topic checks via run_ros.sh
./run_ros.sh bash -c "rostopic echo -n 1 /uav1/livox_mid360/pointcloud/header 2>&1 | head -5"
./run_ros.sh bash -c "timeout 3 rostopic hz /uav1/livox/lidar 2>&1"
```

### Full data pipeline check

Run from swarm_stack root to diagnose where data stops flowing:

```bash
# 1. Bridge delivering data?
./run_ros.sh bash -c "timeout 3 rostopic hz /uav1/livox_mid360/pointcloud 2>&1"
./run_ros.sh bash -c "timeout 3 rostopic hz /uav1/sensors/imu 2>&1"
# 2. Relays forwarding?
./run_ros.sh bash -c "timeout 3 rostopic hz /uav1/livox/lidar 2>&1"
./run_ros.sh bash -c "timeout 3 rostopic hz /uav1/livox/imu 2>&1"
# 3. FAST-LIO producing?
./run_ros.sh bash -c "timeout 3 rostopic hz /uav1/fast_lio/odom 2>&1"
# 4. Timestamps aligned?
./run_ros.sh bash -c "rostopic echo -n 1 /uav1/livox/lidar/header 2>&1"
./run_ros.sh bash -c "rostopic echo -n 1 /uav1/livox/imu/header 2>&1"
```

### Starting the MRS tmux session

From inside the deploy container (or via run_ros.sh):
```bash
cd /app/catkin_ws/src/goodai_common/simulation/isaac_fastlio && tmux_start.sh
```
`tmux_start.sh` (in `/app/bin/`) merges session.yml imports and starts tmuxinator.
Do NOT use per-session `start.sh` files (deprecated).

## Phases

1. **Phase 1 (done):** Isaac Sim + Pegasus Simulator + PX4 v1.15.2 in Docker with X11
2. **Phase 2 (done):** ROS2 Humble (Isaac Sim internal) + `ROS2Backend` + cameras + lidar
3. **Phase 3 (done):** ros1_bridge container connecting ROS2 to host MRS/Noetic
4. **Phase 4 (done):** FAST-LIO odometry working — `lidar_type: 4`, `sim_handler` with `point_filter_num`, `Example_Rotary` config
