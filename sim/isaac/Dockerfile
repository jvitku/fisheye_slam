FROM nvcr.io/nvidia/isaac-sim:5.1.0

USER root

# Set Isaac Sim path for Pegasus setup.py
ENV ISAACSIM_PATH=/isaac-sim

# System dependencies for Pegasus/PX4
RUN apt-get update && apt-get install -y \
    git cmake make wget python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies for PX4/Pegasus (using Isaac Sim's bundled Python 3.11)
RUN /isaac-sim/kit/python/bin/python3 -m pip install --no-cache-dir kconfiglib

# Clone Pegasus Simulator v5.1.0
RUN git clone -b v5.1.0 https://github.com/PegasusSimulator/PegasusSimulator.git /opt/pegasus

# Install Pegasus extension using Isaac Sim's Python
RUN /isaac-sim/kit/python/bin/python3 -m pip install -e /opt/pegasus/extensions/pegasus.simulator

# PX4 build dependencies (system packages)
RUN apt-get update && apt-get install -y \
    python3-jinja2 python3-empy python3-jsonschema python3-packaging \
    python3-toml python3-numpy python3-future astyle \
    && rm -rf /var/lib/apt/lists/*

# PX4 build uses system python3, so install kconfiglib + deps there too
# Force-upgrade future (system 0.18.2 uses removed 'imp' module on Python 3.12)
RUN pip3 install --no-cache-dir --break-system-packages --force-reinstall \
    future
RUN pip3 install --no-cache-dir --break-system-packages \
    kconfiglib jinja2 empy jsonschema pyros-genmsg packaging toml numpy pyyaml

# Same deps for Isaac Sim's bundled Python (runtime needs)
RUN /isaac-sim/kit/python/bin/python3 -m pip install --no-cache-dir \
    jinja2 empy jsonschema pyros-genmsg packaging toml numpy future

# Clone and build PX4-Autopilot v1.15.2 (v1.14.3 doesn't compile with GCC 13 in this base image)
# Use cmake directly instead of `make px4_sitl_default` which launches SITL after building
RUN git clone -b v1.15.2 --recursive https://github.com/PX4/PX4-Autopilot.git /opt/PX4-Autopilot \
    && cd /opt/PX4-Autopilot \
    && cmake -B build/px4_sitl_default -DCONFIG=px4_sitl_default \
    && cmake --build build/px4_sitl_default -j$(nproc)

# PX4 MAVLink config: high-rate streams for MRS stack (replaces default px4-rc.mavlink)
# Must overwrite both ROMFS source and build dir copy (cmake copied the original during build)
COPY workspace/px4-rc.mavlink /opt/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/px4-rc.mavlink
COPY workspace/px4-rc.mavlink /opt/PX4-Autopilot/build/px4_sitl_default/etc/init.d-posix/px4-rc.mavlink

# Test dependencies (system python — for Tier 1 tests)
RUN pip3 install --no-cache-dir --break-system-packages pytest pymavlink

# Test dependencies (Isaac Sim python — for Tier 2 tests)
RUN /isaac-sim/kit/python/bin/python3 -m pip install --no-cache-dir pytest pymavlink

# --- Phase 2: ROS2 Humble (Isaac Sim's internal libs) ---
# Isaac Sim 5.1.0 bundles ROS2 Humble compiled with Python 3.11 — no separate install needed.
# Just set env vars so the internal libs are found at runtime.
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/isaac-sim/exts/isaacsim.ros2.bridge/humble/lib"
ENV ROS_DISTRO=humble
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Custom RTX Lidar config approximating Livox Mid-360 (32-channel, 59deg vertical FOV)
# Config must go in omni.sensors.nv.common extscache (where Example_Rotary.json lives)
COPY workspace/Livox_Mid360.json /isaac-sim/extscache/omni.sensors.nv.common-3.0.0+69cbf6ad.lx64.r.cp311/data/lidar/Livox_Mid360.json

# Additional Python dependency for ROS2 transforms
RUN /isaac-sim/kit/python/bin/python3 -m pip install --no-cache-dir transforms3d Pillow
