# PX4 SITL (SIH quadrotor, lockstep) for flying minisim trajectories —
# bench/minisim/px4_fly.py runs INSIDE this container with the repo mounted:
# it spawns `make px4_sitl sihsim_quadx` (pre-built here, so runs start fast),
# flies the mission over mavsdk on local UDP, and extracts the ulog to TUM.
#
#   docker build -t fisheye/px4-sitl docker/px4-sitl/
#   docker run --rm -v $REPO:$REPO -w $REPO -e HOME=/tmp fisheye/px4-sitl \
#       python3 -m bench.minisim.px4_fly indoor trajs/foo.tum --px4 /px4
#
# PX4 is BSD (sim/tooling lane — nothing links into the product). The clone is
# shallow AT the release tag so `git describe` works (shallow clones without
# the tag break the PX4 build version stamp).
FROM ubuntu:22.04
ARG PX4_TAG=v1.15.4
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates cmake ninja-build make gcc g++ python3 python3-pip \
        python3-dev pkg-config libxml2-utils rsync file && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir kconfiglib jinja2 empy==3.3.4 pyros-genmsg packaging toml \
        numpy "scipy>=1.10" pyulog mavsdk jsonschema future
RUN git clone --depth 1 --branch ${PX4_TAG} --recurse-submodules --shallow-submodules \
        https://github.com/PX4/PX4-Autopilot.git /px4
# build only (`make px4_sitl sihsim_quadx` would LAUNCH the sim; px4_fly runs
# that at container runtime against this prebuilt tree)
RUN cd /px4 && make px4_sitl_default -j"$(nproc)" && \
    rm -rf /px4/build/px4_sitl_default/rootfs/log
# log dir must be writable by the mounted-uid user; keep the whole tree open
RUN chmod -R a+rwX /px4
WORKDIR /repo
CMD ["bash"]
