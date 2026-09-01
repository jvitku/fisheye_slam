# Isaac Sim 4.5 via pip (proven path from the laptop campaign: no NGC login,
# OMNI_KIT_ACCEPT_EULA=YES). RTX renderer lane for the realism claim (rule 9);
# sim/tooling only — nothing links into the product (Isaac EULA).
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive OMNI_KIT_ACCEPT_EULA=YES
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv libglu1-mesa libxrandr2 libxinerama1 \
        libxcursor1 libxi6 libgl1 libvulkan1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir "isaacsim[all]==4.5.0" --extra-index-url https://pypi.nvidia.com && \
    pip3 install --no-cache-dir numpy pillow pyyaml
WORKDIR /repo
CMD ["bash"]
