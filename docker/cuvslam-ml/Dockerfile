# cuVSLAM + PyTorch: image-conditioning experiments in front of the closed
# tracker (learned low-light enhancement / denoising, learned masks) and the
# XFeat front-end study, all in ONE process so a frame goes GPU-enhancer ->
# cuVSLAM without leaving the GPU. Built on 3dfe/cuvslam (docker/cuvslam);
# torch from the cu126 wheel index to match the CUDA 12.6 runtime base.
FROM 3dfe/cuvslam
ARG TORCH_VERSION=2.6.0
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
        "torch==${TORCH_VERSION}" "torchvision" && \
    pip3 install --no-cache-dir kornia scikit-image && \
    python3 -c "import torch, cuvslam; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuvslam', cuvslam.get_version() if hasattr(cuvslam,'get_version') else '')"
WORKDIR /scripts
CMD ["bash"]
