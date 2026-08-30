# podslam — the in-house, permissively licensed VIO/SLAM (Track F2 product
# lane): Python front-end (OpenCV KLT or learned XFeat) + GTSAM (BSD) factor
# graph back-end, ML extension points throughout. GPU optional (learned
# front-end / image conditioning); the classical path is CPU-only.
# Built on 3dfe/cuvslam-ml only to reuse its torch/OpenCV layers for the
# side-by-side comparison — a production image is FROM a plain CUDA/Ubuntu base.
FROM 3dfe/cuvslam-ml
RUN pip3 install --no-cache-dir "gtsam==4.3a2" scipy && \
    python3 -c "import gtsam, cv2, torch, numpy; print('gtsam ok, cv2', cv2.__version__, 'torch', torch.__version__)"
WORKDIR /scripts
CMD ["bash"]
