"""Image conditioning + masks in front of the front-end (the ML-friendly seam).

A conditioner is `f(img_uint8) -> img_uint8`; a mask provider is
`f(img_uint8, static_mask) -> mask_uint8` with 255 = VALID (podslam convention;
the opposite of cuVSLAM's). Both are built from config strings so a learned
model is one token away: `--preprocess enhance:/models/enhancer.pt`.
"""
from __future__ import annotations

import numpy as np


def build_conditioner(spec: str | None):
    """'none' | 'norm[:mu[:sd]]' | 'clahe[:clip[:tiles]]' | 'gamma:<g>' | 'nlmeans:<h>' |
    'enhance:<torchscript.pt>' chained with '+'."""
    if not spec or spec == "none":
        return lambda img: img
    import cv2
    steps = []
    for part in spec.split("+"):
        name, *p = part.split(":")
        if name == "norm":
            mu = float(p[0]) if p else 90.0
            sd = float(p[1]) if len(p) > 1 else 45.0
            def norm(img, mu=mu, sd=sd):
                x = img.astype(np.float32)
                s = max(float(x.std()), 1.0)
                return np.clip((x - float(x.mean())) * (sd / s) + mu, 0, 255).astype(np.uint8)
            steps.append(norm)
        elif name == "clahe":
            clahe = cv2.createCLAHE(clipLimit=float(p[0]) if p else 3.0,
                                    tileGridSize=(int(p[1]) if len(p) > 1 else 8,) * 2)
            steps.append(clahe.apply)
        elif name == "gamma":
            g = float(p[0]) if p else 0.5
            lut = (np.clip((np.arange(256) / 255.0) ** g, 0, 1) * 255).astype(np.uint8)
            steps.append(lambda img, lut=lut: lut[img])
        elif name == "nlmeans":
            hh = float(p[0]) if p else 10.0
            steps.append(lambda img, hh=hh: cv2.fastNlMeansDenoising(img, None, hh, 7, 21))
        elif name == "enhance":
            import torch
            model = torch.jit.load(p[0]).eval()
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(dev)
            def run(img, model=model, dev=dev):
                with torch.no_grad():
                    x = torch.from_numpy(img).float().to(dev)[None, None] / 255.0
                    return (model(x).clamp(0, 1)[0, 0] * 255.0).round().byte().cpu().numpy()
            steps.append(run)
        else:
            raise ValueError(f"unknown conditioning step {part!r}")

    def chain(img):
        for f in steps:
            img = f(img)
        return np.ascontiguousarray(img)
    return chain


def build_mask_provider(spec: str | None):
    """'none' | 'sat[:thr[:dilate]]' | 'learned:<torchscript.pt>' -> f(img, static) -> mask (255 = valid)."""
    if not spec or spec == "none":
        return lambda img, static: static
    import cv2
    parts = spec.split("+")
    fns = []
    for part in parts:
        name, *p = part.split(":")
        if name == "sat":
            thr = int(p[0]) if p else 250
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(p[1]) if len(p) > 1 else 9,) * 2)
            fns.append(lambda img, thr=thr, k=k: 255 - cv2.dilate((img > thr).astype(np.uint8) * 255, k))
        elif name == "learned":
            import torch
            model = torch.jit.load(p[0]).eval()
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(dev)
            def run(img, model=model, dev=dev):
                with torch.no_grad():
                    x = torch.from_numpy(img).float().to(dev)[None, None] / 255.0
                    return ((model(x)[0, 0] > 0.5).byte() * 255).cpu().numpy()
            fns.append(run)
        else:
            raise ValueError(f"unknown mask step {part!r}")

    def provide(img, static):
        m = static
        for f in fns:
            d = f(img)
            m = d if m is None else np.minimum(m, d)
        return m
    return provide
