"""Learned front-end: XFeat (Apache-2.0, verlab/accelerated_features) detector +
descriptor with mutual-nearest-neighbour matching, temporal and stereo, on the
GPU. Same contract as the KLT front-end, so the tracker cannot tell them apart —
this is the "ML extension" seam exercised end to end. Weights: the candidate
clone's weights/xfeat.pt (or XFEAT_REPO env)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .base import CamObs, FrameFeatures, Frontend
from .common import essential_inliers, stereo_verify

DEFAULTS = dict(top_k=1024, min_cossim=0.82, max_features=400, stereo_max_px=3.0, rim_ang_deg=6.0)


class XFeatFrontend(Frontend):
    def __init__(self, rig, config=None):
        super().__init__(rig, config)
        self.cfg = {**DEFAULTS, **(config or {})}
        import torch
        repo = Path(os.environ.get("XFEAT_REPO", Path(__file__).resolve().parents[2] / "candidates" / "accelerated_features"))
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from modules.xfeat import XFeat
        weights = torch.load(str(repo / "weights" / "xfeat.pt"), map_location="cpu")
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = XFeat(weights=weights, top_k=self.cfg["top_k"]).to(self.dev).eval()
        self.torch = torch
        self.reset()

    def reset(self):
        self.prev = None        # (ids, kpts np (N,2), desc torch (N,64), bearings)
        self.next_id = 0

    def _extract(self, img, mask):
        t = self.torch.from_numpy(img).float().to(self.dev)[None, None] / 255.0
        with self.torch.no_grad():
            out = self.net.detectAndCompute(t, top_k=self.cfg["top_k"])[0]
        k = out["keypoints"].cpu().numpy().astype(np.float32)
        d = out["descriptors"]
        if mask is not None and len(k):
            xi = np.clip(k[:, 0].astype(int), 0, mask.shape[1] - 1); yi = np.clip(k[:, 1].astype(int), 0, mask.shape[0] - 1)
            keep = mask[yi, xi] > 0
            k, d = k[keep], d[self.torch.from_numpy(keep).to(d.device)]
        return k, d

    def _mnn(self, d1, d2):
        if len(d1) == 0 or len(d2) == 0:
            return np.zeros(0, int), np.zeros(0, int)
        cos = d1 @ d2.t()
        m12 = cos.argmax(1); m21 = cos.argmax(0)
        i = self.torch.arange(len(m12), device=cos.device)
        mutual = m21[m12] == i
        good = mutual & (cos[i, m12] > self.cfg["min_cossim"])
        return i[good].cpu().numpy(), m12[good].cpu().numpy()

    def process(self, t_ns, images, masks, dR_imu):
        cam0 = self.rig.cameras[0]
        k0, d0 = self._extract(images[0], masks[0] if masks else None)
        b0, v0 = cam0.model.unproject(k0.astype(np.float64))
        k0, d0, b0 = k0[v0], d0[self.torch.from_numpy(v0).to(d0.device)], b0[v0]
        ids = np.full(len(k0), -1, np.int64)
        n_new = 0
        if self.prev is not None and len(k0):
            pids, pk, pd, pb = self.prev
            a, b = self._mnn(pd, d0)
            if len(a) >= 8:
                R_ic = cam0.T_imu_cam[:3, :3]
                dR_cam = None if dR_imu is None else R_ic.T @ dR_imu @ R_ic
                keep = essential_inliers(pb[a], b0[b], dR_cam, ang_thr_deg=self.cfg["rim_ang_deg"])
                a, b = a[keep], b[keep]
            ids[b] = pids[a]
        fresh = ids < 0
        n_fresh = min(int(fresh.sum()), max(0, self.cfg["max_features"] - int((~fresh).sum())))
        # keep the strongest fresh detections (XFeat returns them score-sorted)
        fresh_idx = np.nonzero(fresh)[0][:n_fresh]
        drop = np.nonzero(fresh)[0][n_fresh:]
        ids[fresh_idx] = np.arange(self.next_id, self.next_id + n_fresh); self.next_id += n_fresh; n_new = n_fresh
        keep = np.ones(len(ids), bool); keep[drop] = False
        k0, b0, ids = k0[keep], b0[keep], ids[keep]
        d0 = d0[self.torch.from_numpy(keep).to(d0.device)]
        cams = [CamObs(ids=ids, px=k0, bearings=b0)]
        for j in range(1, len(self.rig.cameras)):
            cams.append(self._stereo(j, images[j], masks[j] if masks else None, k0, d0, b0, ids))
        self.prev = (ids, k0, d0, b0)
        return FrameFeatures(t_ns=t_ns, cams=cams, n_new=n_new)

    def _stereo(self, j, img_j, mask_j, k0, d0, b0, ids):
        cam_j = self.rig.cameras[j]
        kj, dj = self._extract(img_j, mask_j)
        bj, vj = cam_j.model.unproject(kj.astype(np.float64))
        kj, dj, bj = kj[vj], dj[self.torch.from_numpy(vj).to(dj.device)], bj[vj]
        a, b = self._mnn(d0, dj)
        out_ids, out_px, out_b = [], [], []
        for i, m in zip(a, b):
            p, _ = stereo_verify(self.rig, 0, j, b0[i], bj[m], k0[i], kj[m], max_px=self.cfg["stereo_max_px"])
            if p is not None:
                out_ids.append(ids[i]); out_px.append(kj[m]); out_b.append(bj[m])
        if not out_ids:
            return CamObs()
        return CamObs(ids=np.asarray(out_ids, np.int64), px=np.asarray(out_px, np.float32), bearings=np.asarray(out_b))
