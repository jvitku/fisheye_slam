"""Tests for bench.degrade_bag and bench.euroc2bag."""

import csv

import numpy as np
import pytest

pytest.importorskip("rosbags")
pytest.importorskip("PIL")

from PIL import Image as PILImage
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from bench.degrade_bag import beam_vignette, degrade_bag, night_factor
from bench.euroc2bag import convert as euroc_convert

TS = get_typestore(Stores.ROS1_NOETIC)
NS = 1_000_000_000


@pytest.fixture
def euroc_dir(tmp_path):
    """Minimal EuRoC layout: 2 cams x 10 frames + imu csv."""
    rng = np.random.default_rng(0)
    t0 = 1000 * NS
    for cam in ("cam0", "cam1"):
        d = tmp_path / "mav0" / cam / "data"
        d.mkdir(parents=True)
        for i in range(10):
            img = rng.integers(60, 200, (64, 64), dtype=np.uint8)
            PILImage.fromarray(img, "L").save(d / f"{t0 + i * NS // 10}.png")
    imu_dir = tmp_path / "mav0" / "imu0"
    imu_dir.mkdir(parents=True)
    with open(imu_dir / "data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#timestamp [ns]", "wx", "wy", "wz", "ax", "ay", "az"])
        for i in range(100):
            w.writerow([t0 + i * NS // 100, 0.01, 0.0, 0.0, 0.0, 0.0, 9.81])
    return tmp_path


def read_images(path, topic):
    out = []
    with Reader(path) as r:
        for conn, ts_, raw in r.messages():
            if conn.topic == topic:
                msg = TS.deserialize_ros1(raw, conn.msgtype)
                out.append((ts_, np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width)))
    return out


def test_euroc2bag_counts_and_content(euroc_dir, tmp_path):
    bag = str(tmp_path / "day.bag")
    counts = euroc_convert(euroc_dir, bag, "/cam{i}/image_raw", "/imu0")
    assert counts == {"cam0": 10, "cam1": 10, "imu": 100}
    imgs = read_images(bag, "/cam0/image_raw")
    assert len(imgs) == 10 and imgs[0][1].shape == (64, 64)
    # timestamps strictly increasing per topic
    ts = [t for t, _ in imgs]
    assert ts == sorted(ts)


def test_euroc2bag_16bit_png_linear_scaling(tmp_path):
    """TUM-VI 16-bit PNGs must map linearly (>>8), not clip to white."""
    d = tmp_path / "mav0" / "cam0" / "data"
    d.mkdir(parents=True)
    grad = (np.linspace(0, 65535, 64 * 64, dtype=np.uint32)
            .reshape(64, 64).astype(np.uint16))
    PILImage.fromarray(grad, "I;16").save(d / f"{1000 * NS}.png")
    imu = tmp_path / "mav0" / "imu0"
    imu.mkdir(parents=True)
    (imu / "data.csv").write_text(f"{1000 * NS},0,0,0,0,0,9.81\n")

    bag = str(tmp_path / "out.bag")
    euroc_convert(tmp_path, bag, "/cam{i}/image_raw", "/imu0")
    (_, img), = read_images(bag, "/cam0/image_raw")
    assert img[0, 0] == 0 and img[-1, -1] == 255
    assert 120 < img.astype(float).mean() < 135  # linear ramp, not clipped


def test_night_darkens_and_keeps_beam_center(euroc_dir, tmp_path):
    day = str(tmp_path / "day.bag")
    night = str(tmp_path / "night.bag")
    euroc_convert(euroc_dir, day, "/cam{i}/image_raw", "/imu0")
    degrade_bag(day, night, None, "night", ambient=0.04, beam=0.7,
                beam_sigma=0.45, noise_sigma=6.0)

    day_imgs = dict(read_images(day, "/cam0/image_raw"))
    night_imgs = dict(read_images(night, "/cam0/image_raw"))
    assert day_imgs.keys() == night_imgs.keys()
    for t in day_imgs:
        d, n = day_imgs[t].astype(float), night_imgs[t].astype(float)
        assert n.mean() < 0.5 * d.mean()  # globally much darker
        h, w = n.shape
        center = n[h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8].mean()
        corner = n[:8, :8].mean()
        assert center > corner + 10  # beam vignette: center lit, corners dark


def test_night_is_deterministic(euroc_dir, tmp_path):
    day = str(tmp_path / "day.bag")
    euroc_convert(euroc_dir, day, "/cam{i}/image_raw", "/imu0")
    a, b = str(tmp_path / "a.bag"), str(tmp_path / "b.bag")
    degrade_bag(day, a, None, "night")
    degrade_bag(day, b, None, "night")
    ia, ib = read_images(a, "/cam1/image_raw"), read_images(b, "/cam1/image_raw")
    for (ta, xa), (tb, xb) in zip(ia, ib):
        assert ta == tb
        np.testing.assert_array_equal(xa, xb)


def test_transition_day_start_night_end(euroc_dir, tmp_path):
    day = str(tmp_path / "day.bag")
    trans = str(tmp_path / "trans.bag")
    euroc_convert(euroc_dir, day, "/cam{i}/image_raw", "/imu0")
    degrade_bag(day, trans, None, "transition")
    d = read_images(day, "/cam0/image_raw")
    t = read_images(trans, "/cam0/image_raw")
    np.testing.assert_array_equal(d[0][1], t[0][1])        # start: untouched day
    assert t[-1][1].astype(float).mean() < 0.5 * d[-1][1].astype(float).mean()  # end: night


def test_imu_untouched_by_degradation(euroc_dir, tmp_path):
    day, night = str(tmp_path / "day.bag"), str(tmp_path / "night.bag")
    euroc_convert(euroc_dir, day, "/cam{i}/image_raw", "/imu0")
    degrade_bag(day, night, None, "night")

    def imu_raws(path):
        with Reader(path) as r:
            return [(t, bytes(raw)) for c, t, raw in r.messages() if c.topic == "/imu0"]

    assert imu_raws(day) == imu_raws(night)


def test_night_factor_profile():
    assert night_factor("night", 0.0) == 1.0
    assert night_factor("transition", 0.0) == 0.0
    assert night_factor("transition", 0.5) == pytest.approx(0.5)
    assert night_factor("transition", 1.0) == 1.0


def test_vignette_shape():
    v = beam_vignette(64, 64, 0.45)
    assert v.max() == pytest.approx(1.0, abs=1e-3)
    assert v[32, 32] > v[0, 0] * 2
