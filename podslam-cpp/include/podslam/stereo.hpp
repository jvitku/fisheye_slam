// Two-ray midpoint triangulation with reprojection verification — the C++ twin of
// podslam/frontend/common.py::stereo_verify_batch (single-pair form).
#pragma once
#include <cmath>
#include "podslam/kb4.hpp"

namespace podslam {

struct Mat4 { double m[4][4]; };   // row-major rigid transform

// bearings in the two camera optical frames; T0/T1 = T_imu_cam of each camera.
// Returns kept flag; fills point (IMU frame) and depth along ray 0.
inline bool stereo_verify(const Mat4& T0, const Mat4& T1, const Kb4& cam0, const Kb4& cam1,
                          const double b0[3], const double b1[3],
                          const double px0[2], const double px1[2],
                          double p_out[3], double& depth0,
                          double max_px = 2.0, double min_depth = 0.2, double max_depth = 40.0,
                          double min_parallax_deg = 1.0) {
    auto rot = [](const Mat4& T, const double v[3], double o[3]) {
        for (int r = 0; r < 3; ++r) o[r] = T.m[r][0] * v[0] + T.m[r][1] * v[1] + T.m[r][2] * v[2];
    };
    double d0[3], d1[3];
    rot(T0, b0, d0); rot(T1, b1, d1);
    const double t0[3] = {T0.m[0][3], T0.m[1][3], T0.m[2][3]};
    const double t1[3] = {T1.m[0][3], T1.m[1][3], T1.m[2][3]};
    double cosang = d0[0] * d1[0] + d0[1] * d1[1] + d0[2] * d1[2];
    if (cosang > 1.0) { cosang = 1.0; }
    if (cosang < -1.0) { cosang = -1.0; }
    if (std::acos(cosang) * 180.0 / M_PI < min_parallax_deg) return false;
    const double w[3] = {t0[0] - t1[0], t0[1] - t1[1], t0[2] - t1[2]};
    const double b = cosang;
    const double d = d0[0] * w[0] + d0[1] * w[1] + d0[2] * w[2];
    const double e = d1[0] * w[0] + d1[1] * w[1] + d1[2] * w[2];
    const double den = 1.0 - b * b;
    if (den <= 1e-12) return false;
    const double s0 = (b * e - d) / den;
    const double s1 = (e - b * d) / den;
    for (int r = 0; r < 3; ++r) p_out[r] = 0.5 * ((t0[r] + s0 * d0[r]) + (t1[r] + s1 * d1[r]));
    if (!(s0 >= min_depth && s1 >= min_depth && s0 <= max_depth)) return false;
    // reproject through both cameras (transform IMU -> cam = T^-1 p, T rigid)
    auto to_cam = [&rot](const Mat4& T, const double p[3], double o[3]) {
        const double q[3] = {p[0] - T.m[0][3], p[1] - T.m[1][3], p[2] - T.m[2][3]};
        for (int r = 0; r < 3; ++r) o[r] = T.m[0][r] * q[0] + T.m[1][r] * q[1] + T.m[2][r] * q[2];
    };
    double pc[3], u, v;
    to_cam(T0, p_out, pc);
    if (!cam0.project(pc[0], pc[1], pc[2], u, v)) return false;
    if (std::hypot(u - px0[0], v - px0[1]) > max_px) return false;
    to_cam(T1, p_out, pc);
    if (!cam1.project(pc[0], pc[1], pc[2], u, v)) return false;
    if (std::hypot(u - px1[0], v - px1[1]) > max_px) return false;
    depth0 = s0;
    return true;
}

}  // namespace podslam
