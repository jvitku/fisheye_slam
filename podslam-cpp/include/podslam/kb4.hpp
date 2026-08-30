// Kannala-Brandt (equidistant, OpenCV-fisheye) camera model — C++ transcription of
// tools/fisheye/models.py::KannalaBrandt4, held bit-close to the Python reference
// (golden tests in podslam-cpp/tests, data dumped from the Python model).
//
// Conventions: x right, y down, z forward (optical axis); bearings unit-norm.
#pragma once
#include <cmath>

namespace podslam {

struct Kb4 {
    double fx, fy, cx, cy;
    double k1 = 0.0, k2 = 0.0, k3 = 0.0, k4 = 0.0;
    double max_theta = 110.0 * M_PI / 180.0;   // rays past this are invalid

    double d(double theta) const {
        const double t2 = theta * theta;
        return theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))));
    }
    double d_prime(double theta) const {
        const double t2 = theta * theta;
        return 1.0 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)));
    }

    // (X,Y,Z) camera frame -> pixel; returns validity
    bool project(double X, double Y, double Z, double& u, double& v) const {
        const double r = std::hypot(X, Y);
        const double theta = std::atan2(r, Z);
        const bool valid = theta < max_theta;
        const double dd = d(theta);
        if (r > 1e-12) {
            const double scale = dd / r;
            u = fx * X * scale + cx;
            v = fy * Y * scale + cy;
        } else {
            const double zz = Z > 1e-12 ? Z : 1e-12;
            u = fx * X / zz + cx;
            v = fy * Y / zz + cy;
        }
        return valid;
    }

    // pixel -> unit bearing; returns validity
    bool unproject(double u, double v, double& bx, double& by, double& bz) const {
        const double mx = (u - cx) / fx;
        const double my = (v - cy) / fy;
        const double rd = std::hypot(mx, my);
        double theta = rd;
        for (int i = 0; i < 8; ++i)
            theta -= (d(theta) - rd) / d_prime(theta);
        if (theta < 0.0) theta = 0.0;
        const bool valid = (theta < max_theta) && (std::abs(d(theta) - rd) < 1e-9);
        const double s = rd > 1e-12 ? std::sin(theta) / rd : 1.0;
        bx = mx * s; by = my * s; bz = std::cos(theta);
        const double n = std::sqrt(bx * bx + by * by + bz * bz);
        bx /= n; by /= n; bz /= n;
        return valid;
    }
};

}  // namespace podslam
