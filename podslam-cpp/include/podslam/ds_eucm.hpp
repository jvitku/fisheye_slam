// Double Sphere (Usenko 2018) and EUCM (Khomutenko 2016) — C++ transcriptions of
// tools/fisheye/models.py, golden-tested against the Python reference.
#pragma once
#include <cmath>

namespace podslam {

struct DoubleSphere {
    double fx, fy, cx, cy, xi, alpha;

    bool project(double X, double Y, double Z, double& u, double& v) const {
        const double d1 = std::sqrt(X * X + Y * Y + Z * Z);
        const double zeta = xi * d1 + Z;
        const double d2 = std::sqrt(X * X + Y * Y + zeta * zeta);
        const double denom = alpha * d2 + (1.0 - alpha) * zeta;
        const double w1 = alpha <= 0.5 ? alpha / (1.0 - alpha) : (1.0 - alpha) / alpha;
        const double w2 = (w1 + xi) / std::sqrt(2.0 * w1 * xi + xi * xi + 1.0);
        const bool valid = (Z > -w2 * d1) && (denom > 1e-12);
        const double dn = valid ? denom : 1.0;
        u = fx * X / dn + cx;
        v = fy * Y / dn + cy;
        return valid;
    }

    bool unproject(double u, double v, double& bx, double& by, double& bz) const {
        const double mx = (u - cx) / fx, my = (v - cy) / fy;
        const double r2 = mx * mx + my * my;
        bool valid = true;
        if (alpha > 0.5) valid = r2 <= 1.0 / (2.0 * alpha - 1.0);
        double disc = 1.0 - (2.0 * alpha - 1.0) * r2;
        if (disc < 0.0) disc = 0.0;
        const double mz = (1.0 - alpha * alpha * r2) / (alpha * std::sqrt(disc) + 1.0 - alpha);
        const double k = (mz * xi + std::sqrt(mz * mz + (1.0 - xi * xi) * r2)) / (mz * mz + r2);
        bx = k * mx; by = k * my; bz = k * mz - xi;
        const double n = std::sqrt(bx * bx + by * by + bz * bz);
        bx /= n; by /= n; bz /= n;
        return valid;
    }
};

struct Eucm {
    double fx, fy, cx, cy, alpha, beta;

    bool project(double X, double Y, double Z, double& u, double& v) const {
        const double rho = std::sqrt(beta * (X * X + Y * Y) + Z * Z);
        const double denom = alpha * rho + (1.0 - alpha) * Z;
        const double w = alpha <= 0.5 ? alpha / (1.0 - alpha) : (1.0 - alpha) / alpha;
        const bool valid = (Z > -w * rho) && (denom > 1e-12);
        const double dn = valid ? denom : 1.0;
        u = fx * X / dn + cx;
        v = fy * Y / dn + cy;
        return valid;
    }

    bool unproject(double u, double v, double& bx, double& by, double& bz) const {
        const double mx = (u - cx) / fx, my = (v - cy) / fy;
        const double r2 = mx * mx + my * my;
        const double gamma = 1.0 - alpha;
        double disc = 1.0 - (alpha - gamma) * beta * r2;
        bool valid = disc >= 0.0;
        if (alpha > 0.5) valid = valid && (r2 <= 1.0 / (beta * (2.0 * alpha - 1.0)));
        if (disc < 0.0) disc = 0.0;
        const double mz = (1.0 - beta * alpha * alpha * r2) / (alpha * std::sqrt(disc) + gamma);
        bx = mx; by = my; bz = mz;
        const double n = std::sqrt(bx * bx + by * by + bz * bz);
        bx /= n; by /= n; bz /= n;
        return valid;
    }
};

}  // namespace podslam
