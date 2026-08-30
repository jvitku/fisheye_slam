#include "podslam/ds_eucm.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>

template <class Cam>
int run_file(const char* path) {
    FILE* f = std::fopen(path, "r");
    if (!f) { std::printf("cannot open %s\n", path); return 1; }
    char line[1024];
    if (!std::fgets(line, sizeof line, f)) return 1;
    int n = 0, bad = 0; double max_px = 0, max_b = 0;
    while (std::fgets(line, sizeof line, f)) {
        double v[16]; char* p = line;
        for (int i = 0; i < 16; ++i) { v[i] = std::strtod(p, &p); if (*p == ',') ++p; }
        Cam cam{v[0], v[1], v[2], v[3], v[4], v[5]};
        double u, vv, bx, by, bz;
        const bool ok = cam.project(v[6], v[7], v[8], u, vv);
        if (ok != (v[11] != 0.0)) ++bad;
        const double dpx = std::hypot(u - v[9], vv - v[10]);
        if (ok && dpx > max_px) max_px = dpx;
        const bool bok = cam.unproject(v[9], v[10], bx, by, bz);
        if (bok != (v[15] != 0.0)) ++bad;
        const double db = std::sqrt(std::pow(bx - v[12], 2) + std::pow(by - v[13], 2) + std::pow(bz - v[14], 2));
        if (db > max_b) max_b = db;
        ++n;
    }
    std::fclose(f);
    std::printf("%s: %d samples, max |px| %.3g, max |bearing| %.3g, validity mismatches %d\n", path, n, max_px, max_b, bad);
    return (max_px < 1e-6 && max_b < 1e-9 && bad == 0) ? 0 : 2;
}

int main() {
    int rc = 0;
    rc |= run_file<podslam::DoubleSphere>("podslam-cpp/tests/data/ds_golden.csv");
    rc |= run_file<podslam::Eucm>("podslam-cpp/tests/data/eucm_golden.csv");
    std::puts(rc == 0 ? "DS/EUCM parity: OK" : "DS/EUCM parity: FAILED");
    return rc;
}
