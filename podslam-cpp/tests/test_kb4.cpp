// Golden parity test: the C++ Kb4 must match the Python reference on dumped samples.
#include "podslam/kb4.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

int run_file(const char* path) {
    FILE* f = std::fopen(path, "r");
    if (!f) { std::printf("cannot open %s\n", path); return 1; }
    char line[1024];
    if (!std::fgets(line, sizeof line, f)) return 1;   // header
    int n = 0, bad = 0;
    double max_px = 0, max_b = 0;
    while (std::fgets(line, sizeof line, f)) {
        double v[18];
        char* p = line;
        for (int i = 0; i < 18; ++i) { v[i] = std::strtod(p, &p); if (*p == ',') ++p; }
        podslam::Kb4 cam{v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
        double u, vv, bx, by, bz;
        const bool ok = cam.project(v[8], v[9], v[10], u, vv);
        const double dpx = std::hypot(u - v[11], vv - v[12]);
        if (dpx > max_px) max_px = dpx;
        if (ok != (v[13] != 0.0)) ++bad;
        const bool bok = cam.unproject(v[11], v[12], bx, by, bz);
        const double db = std::sqrt(std::pow(bx - v[14], 2) + std::pow(by - v[15], 2) + std::pow(bz - v[16], 2));
        if (db > max_b) max_b = db;
        if (bok != (v[17] != 0.0)) ++bad;
        ++n;
    }
    std::fclose(f);
    std::printf("%s: %d samples, max |px| diff %.3g, max |bearing| diff %.3g, validity mismatches %d\n",
                path, n, max_px, max_b, bad);
    return (max_px < 1e-6 && max_b < 1e-9 && bad == 0) ? 0 : 2;   // golden CSV carries 12 significant digits
}

int main() {
    int rc = 0;
    rc |= run_file("podslam-cpp/tests/data/kb4_golden.csv");
    rc |= run_file("podslam-cpp/tests/data/kb4_golden_tumvi.csv");
    std::puts(rc == 0 ? "KB4 parity: OK" : "KB4 parity: FAILED");
    return rc;
}
