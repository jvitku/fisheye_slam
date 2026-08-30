#include "podslam/stereo.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
    FILE* fr = std::fopen("podslam-cpp/tests/data/stereo_golden_rig.csv", "r");
    if (!fr) { std::puts("no rig file"); return 1; }
    double rig[12][4]; char line[512];
    for (int r = 0; r < 12; ++r) {
        if (!std::fgets(line, sizeof line, fr)) return 1;
        char* p = line;
        for (int c = 0; c < 4; ++c) { rig[r][c] = std::strtod(p, &p); if (*p == ',') ++p; }
    }
    std::fclose(fr);
    podslam::Mat4 T0, T1;
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) { T0.m[r][c] = rig[r][c]; T1.m[r][c] = rig[4 + r][c]; }
    podslam::Kb4 cam0{rig[8][0], rig[8][1], rig[8][2], rig[8][3], rig[9][0], rig[9][1], rig[9][2], rig[9][3]};
    podslam::Kb4 cam1{rig[10][0], rig[10][1], rig[10][2], rig[10][3], rig[11][0], rig[11][1], rig[11][2], rig[11][3]};
    FILE* f = std::fopen("podslam-cpp/tests/data/stereo_golden.csv", "r");
    if (!f) { std::puts("no golden"); return 1; }
    if (!std::fgets(line, sizeof line, f)) return 1;
    int n = 0, bad_keep = 0; double max_p = 0, max_d = 0;
    while (std::fgets(line, sizeof line, f)) {
        double v[15]; char* p = line;
        for (int i = 0; i < 15; ++i) { v[i] = std::strtod(p, &p); if (*p == ',') ++p; }
        double pt[3], dep = 0;
        const bool keep = podslam::stereo_verify(T0, T1, cam0, cam1, &v[0], &v[3], &v[6], &v[8], pt, dep);
        if (keep != (v[14] != 0.0)) ++bad_keep;
        if (keep && v[14] != 0.0) {
            const double dp = std::sqrt(std::pow(pt[0] - v[10], 2) + std::pow(pt[1] - v[11], 2) + std::pow(pt[2] - v[12], 2));
            if (dp > max_p) max_p = dp;
            const double dd = std::abs(dep - v[13]);
            if (dd > max_d) max_d = dd;
        }
        ++n;
    }
    std::fclose(f);
    std::printf("stereo: %d samples, keep mismatches %d, max |point| %.3g m, max |depth| %.3g m\n", n, bad_keep, max_p, max_d);
    const int rc = (bad_keep == 0 && max_p < 1e-6 && max_d < 1e-6) ? 0 : 2;
    std::puts(rc == 0 ? "stereo parity: OK" : "stereo parity: FAILED");
    return rc;
}
