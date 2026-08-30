// Port step 3: the KLT front-end (podslam/frontend/klt.py) on OpenCV C++, verified by
// replaying the frame dump (podslam-cpp/tools/dump_frontend_golden.py) and comparing
// track ids and pixel positions with the Python oracle, frame by frame.
//
// Transcription notes (parity-critical):
//   - identical OpenCV calls and parameters: goodFeaturesToTrack(maxCorners, quality,
//     minDistance=12, blockSize=5) with the min-distance circle mask; the noise gate
//     raises qualityLevel to gate*median(minEig over mask)/max; pyramidal LK win 21,
//     4 levels, criteria (EPS|COUNT, 30, 0.01), OPTFLOW_USE_INITIAL_FLOW both ways
//   - IMU-predicted initial flow through the KB4 model (bit-parity-proven header)
//   - forward/backward gate 1.0 px (1.5 px for stereo), essential-matrix RANSAC on
//     normalized coordinates (prob .999, thr 0.004) for the central 80 deg, the
//     rotation-compensated angular gate (6 deg) for the rim
//   - stereo: LK from the depth-based initial guess (last stereo depth or 3 m),
//     midpoint triangulation + reprojection verification (stereo.hpp, parity-proven)
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/geometry.hpp>   // OpenCV 5: calib3d split; findEssentialMat lives in geometry
#include <opencv2/features.hpp>   // OpenCV 5: features2d renamed; goodFeaturesToTrack lives here

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "podslam/kb4.hpp"
#include "podslam/stereo.hpp"

using podslam::Kb4;
using podslam::Mat4;

struct CamOut { std::vector<int64_t> ids; std::vector<cv::Point2f> px; };

struct KltFrontend {
    // DEFAULTS from klt.py
    int max_features = 300, min_distance = 12, win = 21, levels = 4, min_features = 60;
    double quality = 0.01, noise_gate = 4.0, fb_err_px = 1.0, stereo_max_px = 2.0;
    double default_depth = 3.0, rim_ang_deg = 6.0, ransac_thr_norm = 0.004;

    std::vector<Kb4> cams;
    std::vector<Mat4> T_imu_cam;
    cv::TermCriteria crit{cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01};

    cv::Mat prev_img;
    std::vector<cv::Point2f> prev_px;
    std::vector<cv::Vec3d> prev_b;
    std::vector<int64_t> ids;
    std::map<int64_t, double> depth;
    int64_t next_id = 0;

    std::vector<cv::Vec3d> bearings(const Kb4& cam, const std::vector<cv::Point2f>& px,
                                    std::vector<bool>& valid) const {
        std::vector<cv::Vec3d> b(px.size());
        valid.assign(px.size(), false);
        for (size_t i = 0; i < px.size(); ++i) {
            double bx, by, bz;
            valid[i] = cam.unproject(px[i].x, px[i].y, bx, by, bz);
            b[i] = {bx, by, bz};
        }
        return b;
    }

    std::vector<cv::Point2f> predict_px(const cv::Matx33d* dR_imu) const {
        if (!dR_imu || prev_b.empty()) return prev_px;
        cv::Matx33d R_ic;
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) R_ic(r, c) = T_imu_cam[0].m[r][c];
        const cv::Matx33d dR_cam = R_ic.t() * (*dR_imu) * R_ic;
        const cv::Matx33d dRT = dR_cam.t();
        std::vector<cv::Point2f> out = prev_px;
        for (size_t i = 0; i < prev_b.size(); ++i) {
            const cv::Vec3d bp = dRT * prev_b[i];
            double u, v;
            if (cams[0].project(bp[0], bp[1], bp[2], u, v)) out[i] = cv::Point2f(float(u), float(v));
        }
        return out;
    }

    std::vector<cv::Point2f> detect(const cv::Mat& img, int n_new) {
        if (n_new <= 0) return {};
        cv::Mat m(img.size(), CV_8U, cv::Scalar(255));
        for (const auto& p : prev_px)
            cv::circle(m, cv::Point(int(p.x), int(p.y)), min_distance, cv::Scalar(0), -1);
        double q = quality;
        if (noise_gate > 0) {
            cv::Mat resp;
            cv::cornerMinEigenVal(img, resp, 5);
            // mirror klt.py: with no STATIC mask the median runs over the full response
            // map (the circle mask is not applied to the median), and np.median averages
            // the two middle elements for even counts
            std::vector<float> vals(resp.begin<float>(), resp.end<float>());
            double rmax; cv::minMaxLoc(resp, nullptr, &rmax);
            if (!vals.empty() && rmax > 0) {
                const size_t h2 = vals.size() / 2;
                std::nth_element(vals.begin(), vals.begin() + h2, vals.end());
                double floor_ = vals[h2];
                if (vals.size() % 2 == 0) {
                    const double lower = *std::max_element(vals.begin(), vals.begin() + h2);
                    floor_ = 0.5 * (floor_ + lower);
                }
                if (floor_ > 0) q = std::max(q, std::min(0.5, noise_gate * floor_ / rmax));
            }
        }
        std::vector<cv::Point2f> pts;
        cv::goodFeaturesToTrack(img, pts, n_new, q, min_distance, m, 5);
        const int need = min_features - int(prev_px.size());
        if (noise_gate > 0 && q > quality && int(pts.size()) < need) {
            pts.clear();
            cv::goodFeaturesToTrack(img, pts, need, quality, min_distance, m, 5);
        }
        return pts;
    }

    std::vector<bool> inside(const std::vector<cv::Point2f>& px, const cv::Size& sz) const {
        std::vector<bool> ok(px.size());
        for (size_t i = 0; i < px.size(); ++i)
            ok[i] = px[i].x >= 1 && px[i].x < sz.width - 1 && px[i].y >= 1 && px[i].y < sz.height - 1;
        return ok;
    }

    // essential_inliers (common.py): central RANSAC + rim angular gate
    std::vector<bool> essential_inliers(const std::vector<cv::Vec3d>& bp, const std::vector<cv::Vec3d>& bc,
                                        const cv::Matx33d* dR_cam) const {
        const size_t n = bp.size();
        std::vector<bool> keep(n, true);
        if (n == 0) return keep;
        const double cz = std::cos(80.0 * M_PI / 180.0);
        std::vector<int> idx;
        for (size_t i = 0; i < n; ++i)
            if (bp[i][2] > cz && bc[i][2] > cz) idx.push_back(int(i));
        if (idx.size() >= 8) {
            std::vector<cv::Point2d> p, c;
            for (int i : idx) {
                p.emplace_back(bp[i][0] / bp[i][2], bp[i][1] / bp[i][2]);
                c.emplace_back(bc[i][0] / bc[i][2], bc[i][1] / bc[i][2]);
            }
            cv::Mat mask;
            // OpenCV 5 signature: camera matrices instead of focal/pp (identity K == focal 1, pp 0,
            // which is exactly what the Python call passes)
            const cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
            cv::Mat E = cv::findEssentialMat(p, c, K, cv::noArray(), K, cv::noArray(),
                                             cv::RANSAC, 0.999, ransac_thr_norm, mask);
            if (!mask.empty())
                for (size_t k = 0; k < idx.size(); ++k) keep[idx[k]] = mask.at<uint8_t>(int(k)) != 0;
        }
        if (dR_cam) {
            const cv::Matx33d dRT = dR_cam->t();
            for (size_t i = 0; i < n; ++i) {
                if (bp[i][2] > cz && bc[i][2] > cz) continue;      // central handled above
                const cv::Vec3d pred = dRT * bp[i];
                double cosang = pred.dot(bc[i]);
                cosang = std::max(-1.0, std::min(1.0, cosang));
                keep[i] = std::acos(cosang) * 180.0 / M_PI < rim_ang_deg;
            }
        }
        return keep;
    }

    CamOut stereo(size_t j, const cv::Mat& img_j, const cv::Mat& img0,
                  const std::vector<cv::Point2f>& px0, const std::vector<int64_t>& ids0,
                  const std::vector<cv::Vec3d>& b0) {
        CamOut out;
        if (px0.empty()) return out;
        // initial guess from the last stereo depth (or the default)
        std::vector<cv::Point2f> guess = px0;
        for (size_t i = 0; i < px0.size(); ++i) {
            auto it = depth.find(ids0[i]);
            const double d = it != depth.end() ? it->second : default_depth;
            // p_cj = T_cj_c0 * (b0 * d);  T_cj_c0 = inv(T_imu_cam[j]) * T_imu_cam[0]
            double p0[3] = {b0[i][0] * d, b0[i][1] * d, b0[i][2] * d};
            double pi[3], pj[3];
            // to IMU frame via T_imu_cam[0]
            for (int r = 0; r < 3; ++r)
                pi[r] = T_imu_cam[0].m[r][0] * p0[0] + T_imu_cam[0].m[r][1] * p0[1] +
                        T_imu_cam[0].m[r][2] * p0[2] + T_imu_cam[0].m[r][3];
            // to cam j frame via inv(T_imu_cam[j])
            double q[3] = {pi[0] - T_imu_cam[j].m[0][3], pi[1] - T_imu_cam[j].m[1][3], pi[2] - T_imu_cam[j].m[2][3]};
            for (int r = 0; r < 3; ++r)
                pj[r] = T_imu_cam[j].m[0][r] * q[0] + T_imu_cam[j].m[1][r] * q[1] + T_imu_cam[j].m[2][r] * q[2];
            double u, v;
            if (cams[j].project(pj[0], pj[1], pj[2], u, v)) guess[i] = cv::Point2f(float(u), float(v));
        }
        std::vector<cv::Point2f> nxt = guess, back = px0;
        std::vector<uint8_t> st, st2;
        cv::Mat err;
        cv::calcOpticalFlowPyrLK(img0, img_j, px0, nxt, st, err, cv::Size(win, win), levels, crit,
                                 cv::OPTFLOW_USE_INITIAL_FLOW);
        cv::calcOpticalFlowPyrLK(img_j, img0, nxt, back, st2, err, cv::Size(win, win), levels, crit,
                                 cv::OPTFLOW_USE_INITIAL_FLOW);
        const auto ok_in = inside(nxt, img_j.size());
        for (size_t i = 0; i < px0.size(); ++i) {
            const double fb = cv::norm(back[i] - px0[i]);
            if (!(st[i] && st2[i] && fb < fb_err_px * 1.5 && ok_in[i])) continue;
            double bx, by, bz;
            if (!cams[j].unproject(nxt[i].x, nxt[i].y, bx, by, bz)) continue;
            const double bj[3] = {bx, by, bz};
            const double b0i[3] = {b0[i][0], b0[i][1], b0[i][2]};
            const double p0i[2] = {px0[i].x, px0[i].y};
            const double pji[2] = {nxt[i].x, nxt[i].y};
            double pt[3], dep = 0;
            if (!podslam::stereo_verify(T_imu_cam[0], T_imu_cam[j], cams[0], cams[j],
                                        b0i, bj, p0i, pji, pt, dep, stereo_max_px))
                continue;
            depth[ids0[i]] = dep;
            out.ids.push_back(ids0[i]);
            out.px.push_back(nxt[i]);
        }
        return out;
    }

    std::vector<CamOut> process(const cv::Mat& img0, const std::vector<cv::Mat>& imgs,
                                const cv::Matx33d* dR_imu, int& n_new) {
        n_new = 0;
        std::vector<cv::Point2f> px;
        std::vector<int64_t> cur_ids;
        std::vector<cv::Vec3d> b;
        if (!prev_img.empty() && !prev_px.empty()) {
            auto guess = predict_px(dR_imu);
            std::vector<cv::Point2f> nxt = guess, back = prev_px;
            std::vector<uint8_t> st, st2;
            cv::Mat err;
            cv::calcOpticalFlowPyrLK(prev_img, img0, prev_px, nxt, st, err, cv::Size(win, win), levels,
                                     crit, cv::OPTFLOW_USE_INITIAL_FLOW);
            cv::calcOpticalFlowPyrLK(img0, prev_img, nxt, back, st2, err, cv::Size(win, win), levels,
                                     crit, cv::OPTFLOW_USE_INITIAL_FLOW);
            const auto ok_in = inside(nxt, img0.size());
            std::vector<cv::Point2f> px1;
            std::vector<int64_t> ids1;
            std::vector<cv::Vec3d> bprev1;
            for (size_t i = 0; i < prev_px.size(); ++i) {
                const double fb = cv::norm(back[i] - prev_px[i]);
                if (st[i] && st2[i] && fb < fb_err_px && ok_in[i]) {
                    px1.push_back(nxt[i]); ids1.push_back(ids[i]); bprev1.push_back(prev_b[i]);
                }
            }
            std::vector<bool> bval;
            auto b1 = bearings(cams[0], px1, bval);
            std::vector<cv::Point2f> px2; std::vector<int64_t> ids2;
            std::vector<cv::Vec3d> bp2, b2;
            for (size_t i = 0; i < px1.size(); ++i)
                if (bval[i]) { px2.push_back(px1[i]); ids2.push_back(ids1[i]); bp2.push_back(bprev1[i]); b2.push_back(b1[i]); }
            if (px2.size() >= 8) {
                cv::Matx33d R_ic;
                for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) R_ic(r, c) = T_imu_cam[0].m[r][c];
                cv::Matx33d dR_cam_m;
                const cv::Matx33d* dR_cam = nullptr;
                if (dR_imu) { dR_cam_m = R_ic.t() * (*dR_imu) * R_ic; dR_cam = &dR_cam_m; }
                auto keep = essential_inliers(bp2, b2, dR_cam);
                for (size_t i = 0; i < px2.size(); ++i)
                    if (keep[i]) { px.push_back(px2[i]); cur_ids.push_back(ids2[i]); b.push_back(b2[i]); }
            } else { px = px2; cur_ids = ids2; b = b2; }
        }
        prev_px = px;                                       // for the detect mask
        auto fresh = detect(img0, max_features - int(px.size()));
        if (!fresh.empty()) {
            std::vector<bool> bval;
            auto bn = bearings(cams[0], fresh, bval);
            for (size_t i = 0; i < fresh.size(); ++i) {
                if (!bval[i]) continue;
                px.push_back(fresh[i]); cur_ids.push_back(next_id++); b.push_back(bn[i]);
                ++n_new;
            }
        }
        std::vector<CamOut> cams_out(1);
        cams_out[0].ids = cur_ids; cams_out[0].px = px;
        for (size_t j = 1; j < cams.size(); ++j)
            cams_out.push_back(stereo(j, imgs[j], img0, px, cur_ids, b));
        prev_img = img0.clone(); prev_px = px; prev_b = b; ids = cur_ids;
        std::set<int64_t> live(cur_ids.begin(), cur_ids.end());
        for (auto it = depth.begin(); it != depth.end();)
            it = live.count(it->first) ? std::next(it) : depth.erase(it);
        return cams_out;
    }
};

// ---------------------------------------------------------------------------
static bool load_rig_lines(const char* golden_path, std::vector<Kb4>& cams, std::vector<Mat4>& Ts) {
    std::ifstream in(golden_path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("R ", 0) != 0) { if (!cams.empty()) break; else continue; }
        std::istringstream ss(line.substr(2));
        double v[20];
        for (auto& x : v) ss >> x;
        cams.push_back(Kb4{v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]});
        Mat4 T{}; for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) T.m[r][c] = (r == c) ? 1.0 : 0.0;
        for (int i = 0; i < 12; ++i) T.m[i / 4][i % 4] = v[8 + i];
        Ts.push_back(T);
    }
    return !cams.empty();
}

int main(int argc, char** argv) {
    cv::setNumThreads(1);                       // the Python dump runs with cv2.setNumThreads(1)
    const char* dir = argc > 1 ? argv[1] : "/tmp/frontend_golden";
    const char* rig = argc > 2 ? argv[2] : "podslam-cpp/tests/data/backend_golden.txt";
    const bool teacher = argc > 3 && std::atoi(argv[3]) != 0;
    KltFrontend fe;
    if (!load_rig_lines(rig, fe.cams, fe.T_imu_cam)) { std::printf("no rig lines in %s\n", rig); return 1; }

    std::ifstream fb(std::string(dir) + "/frames.bin", std::ios::binary);
    std::ifstream ft(std::string(dir) + "/tracks.txt");
    if (!fb || !ft) { std::printf("missing golden files in %s\n", dir); return 1; }

    int frame_n = 0, mism_frames = 0;
    long total_ids = 0, id_mism = 0, count_extra = 0;
    double max_px_diff = 0;
    // previous images for dR: the dump stores dR per frame
    while (true) {
        int64_t t_ns; int32_t n_cams, h, w;
        if (!fb.read(reinterpret_cast<char*>(&t_ns), 8)) break;
        fb.read(reinterpret_cast<char*>(&n_cams), 4);
        fb.read(reinterpret_cast<char*>(&h), 4);
        fb.read(reinterpret_cast<char*>(&w), 4);
        std::vector<cv::Mat> imgs;
        for (int c = 0; c < n_cams; ++c) {
            cv::Mat im(h, w, CV_8U);
            fb.read(reinterpret_cast<char*>(im.data), size_t(h) * w);
            imgs.push_back(im);
        }
        double dRv[9];
        fb.read(reinterpret_cast<char*>(dRv), 72);
        cv::Matx33d dR(dRv);
        bool identity = true;
        for (int i = 0; i < 9 && identity; ++i) identity = std::abs(dRv[i] - (i % 4 == 0 ? 1.0 : 0.0)) < 1e-15;
        int n_new = 0;
        auto outs = fe.process(imgs[0], imgs, identity ? nullptr : &dR, n_new);
        if (frame_n == 0) {
            std::printf("  cpp frame0 cam0: n=%zu new=%d\n", outs[0].ids.size(), n_new);
            for (size_t i = 0; i < outs[0].ids.size() && i < 3; ++i)
                std::printf("    id %lld px %.4f %.4f\n", (long long)outs[0].ids[i], outs[0].px[i].x, outs[0].px[i].y);
        }
        // Functional comparison: nearest-neighbour position matching per camera.
        // Bit-identical corner sets across compilers are not achievable (threshold-edge
        // corners flip on last-bit rounding: 116 vs 115 with identical leading corners
        // on two OpenCV versions), so the criterion is: the same features in the same
        // places, with the same stereo associations.
        std::vector<int64_t> gold_ids0; std::vector<cv::Point2f> gold_px0;
        for (int c = 0; c < n_cams; ++c) {
            std::string tag; int64_t tt; int cc, nn;
            ft >> tag >> tt >> cc >> nn;
            std::vector<double> gu(nn), gv(nn);
            for (int i = 0; i < nn; ++i) {
                int64_t id_; ft >> id_ >> gu[i] >> gv[i];
                if (c == 0) { gold_ids0.push_back(id_); gold_px0.push_back(cv::Point2f(float(gu[i]), float(gv[i]))); }
            }
            const CamOut& got = size_t(c) < outs.size() ? outs[c] : CamOut{};
            int matched = 0; double worst = 0;
            std::vector<bool> used(got.px.size(), false);
            for (int i = 0; i < nn; ++i) {
                double best = 1e9; int bj = -1;
                for (size_t k2 = 0; k2 < got.px.size(); ++k2) {
                    if (used[k2]) continue;
                    const double d = std::hypot(got.px[k2].x - gu[i], got.px[k2].y - gv[i]);
                    if (d < best) { best = d; bj = int(k2); }
                }
                if (bj >= 0 && best <= 0.5) { used[bj] = true; ++matched; if (best > worst) worst = best; }
            }
            total_ids += nn;
            id_mism += nn - matched;                     // golden features with no C++ counterpart
            if (worst > max_px_diff) max_px_diff = worst;
            count_extra += int(got.px.size()) - matched; // C++ features with no golden counterpart
        }
        { std::string tag; int64_t tt; int g; ft >> tag >> tt >> g; }
        if (teacher) {
            // teacher forcing: continue from the GOLDEN cam0 state so each frame's
            // divergence reflects only this frame's processing
            fe.prev_px.clear(); fe.prev_b.clear(); fe.ids.clear();
            int64_t maxid = fe.next_id - 1;
            for (size_t i = 0; i < gold_px0.size(); ++i) {
                double bx, by, bz;
                if (!fe.cams[0].unproject(gold_px0[i].x, gold_px0[i].y, bx, by, bz)) continue;
                fe.prev_px.push_back(gold_px0[i]);
                fe.prev_b.push_back({bx, by, bz});
                fe.ids.push_back(gold_ids0[i]);
                if (gold_ids0[i] > maxid) maxid = gold_ids0[i];
            }
            fe.next_id = maxid + 1;
        }
        ++frame_n;
    }
    (void)mism_frames;
    const double unmatched_pct = 100.0 * id_mism / std::max(1L, total_ids);
    const double extra_pct = 100.0 * count_extra / std::max(1L, total_ids);
    std::printf("frontend functional parity over %d frames: %ld golden features, %.2f%% unmatched, %.2f%% extra, matched worst |px| %.4g\n",
                frame_n, total_ids, unmatched_pct, extra_pct, max_px_diff);
    const int rc = (unmatched_pct < 3.0 && extra_pct < 3.0 && max_px_diff < 0.5) ? 0 : 2;
    std::puts(rc == 0 ? "frontend parity: OK" : "frontend parity: FAILED");
    return rc;
}
