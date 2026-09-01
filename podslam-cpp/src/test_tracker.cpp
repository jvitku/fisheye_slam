// Port step 4: the full tracker (podslam/tracker.py) in C++ — static initialiser,
// per-frame flow, keyframe policy, landmark bookkeeping — wired to the parity-proven
// KltFrontend (test_frontend.cpp) and Window (test_window.cpp).  Runs end-to-end on
// the frame dump (dump_frontend_golden.py: frames.bin + imu.txt) and compares the
// trajectory with the Python run (python_est.tum).
#define PODSLAM_NO_MAIN
#include "test_window.cpp"
#include "test_frontend.cpp"

#include <deque>
#include <chrono>
#include <cstdlib>

using gtsam::Vector3;

static cv::Matx33d skew3(const Vector3& v) {
    return {0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0};
}
static cv::Matx33d exp_so3(const Vector3& w) {
    const double th = w.norm();
    const cv::Matx33d I = cv::Matx33d::eye();
    if (th < 1e-12) return I + skew3(w);
    const Vector3 k = w / th;
    const cv::Matx33d K = skew3(k);
    return I + std::sin(th) * K + (1 - std::cos(th)) * (K * K);
}
static cv::Matx33d rotation_aligning(Vector3 a, Vector3 b) {
    a.normalize(); b.normalize();
    const Vector3 v = a.cross(b);
    const double c = a.dot(b);
    if (v.norm() < 1e-12) {
        if (c > 0) return cv::Matx33d::eye();
        Vector3 axis = a.cross(Vector3(1, 0, 0));
        if (axis.norm() < 1e-6) axis = a.cross(Vector3(0, 1, 0));
        axis.normalize();
        return exp_so3(M_PI * axis);
    }
    const cv::Matx33d K = skew3(v);
    return cv::Matx33d::eye() + K + (K * K) * (1.0 / (1.0 + c));
}

struct ImuBuffer {
    std::deque<double> t;
    std::deque<Vector3> w, a;
    void append(double tt, const Vector3& ww, const Vector3& aa) {
        if (!t.empty() && tt <= t.back()) return;
        t.push_back(tt); w.push_back(ww); a.push_back(aa);
        while (t.size() > 4 && t.back() - t.front() > 20.0) { t.pop_front(); w.pop_front(); a.pop_front(); }
    }
    // samples with t0 < t <= t1 (indices)
    void between(double t0, double t1, std::vector<size_t>& idx) const {
        idx.clear();
        for (size_t i = 0; i < t.size(); ++i)
            if (t[i] > t0 && t[i] <= t1) idx.push_back(i);
    }
};

static cv::Matx33d delta_rotation(const ImuBuffer& buf, double t0, double t1, const Vector3& bg) {
    std::vector<size_t> idx;
    buf.between(t0, t1, idx);
    cv::Matx33d R = cv::Matx33d::eye();
    double t_prev = t0;
    size_t last = size_t(-1);
    for (size_t i : idx) {
        const double dt = buf.t[i] - t_prev;
        if (dt > 0) { R = R * exp_so3((buf.w[i] - bg) * dt); t_prev = buf.t[i]; last = i; }
    }
    if (t1 > t_prev && last != size_t(-1))
        R = R * exp_so3((buf.w[last] - bg) * (t1 - t_prev));
    return R;
}

struct StaticInit {
    double window_s = 1.0, gyro_thr = 0.03, accel_std_thr = 0.35, max_wait_s = 3.0;
    std::vector<double> t;
    std::vector<Vector3> w, a;
    bool done = false;
    cv::Matx33d R_W_I;
    Vector3 gyro_bias{0, 0, 0};

    bool feed(double tt, const Vector3& ww, const Vector3& aa) {
        if (done) return true;
        t.push_back(tt); w.push_back(ww); a.push_back(aa);
        if (t.back() - t.front() < window_s) return false;
        size_t i0 = 0;
        while (i0 < t.size() && t[i0] < t.back() - window_s) ++i0;   // searchsorted(left)
        double wmax = 0;
        Vector3 mean = Vector3::Zero();
        const size_t n = t.size() - i0;
        for (size_t i = i0; i < t.size(); ++i) { wmax = std::max(wmax, w[i].norm()); mean += a[i]; }
        mean /= double(n);
        Vector3 var = Vector3::Zero();
        for (size_t i = i0; i < t.size(); ++i) { const Vector3 d = a[i] - mean; var += d.cwiseProduct(d); }
        const double amax_std = std::sqrt((var / double(n)).maxCoeff());
        const bool still = wmax < gyro_thr && amax_std < accel_std_thr;
        const bool forced = (t.back() - t.front()) > max_wait_s;
        if (still || forced) {
            finalize(i0);
        }
        return done;
    }
    void force() {   // legacy fallback for starving dynamic init
        if (done || t.empty()) return;
        size_t i0 = 0;
        while (i0 < t.size() && t[i0] < t.back() - window_s) ++i0;
        finalize(i0);
    }
    void finalize(size_t i0) {
        const size_t n = t.size() - i0;
        Vector3 mean = Vector3::Zero();
        for (size_t i = i0; i < t.size(); ++i) mean += a[i];
        mean /= double(n);
        {
            Vector3 g_body = mean / std::max(mean.norm(), 1e-9);
            R_W_I = rotation_aligning(g_body, Vector3(0, 0, 1));
            Vector3 bg = Vector3::Zero();
            for (size_t i = i0; i < t.size(); ++i) bg += w[i];
            gyro_bias = bg / double(n);
            done = true;
        }
    }
};

// podslam/init_dynamic.py — moving-platform initialisation.
// CrossCamMatcher: per-pair ROTATION warps (cam i rendered in cam j's geometry at
// infinite depth) so LK only finds the small translation parallax; stereo_verify
// turns matches into metric body-frame points.
struct CrossCamMatcher {
    double max_px = 3.0, fb_max = 1.5;
    int min_cand = 8;
    double cos_max = std::cos(80.0 * M_PI / 180.0), min_overlap = 0.03;
    const std::vector<Kb4>* cams = nullptr;
    const std::vector<Mat4>* Ts = nullptr;
    std::map<std::pair<int,int>, std::pair<cv::Mat,cv::Mat>> maps;   // (i,j) -> map_x, map_y
    std::set<std::pair<int,int>> no_overlap;
    cv::TermCriteria crit{cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01};

    static void rot_v(const Mat4& T, const cv::Vec3d& v, cv::Vec3d& o, bool transpose) {
        for (int r = 0; r < 3; ++r)
            o[r] = transpose ? T.m[0][r]*v[0] + T.m[1][r]*v[1] + T.m[2][r]*v[2]
                             : T.m[r][0]*v[0] + T.m[r][1]*v[1] + T.m[r][2]*v[2];
    }
    // direction cam_j coords -> cam_i coords: R_ci_cj = R_imu_ci^T * R_imu_cj
    cv::Vec3d cj_to_ci(int i, int j, const cv::Vec3d& b) const {
        cv::Vec3d imu, out;
        rot_v((*Ts)[j], b, imu, false);
        rot_v((*Ts)[i], imu, out, true);
        return out;
    }
    const std::pair<cv::Mat,cv::Mat>* pair_map(int i, int j, int w, int h) {
        auto key = std::make_pair(i, j);
        if (no_overlap.count(key)) return nullptr;
        auto it = maps.find(key);
        if (it != maps.end()) return &it->second;
        cv::Mat mx(h, w, CV_32F, cv::Scalar(-1)), my(h, w, CV_32F, cv::Scalar(-1));
        long n_ok = 0;
        for (int y = 0; y < h; ++y) {
            float* rx = mx.ptr<float>(y);
            float* ry = my.ptr<float>(y);
            for (int x = 0; x < w; ++x) {
                double bx, by, bz;
                if (!(*cams)[j].unproject(x + 0.5, y + 0.5, bx, by, bz)) continue;
                cv::Vec3d bi = cj_to_ci(i, j, {bx, by, bz});
                if (bi[2] <= 0.05) continue;
                double u, v;
                if (!(*cams)[i].project(bi[0], bi[1], bi[2], u, v)) continue;
                rx[x] = float(u); ry[x] = float(v); ++n_ok;
            }
        }
        if (double(n_ok) / (double(w) * h) < min_overlap) { no_overlap.insert(key); return nullptr; }
        return &(maps[key] = {mx, my});
    }

    // feats: per-cam (ids, px, bearings); imgs: per-cam gray. out: tid -> p_imu
    void match(const std::vector<cv::Mat>& imgs,
               const std::vector<CamOut>& feats,
               const std::vector<std::vector<cv::Vec3d>>& bearings,
               std::map<int64_t, cv::Vec3d>& pts) {
        std::map<int64_t, double> best_err;
        const int n = int(cams->size());
        for (int i = 0; i < n; ++i) {
            if (feats[i].ids.empty()) continue;
            for (int j = 0; j < n; ++j) {
                if (j == i) continue;
                const int w = imgs[j].cols, h = imgs[j].rows;
                auto* m = pair_map(i, j, w, h);
                if (!m) continue;
                std::vector<int> idx;
                std::vector<cv::Point2f> guess;
                for (size_t nfe = 0; nfe < feats[i].ids.size(); ++nfe) {
                    const cv::Vec3d& bi = bearings[i][nfe];
                    if (bi[2] <= cos_max) continue;
                    cv::Vec3d bj;
                    { cv::Vec3d imu; rot_v((*Ts)[i], bi, imu, false); rot_v((*Ts)[j], imu, bj, true); }
                    if (bj[2] <= cos_max) continue;
                    double u, v;
                    if (!(*cams)[j].project(bj[0], bj[1], bj[2], u, v)) continue;
                    const int xi = std::min(std::max(int(u), 0), w - 1), yi = std::min(std::max(int(v), 0), h - 1);
                    if (m->first.at<float>(yi, xi) < 0) continue;
                    idx.push_back(int(nfe));
                    guess.emplace_back(float(u), float(v));
                }
                if (int(idx.size()) < min_cand) continue;
                cv::Mat warp;
                cv::remap(imgs[i], warp, m->first, m->second, cv::INTER_LINEAR,
                          cv::BORDER_CONSTANT, cv::Scalar(0));
                std::vector<cv::Point2f> nxt = guess, back = guess;
                std::vector<uint8_t> st, st2;
                cv::Mat err;
                cv::calcOpticalFlowPyrLK(warp, imgs[j], guess, nxt, st, err, cv::Size(21, 21), 3,
                                         crit, cv::OPTFLOW_USE_INITIAL_FLOW);
                cv::calcOpticalFlowPyrLK(imgs[j], warp, nxt, back, st2, err, cv::Size(21, 21), 3,
                                         crit, cv::OPTFLOW_USE_INITIAL_FLOW);
                for (size_t q = 0; q < idx.size(); ++q) {
                    if (!(st[q] && st2[q])) continue;
                    if (cv::norm(back[q] - guess[q]) >= fb_max) continue;
                    if (nxt[q].x < 0 || nxt[q].x >= w || nxt[q].y < 0 || nxt[q].y >= h) continue;
                    double bx, by, bz;
                    if (!(*cams)[j].unproject(nxt[q].x, nxt[q].y, bx, by, bz)) continue;
                    const int fi = idx[q];
                    const double b0[3] = {bearings[i][fi][0], bearings[i][fi][1], bearings[i][fi][2]};
                    const double b1[3] = {bx, by, bz};
                    const double p0[2] = {feats[i].px[fi].x, feats[i].px[fi].y};
                    const double p1[2] = {nxt[q].x, nxt[q].y};
                    double pt[3], dep = 0;
                    if (!podslam::stereo_verify((*Ts)[i], (*Ts)[j], (*cams)[i], (*cams)[j],
                                                b0, b1, p0, p1, pt, dep, max_px))
                        continue;
                    const int64_t tid = feats[i].ids[fi];
                    const double e = double(err.at<float>(int(q)));
                    auto be = best_err.find(tid);
                    if (be == best_err.end() || e < be->second) {
                        best_err[tid] = e;
                        pts[tid] = {pt[0], pt[1], pt[2]};
                    }
                }
            }
        }
    }
};

// DynamicInitializer (solver): gyro chain + robust translation chain + linear
// LSQ for gravity (frame-0 coords) and per-frame velocities; |g| pinned, v re-solved.
struct DynInit {
    double window_s = 1.5, g_mag = 9.81;
    int min_frames = 6, min_common = 8;
    struct Frame { double t; std::map<int64_t, cv::Vec3d> pts; };
    std::vector<Frame> frames;

    void add(double t, std::map<int64_t, cv::Vec3d>& pts) {
        if (!pts.empty()) frames.push_back({t, pts});
    }
    bool ready() const {
        return int(frames.size()) >= min_frames &&
               frames.back().t - frames.front().t >= window_s - 1e-9;
    }

    // returns true + outputs on success
    bool solve(const ImuBuffer& imu, cv::Matx33d& R_W_I, Vector3& v_W, Vector3& p_W) {
        if (!ready()) return false;
        const int m = int(frames.size());
        std::vector<cv::Matx33d> C(1, cv::Matx33d::eye());
        std::vector<Vector3> dP, dV, o(1, Vector3::Zero());
        std::vector<double> dts;
        for (int k = 1; k < m; ++k) {
            const double ta = frames[k-1].t, tb = frames[k].t;
            std::vector<size_t> idx;
            imu.between(ta, tb, idx);
            if (idx.empty()) return false;
            // gyro delta + bias-0 preintegration (frame k-1 coords)
            cv::Matx33d R = cv::Matx33d::eye();
            Vector3 dv = Vector3::Zero(), dp = Vector3::Zero();
            double t_prev = ta;
            size_t last = size_t(-1);
            auto step = [&](const Vector3& a, const Vector3& w, double dt) {
                Vector3 a0; for (int r = 0; r < 3; ++r) a0[r] = R(r,0)*a[0] + R(r,1)*a[1] + R(r,2)*a[2];
                dp += dv * dt + 0.5 * a0 * dt * dt;
                dv += a0 * dt;
                R = R * exp_so3(w * dt);
            };
            for (size_t i : idx) {
                const double dt = imu.t[i] - t_prev;
                if (dt > 0) { step(imu.a[i], imu.w[i], dt); t_prev = imu.t[i]; last = i; }
            }
            if (tb > t_prev && last != size_t(-1)) step(imu.a[last], imu.w[last], tb - t_prev);
            C.push_back(C[k-1] * R);
            dP.push_back(dp); dV.push_back(dv); dts.push_back(tb - ta);
            // translation chain: median + 3xMAD inlier re-mean of per-point votes
            std::vector<Vector3> d;
            for (const auto& [tid, pb] : frames[k].pts) {
                auto it = frames[k-1].pts.find(tid);
                if (it == frames[k-1].pts.end()) continue;
                Vector3 va, vb;
                for (int r = 0; r < 3; ++r) {
                    va[r] = C[k-1](r,0)*it->second[0] + C[k-1](r,1)*it->second[1] + C[k-1](r,2)*it->second[2];
                    vb[r] = C[k](r,0)*pb[0] + C[k](r,1)*pb[1] + C[k](r,2)*pb[2];
                }
                d.push_back(va - vb);
            }
            if (int(d.size()) < min_common) return false;
            auto med = [&](int axis) {
                std::vector<double> v; v.reserve(d.size());
                for (auto& x : d) v.push_back(x[axis]);
                std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
                return v[v.size()/2];
            };
            const Vector3 mm(med(0), med(1), med(2));
            std::vector<double> r; r.reserve(d.size());
            for (auto& x : d) r.push_back((x - mm).norm());
            std::vector<double> rs = r;
            std::nth_element(rs.begin(), rs.begin() + rs.size()/2, rs.end());
            const double thr = std::max(3.0 * rs[rs.size()/2], 0.05);
            Vector3 t_ab = Vector3::Zero();
            int n_in = 0;
            for (size_t q = 0; q < d.size(); ++q)
                if (r[q] <= thr) { t_ab += d[q]; ++n_in; }
            if (n_in < min_common) return false;
            o.push_back(o[k-1] + t_ab / double(n_in));
        }
        // LSQ: x = [v_0..v_{m-1}, g0]
        const int nu = 3*m + 3, ne = 6*(m-1);
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ne, nu);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(ne);
        for (int k = 0; k < m-1; ++k) {
            const double dt = dts[k];
            Vector3 cdp, cdv;
            for (int r = 0; r < 3; ++r) {
                cdp[r] = C[k](r,0)*dP[k][0] + C[k](r,1)*dP[k][1] + C[k](r,2)*dP[k][2];
                cdv[r] = C[k](r,0)*dV[k][0] + C[k](r,1)*dV[k][1] + C[k](r,2)*dV[k][2];
            }
            for (int r = 0; r < 3; ++r) {
                A(6*k+r, 3*k+r) = dt;
                A(6*k+r, 3*m+r) = 0.5 * dt * dt;
                b(6*k+r) = o[k+1][r] - o[k][r] - cdp[r];
                A(6*k+3+r, 3*k+r) = -1.0;
                A(6*k+3+r, 3*(k+1)+r) = 1.0;
                A(6*k+3+r, 3*m+r) = -dt;
                b(6*k+3+r) = cdv[r];
            }
        }
        Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
        Vector3 g0(x(3*m), x(3*m+1), x(3*m+2));
        const double gn = g0.norm();
        if (gn < 0.5*g_mag || gn > 1.5*g_mag) return false;
        const Vector3 u = g0 / gn, g_fix = g_mag * u;
        Eigen::MatrixXd Av = A.leftCols(3*m);
        Eigen::VectorXd bv = b - A.rightCols(3) * Eigen::Vector3d(g_fix[0], g_fix[1], g_fix[2]);
        Eigen::VectorXd v = Av.colPivHouseholderQr().solve(bv);
        const cv::Matx33d R_W_B0 = rotation_aligning(-u, Vector3(0, 0, 1));
        R_W_I = R_W_B0 * C[m-1];
        Vector3 vm(v(3*(m-1)), v(3*(m-1)+1), v(3*(m-1)+2));
        for (int r = 0; r < 3; ++r) {
            v_W[r] = R_W_B0(r,0)*vm[0] + R_W_B0(r,1)*vm[1] + R_W_B0(r,2)*vm[2];
            p_W[r] = R_W_B0(r,0)*o[m-1][0] + R_W_B0(r,1)*o[m-1][1] + R_W_B0(r,2)*o[m-1][2];
        }
        return true;
    }
};

struct Tracker {
    // TrackerConfig defaults (mirrors tracker.py)
    int kf_every = 3;
    double kf_min_track_ratio = 0.6, lag_s = 4.0;
    double kf_dense_init_s = 0.0;      // keyframe every frame this long after init
    double t_init = -1;
    int max_landmarks_per_kf = 120;
    bool setup_dw = false;
    // init modes (tracker.py): static | auto (deferred warm-up; dynamic solve on moving starts)
    std::string init_mode = "static";
    double init_window_s = 1.5, init_max_s = 6.0;
    int init_stride = 2, init_frame_i = -1;
    double t_first_frame = -1;
    double dyn_vel_sigma = 0.3, dyn_tilt_sigma = 0.03;
    DynInit dyn;
    CrossCamMatcher matcher;

    KltFrontend fe;
    MultiKltFrontend mfe;
    bool per_cam = false;              // multiklt: disjoint per-camera id spaces
    Window* win = nullptr;
    std::vector<std::array<double, 20>> rig_rows;
    double px_sigma = 1.5;
    std::array<double, 4> noise_scale{5.7, 1.8, 1.2, 4.5};
    std::array<double, 4> imu_noise{};      // gyro_nd, gyro_rw, accel_nd, accel_rw
    double accel_scale = 1.0;

    ImuBuffer imu;
    StaticInit init;
    std::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params> pp;
    std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> pim;
    double pim_t_last = -1;
    gtsam::NavState kf_navstate;
    gtsam::imuBias::ConstantBias bias;
    Vector3 gyro_bias{0, 0, 0};
    int k = -1;
    double t_prev_frame = -1;
    int frames_since_kf = 0, n_tracks_at_kf = 0;
    std::map<int64_t, long> track_to_lm;
    long next_lm = 0;
    bool initialized = false;

    void setup() {
        win = new Window(rig_rows, px_sigma);
        pp = gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedU(9.81);
        pp->setGyroscopeCovariance(gtsam::I_3x3 * std::pow(noise_scale[1] * imu_noise[0], 2));
        pp->setAccelerometerCovariance(gtsam::I_3x3 * std::pow(noise_scale[0] * imu_noise[2], 2));
        pp->setIntegrationCovariance(gtsam::I_3x3 * 1e-8);
        pp->setBiasAccCovariance(gtsam::I_3x3 * std::pow(noise_scale[2] * imu_noise[3], 2));
        pp->setBiasOmegaCovariance(gtsam::I_3x3 * std::pow(noise_scale[3] * imu_noise[1], 2));
        pim = std::make_unique<gtsam::PreintegratedCombinedMeasurements>(pp, bias);
    }

    void register_imu(double t, const Vector3& g, const Vector3& a_raw) {
        const Vector3 a = a_raw * accel_scale;
        imu.append(t, g, a);
        if (!initialized) init.feed(t, g, a);
    }

    void integrate_until(double t1) {
        if (pim_t_last < 0) { pim_t_last = t1; return; }
        std::vector<size_t> idx;
        imu.between(pim_t_last, t1, idx);
        size_t last = size_t(-1);
        for (size_t i : idx) {
            const double dt = imu.t[i] - pim_t_last;
            if (dt > 0) { pim->integrateMeasurement(imu.a[i], imu.w[i], dt); pim_t_last = imu.t[i]; last = i; }
        }
        if (t1 > pim_t_last && last != size_t(-1)) {
            pim->integrateMeasurement(imu.a[last], imu.w[last], t1 - pim_t_last);
            pim_t_last = t1;
        }
    }

    size_t n_cams() const { return per_cam ? mfe.subs.size() : fe.cams.size(); }
    const Kb4& cam_of(size_t c) const { return per_cam ? mfe.subs[c].cams[0] : fe.cams[c]; }
    const Mat4& T_of(size_t c) const { return per_cam ? mfe.subs[c].T_imu_cam[0] : fe.T_imu_cam[c]; }

    bool in_front(const std::array<double, 3>& p_w, const gtsam::Pose3& T_W_I) const {
        const gtsam::Point3 p_i = T_W_I.transformTo(gtsam::Point3(p_w[0], p_w[1], p_w[2]));
        for (size_t c = 0; c < n_cams(); ++c) {
            const auto& T = T_of(c);
            const double q[3] = {p_i.x() - T.m[0][3], p_i.y() - T.m[1][3], p_i.z() - T.m[2][3]};
            const double z = T.m[0][2] * q[0] + T.m[1][2] * q[1] + T.m[2][2] * q[2];
            if (z > 0.15) return true;
        }
        return false;
    }

    // tracker.py _add_landmarks_and_factors (generalized over every camera's
    // native tracks with a seen-set; stereo-centric rigs — shared ids — behave
    // bit-identically to the old cam0-only loop, per-camera rigs contribute
    // every camera's tracks as mono smart landmarks)
    void add_landmarks(int kk, double t, const std::vector<CamOut>& cams,
                       const std::vector<std::vector<cv::Vec3d>>& bearings,
                       const gtsam::Pose3& T_W_I) {
        std::vector<std::map<int64_t, size_t>> by_cam(cams.size());
        for (size_t c = 0; c < cams.size(); ++c)
            for (size_t i = 0; i < cams[c].ids.size(); ++i) by_cam[c][cams[c].ids[i]] = i;
        int n_new = 0;
        std::set<int64_t> seen;
        for (size_t c0 = 0; c0 < cams.size(); ++c0) {
            for (size_t n = 0; n < cams[c0].ids.size(); ++n) {
                const int64_t tid = cams[c0].ids[n];
                if (!seen.insert(tid).second) continue;
                long j = -1;
                auto it = track_to_lm.find(tid);
                if (it != track_to_lm.end()) {
                    j = it->second;
                    const auto lt = win->landmark_t.find(j);
                    const bool alive = lt != win->landmark_t.end() && (t - lt->second) < win->lag_s;
                    if (!alive) { track_to_lm.erase(it); j = -1; }
                }
                const cv::Vec3d& b0 = bearings[c0][n];
                if (j < 0) {
                    if (n_new >= max_landmarks_per_kf * 3 || !win->can_observe(b0[0], b0[1], b0[2])) continue;
                    j = next_lm++;
                    track_to_lm[tid] = j;
                    ++n_new;
                    win->add_observation({kk, int(c0), j, b0[0] / b0[2], b0[1] / b0[2]}, t);
                    for (size_t c = 0; c < cams.size(); ++c) {
                        if (c == c0) continue;
                        auto f = by_cam[c].find(tid);
                        if (f == by_cam[c].end()) continue;
                        const cv::Vec3d& bc = bearings[c][f->second];
                        if (win->can_observe(bc[0], bc[1], bc[2]))
                            win->add_observation({kk, int(c), j, bc[0] / bc[2], bc[1] / bc[2]}, t);
                    }
                    continue;
                }
                const auto pit = win->lm_point.find(j);
                if (pit != win->lm_point.end() && !in_front(pit->second, T_W_I)) {
                    track_to_lm.erase(tid);
                    win->lm_meas.erase(j); win->landmark_t.erase(j); win->lm_point.erase(j);
                    continue;
                }
                if (win->can_observe(b0[0], b0[1], b0[2]))
                    win->add_observation({kk, int(c0), j, b0[0] / b0[2], b0[1] / b0[2]}, t);
                for (size_t c = 0; c < cams.size(); ++c) {
                    if (c == c0) continue;
                    auto f = by_cam[c].find(tid);
                    if (f == by_cam[c].end()) continue;
                    const cv::Vec3d& bc = bearings[c][f->second];
                    if (win->can_observe(bc[0], bc[1], bc[2]))
                        win->add_observation({kk, int(c), j, bc[0] / bc[2], bc[1] / bc[2]}, t);
                }
            }
        }
    }

    // returns (ok, pose); keyframe flag via out param
    void finish_init(double t, const cv::Matx33d& Rwi, const Vector3& p0, const Vector3& v0,
                     double tilt_sigma, double vel_sigma,
                     const std::vector<CamOut>* feats_in, const std::vector<cv::Mat>& imgs) {
        bias = gtsam::imuBias::ConstantBias(Vector3::Zero(), gyro_bias);
        gtsam::Matrix3 Rm;
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) Rm(r, c) = Rwi(r, c);
        const gtsam::Pose3 T(gtsam::Rot3(Rm), gtsam::Point3(p0[0], p0[1], p0[2]));
        k = 0;
        win->initialize(0, t, T, v0, bias, tilt_sigma, vel_sigma);
        kf_navstate = gtsam::NavState(T, v0);
        pim = std::make_unique<gtsam::PreintegratedCombinedMeasurements>(pp, bias);
        pim_t_last = t;
        t_init = t;
        int n_new = 0;
        std::vector<CamOut> outs = feats_in ? *feats_in
                                 : (per_cam ? mfe.process(imgs, nullptr, n_new)
                                            : fe.process(imgs[0], imgs, nullptr, n_new));
        std::vector<std::vector<cv::Vec3d>> bs(outs.size());
        for (size_t c = 0; c < outs.size(); ++c) {
            std::vector<bool> v;
            bs[c] = fe.bearings(cam_of(c), outs[c].px, v);
        }
        add_landmarks(0, t, outs, bs, T);
        win->optimize(0, t);
        const auto& [ps, vs, bsv] = win->kf_state[0];
        kf_navstate = gtsam::NavState(ps, vs);
        t_prev_frame = t;
        frames_since_kf = 0;
        n_tracks_at_kf = 0;
        for (size_t c = 0; c < outs.size(); ++c) n_tracks_at_kf += int(outs[c].ids.size());
        if (!per_cam) n_tracks_at_kf = int(outs[0].ids.size());
        initialized = true;
    }

    bool pre_init_auto(double t, const std::vector<cv::Mat>& imgs, gtsam::Pose3& pose_out, bool& is_kf_out) {
        if (t_first_frame < 0) t_first_frame = t;
        // deferred warm-up: give still-detection its window first
        if (!init.done && (t - t_first_frame) < init.window_s + 0.15) return false;
        if (init.done && t_prev_frame < 0) {   // still start: bit-identical to static mode
            gyro_bias = init.gyro_bias;
            finish_init(t, init.R_W_I, Vector3::Zero(), Vector3::Zero(), 0.01, 0.1, nullptr, imgs);
            pose_out = std::get<0>(win->kf_state[0]); is_kf_out = true;
            return true;
        }
        // moving start: warm the front-end, collect metric structure, solve
        const cv::Matx33d dR = t_prev_frame < 0 ? cv::Matx33d::eye()
                              : delta_rotation(imu, t_prev_frame, t, Vector3::Zero());
        int n_new = 0;
        auto outs = per_cam ? mfe.process(imgs, t_prev_frame < 0 ? nullptr : &dR, n_new)
                            : fe.process(imgs[0], imgs, t_prev_frame < 0 ? nullptr : &dR, n_new);
        t_prev_frame = t;
        if (++init_frame_i % std::max(init_stride, 1) == 0) {
            std::vector<std::vector<cv::Vec3d>> bs(outs.size());
            for (size_t c = 0; c < outs.size(); ++c) {
                std::vector<bool> v;
                bs[c] = fe.bearings(cam_of(c), outs[c].px, v);
            }
            std::map<int64_t, cv::Vec3d> pts;
            matcher.match(imgs, outs, bs, pts);
            dyn.add(t, pts);
        }
        if (dyn.ready()) {
            cv::Matx33d Rwi; Vector3 vW, pW;
            if (dyn.solve(imu, Rwi, vW, pW)) {
                gyro_bias = Vector3::Zero();
                finish_init(t, Rwi, pW, vW, dyn_tilt_sigma, dyn_vel_sigma, &outs, imgs);
                std::printf("dynamic init: %zu frames, |v| %.2f m/s\n", dyn.frames.size(), vW.norm());
                pose_out = std::get<0>(win->kf_state[0]); is_kf_out = true;
                return true;
            }
        }
        if (t - t_first_frame > init_max_s) {   // starvation: legacy forced-static
            init.force();
            gyro_bias = init.gyro_bias;
            finish_init(t, init.R_W_I, Vector3::Zero(), Vector3::Zero(), 0.01, 0.1, &outs, imgs);
            pose_out = std::get<0>(win->kf_state[0]); is_kf_out = true;
            return true;
        }
        return false;
    }

    bool track(double t, const std::vector<cv::Mat>& imgs, gtsam::Pose3& pose_out, bool& is_kf_out) {
        is_kf_out = false;
        if (!initialized) {
            if (init_mode == "auto") return pre_init_auto(t, imgs, pose_out, is_kf_out);
            if (!init.done) return false;
            // _initialize
            gyro_bias = init.gyro_bias;
            bias = gtsam::imuBias::ConstantBias(Vector3::Zero(), gyro_bias);
            gtsam::Matrix3 Rm;
            for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) Rm(r, c) = init.R_W_I(r, c);
            const gtsam::Pose3 T(gtsam::Rot3(Rm), gtsam::Point3(0, 0, 0));
            k = 0;
            win->initialize(0, t, T, Vector3::Zero(), bias);
            kf_navstate = gtsam::NavState(T, Vector3::Zero());
            pim = std::make_unique<gtsam::PreintegratedCombinedMeasurements>(pp, bias);
            pim_t_last = t;
            t_init = t;
            int n_new = 0;
            auto outs = per_cam ? mfe.process(imgs, nullptr, n_new)
                                : fe.process(imgs[0], imgs, nullptr, n_new);
            std::vector<std::vector<cv::Vec3d>> bs(outs.size());
            for (size_t c = 0; c < outs.size(); ++c) {
                std::vector<bool> v;
                bs[c] = fe.bearings(cam_of(c), outs[c].px, v);
            }
            add_landmarks(0, t, outs, bs, T);
            win->optimize(0, t);
            const auto& [ps, vs, bsv] = win->kf_state[0];
            kf_navstate = gtsam::NavState(ps, vs);
            t_prev_frame = t;
            frames_since_kf = 0;
            n_tracks_at_kf = 0;
            for (size_t c = 0; c < outs.size(); ++c) n_tracks_at_kf += int(outs[c].ids.size());
            if (!per_cam) n_tracks_at_kf = int(outs[0].ids.size());
            initialized = true;
            pose_out = ps;
            is_kf_out = true;
            return true;
        }
        const cv::Matx33d dR = delta_rotation(imu, t_prev_frame, t, gyro_bias);
        int n_new = 0;
        auto outs = per_cam ? mfe.process(imgs, &dR, n_new)
                            : fe.process(imgs[0], imgs, &dR, n_new);
        t_prev_frame = t;
        ++frames_since_kf;
        integrate_until(t);
        const gtsam::NavState predicted = pim->predict(kf_navstate, bias);
        int n0 = int(outs[0].ids.size());
        if (per_cam) { n0 = 0; for (size_t c = 0; c < outs.size(); ++c) n0 += int(outs[c].ids.size()); }
        bool is_kf = frames_since_kf >= kf_every || n0 < kf_min_track_ratio * std::max(n_tracks_at_kf, 1);
        if (kf_dense_init_s > 0 && t_init >= 0 && (t - t_init) < kf_dense_init_s) is_kf = true;
        if (!is_kf) { pose_out = predicted.pose(); return true; }
        // _keyframe
        ++k;
        win->add_keyframe(k, t, *pim, predicted, bias);
        std::vector<std::vector<cv::Vec3d>> bs(outs.size());
        for (size_t c = 0; c < outs.size(); ++c) {
            std::vector<bool> v;
            bs[c] = fe.bearings(cam_of(c), outs[c].px, v);
        }
        add_landmarks(k, t, outs, bs, predicted.pose());
        win->optimize(k, t);
        const auto& [ps, vs, bsv] = win->kf_state[k];
        bias = bsv;
        gyro_bias = bias.gyroscope();
        kf_navstate = gtsam::NavState(ps, vs);
        pim = std::make_unique<gtsam::PreintegratedCombinedMeasurements>(pp, bias);
        pim_t_last = t;
        frames_since_kf = 0;
        n_tracks_at_kf = n0;
        pose_out = ps;
        is_kf_out = true;
        return true;
    }
};

#ifndef PODSLAM_TRACKER_NO_MAIN
int main(int argc, char** argv) {
    cv::setNumThreads(1);
    const char* dir = argc > 1 ? argv[1] : "/tmp/frontend_golden";
    const char* rig = argc > 2 ? argv[2] : "podslam-cpp/tests/data/backend_golden.txt";
    const int kf_every_arg = argc > 3 ? std::atoi(argv[3]) : 0;
    const bool multiklt = argc > 4 && std::string(argv[4]) == "multiklt";

    Tracker tr;
    if (!load_rig_lines(rig, tr.fe.cams, tr.fe.T_imu_cam)) { std::printf("no rig\n"); return 1; }
    tr.rig_rows.clear();
    std::vector<double> fovs;
    {   // reload R + N (+ optional F fov) lines for the window/imu parameters
        std::ifstream in(rig);
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ss(line);
            std::string tag; ss >> tag;
            if (tag == "R") { std::array<double, 20> r{}; for (auto& x : r) ss >> x; tr.rig_rows.push_back(r); }
            else if (tag == "N") {
                double v[10]; for (auto& x : v) ss >> x;
                tr.imu_noise = {v[0], v[1], v[2], v[3]};
                tr.accel_scale = v[4];
                tr.noise_scale = {v[5], v[6], v[7], v[8]};
                tr.px_sigma = v[9];
            }
            else if (tag == "F") { double v; while (ss >> v) fovs.push_back(v); }
        }
    }
    if (multiklt) {
        // image dims from the first frames.bin record (needed for the circle masks)
        std::ifstream peek(std::string(dir) + "/frames.bin", std::ios::binary);
        int64_t tns0; int32_t nc0, h0, w0;
        if (!(peek.read(reinterpret_cast<char*>(&tns0), 8) &&
              peek.read(reinterpret_cast<char*>(&nc0), 4) &&
              peek.read(reinterpret_cast<char*>(&h0), 4) &&
              peek.read(reinterpret_cast<char*>(&w0), 4))) { std::printf("no frames.bin\n"); return 1; }
        tr.mfe.init(tr.fe.cams, tr.fe.T_imu_cam, fovs, w0, h0);
        tr.per_cam = true;
        std::printf("multiklt: %zu per-camera trackers, %zu circle masks\n",
                    tr.mfe.subs.size(), fovs.size());
    }
    if (kf_every_arg > 0) tr.kf_every = kf_every_arg;
    if (const char* e = std::getenv("PODSLAM_INIT_MODE")) tr.init_mode = e;
    if (tr.init_mode == "auto") tr.init.max_wait_s = 1e12;
    if (const char* e = std::getenv("PODSLAM_DW"))    tr.setup_dw = std::atoi(e) != 0;
    if (const char* e = std::getenv("PODSLAM_DENSE")) tr.kf_dense_init_s = std::atof(e);
    tr.setup();
    if (tr.setup_dw) { tr.win->dyn_weight = true; std::printf("dyn-weight ON\n"); }
    tr.matcher.cams = &tr.fe.cams; tr.matcher.Ts = &tr.fe.T_imu_cam;
    if (tr.init_mode == "auto") std::printf("init-mode auto\n");
    if (tr.kf_dense_init_s > 0) std::printf("kf-dense-init %.1f s\n", tr.kf_dense_init_s);

    // IMU rows (t_ns gx gy gz ax ay az) and frames.bin interleaved by time
    struct ImuRow { double t; Vector3 w, a; };
    std::vector<ImuRow> imu_rows;
    {
        std::ifstream fi(std::string(dir) + "/imu.txt");
        int64_t tns; double gx, gy, gz, ax, ay, az;
        while (fi >> tns >> gx >> gy >> gz >> ax >> ay >> az)
            imu_rows.push_back({tns * 1e-9, Vector3(gx, gy, gz), Vector3(ax, ay, az)});
    }
    std::ifstream fb(std::string(dir) + "/frames.bin", std::ios::binary);
    if (!fb || imu_rows.empty()) { std::printf("missing golden data in %s\n", dir); return 1; }
    std::ofstream et(std::string(dir) + "/cpp_est.tum");
    size_t imu_i = 0;
    int frames = 0, tracked = 0, kfs = 0;
    const auto t_start = std::chrono::steady_clock::now();
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
        fb.read(reinterpret_cast<char*>(dRv), 72);   // recomputed internally; skip
        const double t = t_ns * 1e-9;
        while (imu_i < imu_rows.size() && imu_rows[imu_i].t <= t) {
            tr.register_imu(imu_rows[imu_i].t, imu_rows[imu_i].w, imu_rows[imu_i].a);
            ++imu_i;
        }
        gtsam::Pose3 pose; bool is_kf = false;
        const bool ok = tr.track(t, imgs, pose, is_kf);
        ++frames;
        if (ok) {
            ++tracked; kfs += is_kf ? 1 : 0;
            const auto q = pose.rotation().toQuaternion();
            char buf[256];
            std::snprintf(buf, sizeof buf, "%.6f %f %f %f %.7f %.7f %.7f %.7f\n", t,
                          pose.translation().x(), pose.translation().y(), pose.translation().z(),
                          q.x(), q.y(), q.z(), q.w());
            et << buf;
        }
    }
    et.close();
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::printf("wall %.1f s, %.1f ms/frame\n", wall, 1000.0 * wall / std::max(1, frames));
    std::printf("tracked %d/%d frames, %d keyframes; window stats: absorbed %d outliers %d marginalised %d failed %d\n",
                tracked, frames, kfs, tr.win->n_absorbed, tr.win->n_outliers, tr.win->n_marginalized, tr.win->n_failed);

    // compare with python_est.tum by timestamp
    std::ifstream pe(std::string(dir) + "/python_est.tum");
    std::ifstream ce(std::string(dir) + "/cpp_est.tum");
    std::map<long long, std::array<double, 3>> py;
    double t0 = 0; double x, y, z, qx, qy, qz, qw;
    while (pe >> t0 >> x >> y >> z >> qx >> qy >> qz >> qw)
        py[llround(t0 * 1e4)] = {x, y, z};
    int n = 0; double sum2 = 0, maxd = 0;
    while (ce >> t0 >> x >> y >> z >> qx >> qy >> qz >> qw) {
        auto it = py.find(llround(t0 * 1e4));
        if (it == py.end()) continue;
        const double d = std::sqrt(std::pow(x - it->second[0], 2) + std::pow(y - it->second[1], 2) + std::pow(z - it->second[2], 2));
        sum2 += d * d; maxd = std::max(maxd, d);
        ++n;
    }
    std::printf("trajectory vs python over %d shared frames: rmse %.4f m, max %.4f m\n",
                n, n ? std::sqrt(sum2 / n) : -1.0, maxd);
    // The mutual C++/Python distance is informational only: both trajectories drift
    // from ground truth by ~9 cm on a full flight and their front-end divergence is
    // chaotic (documented), so the real acceptance gate is ATE vs GT computed by
    // bench/evaluate.py on cpp_est.tum (full room1 day: C++ 7.7 cm vs Python 8.8 cm).
    const int rc = (n > 100 && std::sqrt(sum2 / std::max(1, n)) < 0.15) ? 0 : 2;
    std::puts(rc == 0 ? "tracker end-to-end: OK (mutual distance within the chaotic band; gate on ATE-vs-GT externally)" : "tracker end-to-end: CHECK");
    return rc;
}
#endif
