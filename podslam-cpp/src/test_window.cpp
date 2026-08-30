// Port step 2: the sliding-window backend (podslam/backend_smart.py) on native GTSAM,
// verified by replaying the golden dump (podslam-cpp/tools/dump_backend_golden.py)
// and comparing every keyframe's solved state with the Python oracle.
//
// The transcription mirrors backend_smart.py exactly:
//   - smart rig factors (PinholePose<Cal3_S2>, unit K, normalized measurements),
//     HESSIAN linearization, ZERO_ON_DEGENERACY, rank tol 1e-9, distance/outlier
//     thresholds disabled, EPI off
//   - CombinedImuFactor between consecutive keyframes (per-frame integration with
//     sample-holding, exactly like podslam/imu.py integrate_until)
//   - hard gauge anchor on the first pose (pos/yaw 1e-3, tilt 0.01)
//   - LM with absolute stop (rel 0, abs 1e-2, 6 iterations)
//   - chi2 gate 5.0 per measurement at the converged solution
//   - marginalisation mode "all": every landmark seen from a dropped keyframe is
//     absorbed into the prior (>= 4 obs, gate passed) via partial elimination,
//     kept as LinearContainerFactors
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/CameraSet.h>
#include <gtsam/geometry/PinholePose.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/SmartProjectionRigFactor.h>
#include <gtsam/inference/Symbol.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;
using Camera = gtsam::PinholePose<gtsam::Cal3_S2>;
using SmartRig = gtsam::SmartProjectionRigFactor<Camera>;

struct Obs { int k, cam; long j; double mx, my; };
struct KfRec { int k; double t; double T[12]; double vel[3]; double bias[6]; std::vector<Obs> obs; };

struct Window {
    // ---- configuration (mirrors backend_smart defaults)
    double lag_s = 4.0, chi2_gate = 5.0, abs_err_tol = 1e-2;
    int max_iters = 6, min_obs_prior = 4, max_window_kf = 32;

    std::shared_ptr<gtsam::CameraSet<Camera>> cam_set;
    gtsam::SharedNoiseModel noise;
    gtsam::SmartProjectionParams sp;

    std::map<int, double> kf_t;
    std::map<int, std::tuple<gtsam::Pose3, gtsam::Vector3, gtsam::imuBias::ConstantBias>> kf_state;
    std::map<int, gtsam::CombinedImuFactor::shared_ptr> imu_factor;
    std::map<long, std::vector<Obs>> lm_meas;
    std::map<long, double> landmark_t;
    std::vector<gtsam::NonlinearFactor::shared_ptr> prior_factors;
    std::set<int> graph_keys;
    int n_absorbed = 0, n_rejected = 0, n_outliers = 0, n_marginalized = 0, n_failed = 0;

    Window(const std::vector<std::array<double, 20>>& rig_rows, double px_sigma)
        : sp(gtsam::HESSIAN, gtsam::ZERO_ON_DEGENERACY) {
        cam_set = std::make_shared<gtsam::CameraSet<Camera>>();
        double sig_sum = 0;
        auto K = std::make_shared<gtsam::Cal3_S2>(1.0, 1.0, 0.0, 0.0, 0.0);
        for (const auto& r : rig_rows) {
            gtsam::Matrix4 T = gtsam::Matrix4::Identity();
            for (int i = 0; i < 12; ++i) T(i / 4, i % 4) = r[8 + i];
            cam_set->push_back(Camera(gtsam::Pose3(T), K));
            sig_sum += px_sigma / r[0];
        }
        noise = gtsam::noiseModel::Isotropic::Sigma(2, sig_sum / rig_rows.size());
        sp.setRankTolerance(1e-9);
        sp.setLandmarkDistanceThreshold(-1.0);
        sp.setDynamicOutlierRejectionThreshold(-1.0);
    }

    SmartRig::shared_ptr smart_factor(const std::vector<Obs>& meas) const {
        auto f = std::make_shared<SmartRig>(noise, cam_set, sp);
        for (const auto& o : meas) f->add(gtsam::Point2(o.mx, o.my), X(o.k), size_t(o.cam));
        return f;
    }

    gtsam::Values values_for(const std::vector<int>& keys) const {
        gtsam::Values v;
        for (int k : keys) {
            const auto& [p, vel, b] = kf_state.at(k);
            v.insert(X(k), p); v.insert(V(k), vel); v.insert(B(k), b);
        }
        return v;
    }

    void initialize(int k, double t, const gtsam::Pose3& pose, const gtsam::Vector3& vel,
                    const gtsam::imuBias::ConstantBias& bias) {
        kf_t[k] = t; kf_state[k] = {pose, vel, bias}; graph_keys.insert(k);
        gtsam::Vector6 sp_;                       // rot(tilt,tilt,yaw), trans(pos)
        sp_ << 0.01, 0.01, 1e-3, 1e-3, 1e-3, 1e-3;
        prior_factors.push_back(std::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            X(k), pose, gtsam::noiseModel::Diagonal::Sigmas(sp_)));
        prior_factors.push_back(std::make_shared<gtsam::PriorFactor<gtsam::Vector3>>(
            V(k), vel, gtsam::noiseModel::Isotropic::Sigma(3, 0.1)));
        gtsam::Vector6 sb; sb << 0.1, 0.1, 0.1, 0.01, 0.01, 0.01;
        prior_factors.push_back(std::make_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
            B(k), bias, gtsam::noiseModel::Diagonal::Sigmas(sb)));
    }

    void add_keyframe(int k, double t, const gtsam::PreintegratedCombinedMeasurements& pim,
                      const gtsam::NavState& predicted, const gtsam::imuBias::ConstantBias& bias_prev) {
        kf_t[k] = t; graph_keys.insert(k);
        kf_state[k] = {predicted.pose(), predicted.velocity(), bias_prev};
        imu_factor[k] = std::make_shared<gtsam::CombinedImuFactor>(
            X(k - 1), V(k - 1), X(k), V(k), B(k - 1), B(k), pim);
    }

    void add_observation(const Obs& o, double t) {
        lm_meas[o.j].push_back(o);
        landmark_t[o.j] = t;
    }

    void marginalize(const std::vector<int>& gone, const std::vector<int>& remaining, double t_now) {
        std::set<int> gset(gone.begin(), gone.end());
        std::set<gtsam::Key> keys_m;
        for (int k : gone) { keys_m.insert(X(k)); keys_m.insert(V(k)); keys_m.insert(B(k)); }
        std::vector<int> all(gone); all.insert(all.end(), remaining.begin(), remaining.end());
        gtsam::Values values = values_for(all);
        gtsam::GaussianFactorGraph gfg;
        std::vector<gtsam::NonlinearFactor::shared_ptr> kept;
        for (const auto& f : prior_factors) {
            bool touches = false;
            for (auto key : f->keys()) if (keys_m.count(key)) { touches = true; break; }
            if (touches) gfg.push_back(f->linearize(values)); else kept.push_back(f);
        }
        for (int kk : all) {
            auto it = imu_factor.find(kk);
            if (it == imu_factor.end()) continue;
            if ((gset.count(kk) || gset.count(kk - 1)) && kf_state.count(kk - 1))
                gfg.push_back(it->second->linearize(values));
        }
        std::set<int> rset(remaining.begin(), remaining.end());
        std::vector<long> consumed;
        for (auto& [j, meas] : lm_meas) {
            bool touches = false;
            for (const auto& o : meas) if (gset.count(o.k)) { touches = true; break; }
            if (!touches) continue;
            std::vector<Obs> inside;
            for (const auto& o : meas) if (gset.count(o.k) || rset.count(o.k)) inside.push_back(o);
            if (int(inside.size()) >= std::max(2, min_obs_prior)) {
                auto f = smart_factor(inside);
                try {
                    if (f->error(values) / inside.size() <= chi2_gate) {
                        auto lin = f->linearize(values);
                        if (lin) { gfg.push_back(lin); ++n_absorbed; }
                    } else ++n_rejected;
                } catch (...) { ++n_rejected; }
            }
            consumed.push_back(j);
        }
        std::set<gtsam::Key> present;
        for (const auto& f : gfg) for (auto key : f->keys()) present.insert(key);
        gtsam::Ordering order;
        for (int k : gone) for (auto key : {X(k), V(k), B(k)}) if (present.count(key)) order.push_back(key);
        prior_factors = kept;
        if (order.size() > 0) {
            auto [bn, rem] = gfg.eliminatePartialMultifrontal(order);
            (void)bn;
            for (const auto& lf : *rem)
                if (lf && lf->size() > 0)
                    prior_factors.push_back(std::make_shared<gtsam::LinearContainerFactor>(lf, values));
        }
        for (long j : consumed) { lm_meas.erase(j); landmark_t.erase(j); }
        for (int k : gone) {
            graph_keys.erase(k); kf_state.erase(k); imu_factor.erase(k); kf_t.erase(k);
        }
        n_marginalized += int(gone.size());
        (void)t_now;
    }

    void optimize(int k, double t) {
        std::vector<int> window(graph_keys.begin(), graph_keys.end());
        std::set<int> kset(window.begin(), window.end());
        gtsam::NonlinearFactorGraph graph;
        gtsam::Values values = values_for(window);
        int k0 = window.front();
        for (int kk : window)
            if (kk != k0 && imu_factor.count(kk) && kset.count(kk - 1)) graph.push_back(imu_factor[kk]);
        for (const auto& f : prior_factors) graph.push_back(f);
        std::map<long, SmartRig::shared_ptr> factors;
        for (auto& [j, meas] : lm_meas) {
            std::vector<Obs> inwin;
            for (const auto& o : meas) if (kset.count(o.k)) inwin.push_back(o);
            if (inwin.size() < 2) continue;
            auto f = smart_factor(inwin);
            graph.push_back(f); factors[j] = f;
        }
        gtsam::LevenbergMarquardtParams params;
        params.setMaxIterations(max_iters);
        params.setRelativeErrorTol(0.0);
        params.setAbsoluteErrorTol(abs_err_tol);
        params.setVerbosityLM("SILENT");
        bool ok = true;
        gtsam::Values result;
        try {
            result = gtsam::LevenbergMarquardtOptimizer(graph, values, params).optimize();
        } catch (...) { ok = false; ++n_failed; }
        if (ok) {
            for (int kk : window)
                kf_state[kk] = {result.at<gtsam::Pose3>(X(kk)), result.at<gtsam::Vector3>(V(kk)),
                                result.at<gtsam::imuBias::ConstantBias>(B(kk))};
            std::vector<long> outliers;
            for (auto& [j, f] : factors) {
                try {
                    if (f->error(result) / std::max<size_t>(1, f->measured().size()) > chi2_gate)
                        outliers.push_back(j);
                } catch (...) {}
            }
            for (long j : outliers) { lm_meas.erase(j); landmark_t.erase(j); }
            n_outliers += int(outliers.size());
        }
        // slide
        double cutoff = t - lag_s;
        std::vector<int> gone, remaining;
        int n_over = std::max(0, int(window.size()) - max_window_kf);
        for (size_t i = 0; i < window.size(); ++i) {
            if (kf_t[window[i]] < cutoff || int(i) < n_over) gone.push_back(window[i]);
            else remaining.push_back(window[i]);
        }
        if (!gone.empty() && !remaining.empty()) marginalize(gone, remaining, t);
        std::vector<long> stale;
        for (auto& [j, tj] : landmark_t) if (tj < cutoff) stale.push_back(j);
        for (long j : stale) { lm_meas.erase(j); landmark_t.erase(j); }
        std::set<int> rset2(graph_keys.begin(), graph_keys.end());
        for (auto& [j, meas] : lm_meas) {
            std::vector<Obs> keep2;
            for (const auto& o : meas) if (rset2.count(o.k)) keep2.push_back(o);
            meas.swap(keep2);
        }
    }
};

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "podslam-cpp/tests/data/backend_golden.txt";
    const double lag_override = argc > 2 ? std::atof(argv[2]) : 0.0;
    const bool parity_iters = argc > 3 && std::atoi(argv[3]) != 0;
    std::ifstream in(path);
    if (!in) { std::printf("cannot open %s\n", path); return 1; }
    std::vector<std::array<double, 20>> rig_rows;
    double imu_p[10] = {0};
    struct ImuRow { double t, w[3], a[3]; };
    std::vector<ImuRow> imu_rows;
    std::vector<double> frames;
    std::vector<KfRec> kfs;
    bool have_init = false;
    double init_state[22];   // t, T(12), vel(3), bias(6)
    std::vector<Obs> pending;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tag; ss >> tag;
        if (tag == "R") { std::array<double, 20> r{}; for (auto& x : r) ss >> x; rig_rows.push_back(r); }
        else if (tag == "N") { for (auto& x : imu_p) ss >> x; }
        else if (tag == "I") { ImuRow r{}; ss >> r.t >> r.w[0] >> r.w[1] >> r.w[2] >> r.a[0] >> r.a[1] >> r.a[2]; imu_rows.push_back(r); }
        else if (tag == "F") { double t; ss >> t; frames.push_back(t); }
        else if (tag == "S") { for (auto& x : init_state) ss >> x; have_init = true; }
        else if (tag == "O") { Obs o{}; ss >> o.k >> o.cam >> o.j >> o.mx >> o.my; pending.push_back(o); }
        else if (tag == "K") {
            KfRec kf{}; ss >> kf.k >> kf.t;
            for (auto& x : kf.T) ss >> x;
            for (auto& x : kf.vel) ss >> x;
            for (auto& x : kf.bias) ss >> x;
            kf.obs.swap(pending);
            kfs.push_back(std::move(kf));
        }
    }
    std::printf("golden: %zu cams, %zu imu, %zu frames, %zu keyframes, init %s\n",
                rig_rows.size(), imu_rows.size(), frames.size(), kfs.size(), have_init ? "yes" : "NO");
    if (!have_init || kfs.empty()) return 1;

    Window win(rig_rows, imu_p[9]);
    if (lag_override > 0) win.lag_s = lag_override;
    if (lag_override > 0) win.max_window_kf = 100000;
    if (parity_iters) win.abs_err_tol = 0.0;
    // IMU preintegration params (mirrors podslam/imu.py; accel_scale already applied in the dump)
    auto pp = gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedU(9.81);
    const double kg = imu_p[6], ka = imu_p[5], kaw = imu_p[7], kgw = imu_p[8];
    pp->setGyroscopeCovariance(gtsam::I_3x3 * std::pow(kg * imu_p[0], 2));
    pp->setAccelerometerCovariance(gtsam::I_3x3 * std::pow(ka * imu_p[2], 2));
    pp->setIntegrationCovariance(gtsam::I_3x3 * 1e-8);
    pp->setBiasAccCovariance(gtsam::I_3x3 * std::pow(kaw * imu_p[3], 2));
    pp->setBiasOmegaCovariance(gtsam::I_3x3 * std::pow(kgw * imu_p[1], 2));

    gtsam::Matrix4 T0 = gtsam::Matrix4::Identity();
    for (int i = 0; i < 12; ++i) T0(i / 4, i % 4) = init_state[1 + i];
    gtsam::Vector3 v0(init_state[13], init_state[14], init_state[15]);
    gtsam::Vector6 bv0; for (int i = 0; i < 6; ++i) bv0(i) = init_state[16 + i];
    gtsam::imuBias::ConstantBias bias(bv0);
    win.initialize(kfs[0].k, kfs[0].t, gtsam::Pose3(T0), v0, bias);
    for (const auto& o : kfs[0].obs) win.add_observation(o, kfs[0].t);
    win.optimize(kfs[0].k, kfs[0].t);

    auto& [p0, vel0, b0] = win.kf_state[kfs[0].k];
    gtsam::NavState nav(p0, vel0);
    gtsam::imuBias::ConstantBias cur_bias = b0;
    gtsam::PreintegratedCombinedMeasurements pim(pp, cur_bias);
    double t_last = kfs[0].t;
    size_t imu_i = 0, frame_i = 0;
    while (frame_i < frames.size() && frames[frame_i] <= kfs[0].t + 1e-9) ++frame_i;

    auto integrate_until = [&](double t1) {
        // samples with t_prev < t <= t1, dt from the running t_last (podslam/imu.py)
        while (imu_i < imu_rows.size() && imu_rows[imu_i].t <= t_last + 1e-12) ++imu_i;
        size_t last_used = size_t(-1);
        for (size_t i = imu_i; i < imu_rows.size() && imu_rows[i].t <= t1 + 1e-12; ++i) {
            double dt = imu_rows[i].t - t_last;
            if (dt > 0) {
                pim.integrateMeasurement(gtsam::Vector3(imu_rows[i].a[0], imu_rows[i].a[1], imu_rows[i].a[2]),
                                         gtsam::Vector3(imu_rows[i].w[0], imu_rows[i].w[1], imu_rows[i].w[2]), dt);
                t_last = imu_rows[i].t;
                last_used = i;
            }
            imu_i = i + 1;
        }
        if (t1 > t_last) {
            size_t i = last_used != size_t(-1) ? last_used : (imu_i > 0 ? imu_i - 1 : size_t(-1));
            if (i != size_t(-1)) {
                pim.integrateMeasurement(gtsam::Vector3(imu_rows[i].a[0], imu_rows[i].a[1], imu_rows[i].a[2]),
                                         gtsam::Vector3(imu_rows[i].w[0], imu_rows[i].w[1], imu_rows[i].w[2]), t1 - t_last);
                t_last = t1;
            }
        }
    };

    double max_pos = 0, max_rot = 0, max_vel = 0, max_bias = 0;
    for (size_t n = 1; n < kfs.size(); ++n) {
        const auto& kf = kfs[n];
        while (frame_i < frames.size() && frames[frame_i] <= kf.t + 1e-9) {
            integrate_until(frames[frame_i]);
            ++frame_i;
        }
        if (t_last < kf.t) integrate_until(kf.t);
        gtsam::NavState pred = pim.predict(nav, cur_bias);
        win.add_keyframe(kf.k, kf.t, pim, pred, cur_bias);
        for (const auto& o : kf.obs) win.add_observation(o, kf.t);
        win.optimize(kf.k, kf.t);
        auto& [ps, vs, bs] = win.kf_state[kf.k];
        gtsam::Matrix4 Tg = gtsam::Matrix4::Identity();
        for (int i = 0; i < 12; ++i) Tg(i / 4, i % 4) = kf.T[i];
        gtsam::Pose3 gold(Tg);
        const double dp = (ps.translation() - gold.translation()).norm();
        const double dr = gtsam::Rot3::Logmap(ps.rotation().between(gold.rotation())).norm();
        const double dv = (vs - gtsam::Vector3(kf.vel[0], kf.vel[1], kf.vel[2])).norm();
        gtsam::Vector6 gb; for (int i = 0; i < 6; ++i) gb(i) = kf.bias[i];
        const double db = (bs.vector() - gb).norm();
        if (n <= 10 || dp > std::max(1e-5, max_pos) || (n % 20 == 0))
            std::printf("  kf %3zu: dpos %.3g drot %.3g dvel %.3g dbias %.3g\n", n, dp, dr, dv, db);
        max_pos = std::max(max_pos, dp); max_rot = std::max(max_rot, dr);
        max_vel = std::max(max_vel, dv); max_bias = std::max(max_bias, db);
        nav = gtsam::NavState(ps, vs);
        cur_bias = bs;
        pim = gtsam::PreintegratedCombinedMeasurements(pp, cur_bias);
        t_last = kf.t;
    }
    std::printf("window parity over %zu keyframes: max |pos| %.3g m, |rot| %.3g rad, |vel| %.3g, |bias| %.3g\n",
                kfs.size() - 1, max_pos, max_rot, max_vel, max_bias);
    std::printf("stats: absorbed %d rejected %d outliers %d marginalised %d failed %d\n",
                win.n_absorbed, win.n_rejected, win.n_outliers, win.n_marginalized, win.n_failed);
    const int rc = (max_pos < 1e-4 && max_rot < 1e-4) ? 0 : 2;
    std::puts(rc == 0 ? "window parity: OK" : "window parity: FAILED");
    return rc;
}
