#include "hfocus.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <stdexcept>
#include <string>
#include "fsutil.hpp"
#include "rz.hpp"
#include "sim.hpp"

// Butterworth edges by the bilinear transform, prewarped at the edge (K = tan(omega_c dt / 2)); section k of an edge
// of n poles has Q = 1 / (2 cos(pi (2k + 1) / (2n))). High-pass s^2 / (s^2 + s/Q + 1), low-pass 1 / (...).
HFFilter hfocus_filter(double n_min, double n_max, int order, double dt) {
    HFFilter f;
    for (int e = 0; e < 2; ++e) {
        const double w = e == 0 ? n_min : n_max;
        if (!(w > 0)) continue;                        // n_min = 0: no high-pass; n_max = 0: no low-pass
        const double K = std::tan(0.5 * w * dt);
        for (int k = 0; k < order / 2; ++k) {
            const double Q = 1 / (2 * std::cos(M_PI * (2 * k + 1) / (2.0 * order)));
            const double nrm = 1 / (1 + K / Q + K * K);
            const int s = f.ns++;
            if (e == 0) { f.b0[s] = nrm; f.b1[s] = -2 * nrm; f.b2[s] = nrm; }
            else { f.b0[s] = K * K * nrm; f.b1[s] = 2 * K * K * nrm; f.b2[s] = K * K * nrm; }
            f.a1[s] = 2 * (K * K - 1) * nrm;
            f.a2[s] = (1 - K / Q + K * K) * nrm;
        }
    }
    return f;
}

double hfocus_gain(const HFFilter& f, double omega, double dt) {
    const std::complex<double> z1 = std::exp(std::complex<double>(0, -omega * dt));   // z^-1
    std::complex<double> h = 1;
    for (int s = 0; s < f.ns; ++s) h *= (f.b0[s] + z1 * (f.b1[s] + z1 * f.b2[s])) / (1.0 + z1 * (f.a1[s] + z1 * f.a2[s]));
    return std::abs(h);
}

// ---------------------------------------------------------------- kernels (free functions)
// One sample x of signal c at node q (n nodes) through the cascade; the states st[((c ns + s) 2 + w) n + q].
HD inline double hf_cascade(const HFFilter& f, double x, double* st, int c, long q, long n) {
    for (int s = 0; s < f.ns; ++s) {
        double* s1 = st + (long(c * f.ns + s) * 2) * n + q;
        double* s2 = s1 + n;
        const double y = f.b0[s] * x + *s1;
        *s1 = f.b1[s] * x - f.a1[s] * y + *s2;
        *s2 = f.b2[s] * x - f.a2[s] * y;
        x = y;
    }
    return x;
}
// out[(3 a + 0) n + q]: the peak of |E_band|^2 at angle a, + 1: its time, + 2: the fluence
HD inline void hf_record(double* out, int a, long q, long n, double I, double t, double dt) {
    double* o = out + long(3 * a) * n + q;
    if (I > o[0]) { o[0] = I; o[n] = t; }
    o[2 * n] += I * dt;
}

// Cart3D (and 2D): E at the node (i, j, k): Ex there, Ey and Ez averaged from j -+ 1/2, k -+ 1/2 (2D: Ez itself).
void hfocus_step_3d(const FV Ex, const FV Ey, const FV Ez, const HFFilter f, int i0, int j0, int k0, int ni, int nj, int nk,
                    double* st, double* out, double t, double dt) {
    const long n = long(ni) * nj * nk;
    par_for(n, [=] HD(long q) {
        const int i = i0 + int(q % ni), j = j0 + int((q / ni) % nj), k = k0 + int(q / (long(ni) * nj));
        const double e[3] = {Ex(i, j, k), 0.5 * (Ey(i, j - 1, k) + Ey(i, j, k)), 0.5 * (Ez(i, j, k - 1) + Ez(i, j, k))};
        double I = 0;
        for (int c = 0; c < 3; ++c) {
            const double y = hf_cascade(f, e[c], st, c, q, n);
            I += y * y;
        }
        hf_record(out, 0, q, n, I, t, dt);
    });
}

// RZ: (E_z, E_r, E_phi) at the node (i, r_j) at phi = 0 and pi/2 from the S mode slots (E_r averaged from the half
// nodes; on the axis E_r of mode m at -r_1/2 is (-1)^(m+1) E_r(r_1/2)); F(phi) = F_0 + 2 sum_m (Re F_m cos m phi -
// Im F_m sin m phi). On the axis the transverse field is Cartesian: E_x = 2 Re E_r,1, E_y = -2 Im E_r,1 at both angles.
void hfocus_step_rz(const FV Ez, const FV Er, const FV Ep, int S, const HFFilter f, int i0, int ni, int nj, double* st,
                    double* out, double t, double dt) {
    const long n = long(ni) * nj;
    par_for(n, [=] HD(long q) {
        const int i = i0 + int(q % ni), j = int(q / ni);
        double e[6] = {0, 0, 0, 0, 0, 0};          // phi = 0: z, r, phi; phi = pi/2: z, r, phi
        for (int k = 0; k < S; ++k) {
            const int m = (k + 1) / 2;
            const bool im = k > 0 && k % 2 == 0;
            const double ez = Ez(i, j, k), ep = Ep(i, j, k);
            const double er = j > 0 ? 0.5 * (Er(i, j - 1, k) + Er(i, j, k)) : (m % 2 == 1 ? Er(i, 0, k) : 0.0);
            const double c9 = m % 4 == 0 ? 1.0 : (m % 4 == 2 ? -1.0 : 0.0), s9 = m % 4 == 1 ? 1.0 : (m % 4 == 3 ? -1.0 : 0.0);
            const double w0 = k == 0 ? 1.0 : (im ? 0.0 : 2.0), w9 = k == 0 ? 1.0 : 2 * (im ? -s9 : c9);
            e[0] += w0 * ez; e[1] += w0 * er; e[2] += w0 * ep;
            e[3] += w9 * ez; e[4] += w9 * er; e[5] += w9 * ep;
        }
        if (j == 0) {
            const double ex = S > 1 ? 2 * Er(i, 0, 1) : 0.0, ey = S > 2 ? -2 * Er(i, 0, 2) : 0.0;
            e[1] = e[4] = ex; e[2] = e[5] = ey;
        }
        for (int a = 0; a < 2; ++a) {
            double I = 0;
            for (int c = 0; c < 3; ++c) {
                const double y = hf_cascade(f, e[3 * a + c], st, 3 * a + c, q, n);
                I += y * y;
            }
            hf_record(out, a, q, n, I, t, dt);
        }
    });
}

namespace {
class HFocus : public Operator {
public:
    explicit HFocus(const HFocusConfig& c) : c_(c) {}
    const char* name() const override { return "harmonic_focus"; }
    Stage stage() const override { return Stage::Fields; }     // after the field step: E at t_{n+1}

    void setup(Simulation& s) override {
        const HFocusConfig& H = c_;
        if (s.cfg.window.enabled) throw std::runtime_error("harmonic_focus: not with a moving window");
        if (s.restart_step >= 0)
            throw std::runtime_error("harmonic_focus: the filter states are not in the checkpoint; run without a restart");
        const Domain& d = s.dom;
        rz_ = s.cfg.rz;
        f_ = hfocus_filter(H.n_min, H.n_max, H.order, d.dt());
        // the region: global node ranges [g0, g1) per axis, intersected with the owned cells
        const int na = rz_ ? 2 : 3;
        long np = 1;
        for (int a = 0; a < 3; ++a) {
            int g0 = 0, g1 = d.N[a];
            if (a < na) {
                if (rz_ && a == 1) {                    // the radial nodes r_j <= rmax
                    const RadialGrid& g = s.rz->grid();
                    g1 = 0;
                    while (g1 < d.N[1] && g.r[size_t(g1)] <= H.hi[1] + 1e-12) ++g1;
                } else {
                    g0 = std::max(0, int(std::ceil(H.lo[a] / d.h[a] - 1e-9)));
                    g1 = int(std::min<double>(d.N[a], std::floor(H.hi[a] / d.h[a] + 1e-9) + 1));
                }
            }
            if (g1 <= g0) throw std::runtime_error("harmonic_focus: the region holds no node");
            const int l0 = std::max(g0, d.off[a]), l1 = std::min(g1, d.off[a] + d.n[a]);
            i0_[a] = l0 - d.off[a]; n_[a] = std::max(0, l1 - l0);
            np *= n_[a];
        }
        np_ = rz_ ? long(n_[0]) * n_[1] : np;
        const int nsig = rz_ ? 6 : 3, nang = rz_ ? 2 : 1;
        nq_ = 3 * nang;
        if (np_ > 0) {
            st_.resize(size_t(nsig) * f_.ns * 2 * np_);
            out_.resize(size_t(nq_) * np_);
        }
        dir_ = s.cfg.dir + "/harmonic_focus" + (H.name.empty() ? "" : "_" + H.name);
        make_dirs(dir_);
        if (np_ > 0) {                                 // the layout of this rank's part
            FILE* m = std::fopen((dir_ + "/meta_r" + std::to_string(s.comm.rank) + ".txt").c_str(), "w");
            if (!m) throw std::runtime_error("cannot open " + dir_ + "/meta_r*.txt");
            std::fprintf(m, "geometry %s\nband %.17g %.17g\norder %d\ndt %.17g\nh %.17g %.17g %.17g\nlambda0 %.17g\n",
                         rz_ ? "rz" : (d.flat ? "cartesian2d" : "cartesian"), H.n_min, H.n_max, H.order, d.dt(), d.h[0], d.h[1],
                         d.h[2], s.cfg.lambda0);
            std::fprintf(m, "start %d %d %d\nsize %d %d %d\n", d.off[0] + i0_[0], d.off[1] + i0_[1], rz_ ? 0 : d.off[2] + i0_[2],
                         n_[0], n_[1], rz_ ? 1 : n_[2]);
            std::fprintf(m, "quantities %s\n", rz_ ? "Imax_phi0 tmax_phi0 fluence_phi0 Imax_phi90 tmax_phi90 fluence_phi90"
                                                  : "Imax tmax fluence");
            if (rz_) {
                const RadialGrid& g = s.rz->grid();
                std::fprintf(m, "r_nodes");
                for (int j = 0; j < n_[1]; ++j) std::fprintf(m, " %.17g", g.r[size_t(j)]);
                std::fprintf(m, "\n");
            }
            std::fprintf(m, "# r<rank>.bin: float64 [quantities][size z][size y][size x] (RZ: [quantities][r][z]), units of "
                            "(m c omega0 / e)^2 and 1/omega0; written at the end of the run and every %ld steps\n", H.every);
            std::fclose(m);
        }
    }

    void apply(Simulation& s) override {
        const Domain& d = s.dom;
        const long step = s.istep + 1;
        const double t = step * d.dt();
        if (np_ > 0) {
            if (rz_) {
                const FieldsRZ& F = *s.rz;
                hfocus_step_rz(F.ez.view(), F.er.view(), F.ep.view(), d.n[2], f_, i0_[0], n_[0], n_[1], st_.data(), out_.data(),
                               t, d.dt());
            } else {
                const Fields& F = *s.fields;
                hfocus_step_3d(F.ex.view(), F.ey.view(), F.ez.view(), f_, i0_[0], i0_[1], i0_[2], n_[0], n_[1], n_[2],
                               st_.data(), out_.data(), t, d.dt());
            }
        }
        const long every = c_.every;
        if ((every > 0 && step % every == 0) || t >= s.cfg.t_end - 0.5 * d.dt()) write(s, step);
    }

private:
    void write(Simulation& s, long step) {
        if (np_ == 0) return;
        const std::vector<double> h = out_.host();
        const std::string fn = dir_ + "/r" + std::to_string(s.comm.rank) + ".bin";
        FILE* b = std::fopen(fn.c_str(), "wb");
        if (!b) throw std::runtime_error("cannot open " + fn);
        std::fwrite(h.data(), sizeof(double), h.size(), b);
        std::fclose(b);
        FILE* m = std::fopen((dir_ + "/step_r" + std::to_string(s.comm.rank) + ".txt").c_str(), "w");   // the step written
        if (m) { std::fprintf(m, "%ld %.17g\n", step, step * s.dom.dt()); std::fclose(m); }
    }

    HFocusConfig c_;
    bool rz_ = false;
    HFFilter f_;
    int i0_[3] = {0, 0, 0}, n_[3] = {0, 0, 0}, nq_ = 0;
    long np_ = 0;
    Buf<double> st_, out_;
    std::string dir_;
};
}  // namespace

std::unique_ptr<Operator> make_hfocus_operator(const HFocusConfig& c) { return std::make_unique<HFocus>(c); }
