// Wake diagnostics (src/wake.hpp, docs/specs/structure3d.md).
#include "wake.hpp"
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "fsutil.hpp"
#include "rz.hpp"
#include "sim.hpp"

namespace {
const double QE = 1.602176634e-19, ME = 9.1093837015e-31, CL = 299792458.0, EPS0 = 8.8541878128e-12;
double k0_of(const Simulation& s) { return 2 * M_PI / s.cfg.lambda0; }
double e0_of(const Simulation& s) { return ME * CL * CL * k0_of(s) / QE; }   // field unit m c omega0 / e [V/m]

FILE* open_out(const Simulation& s, const std::string& name) {
    make_dirs(s.cfg.dir);
    const std::string p = s.cfg.dir + "/" + name;
    FILE* f = std::fopen(p.c_str(), s.restart_step >= 0 ? "a" : "w");
    if (!f) throw std::runtime_error("cannot open " + p + ": " + std::strerror(errno));
    return f;
}
}  // namespace

// ------------------------------------------------------------------------------------------------ kernels (free functions)
// F/q of a particle moving with c along +x at node (j, k), for the local columns 0 .. nx - 1: out[3 i + c]
void wake_line(const FV Ex, const FV Ey, const FV Ez, const FV By, const FV Bz, int j, int k, int nx, double* out) {
    par_for(nx, [=] HD(long q) {
        const int i = int(q);
        out[3 * i] = Ex(i, j, k);
        out[3 * i + 1] = 0.5 * (Ey(i, j, k) + Ey(i, j - 1, k)) - 0.5 * (Bz(i, j, k) + Bz(i, j - 1, k));
        out[3 * i + 2] = 0.5 * (Ez(i, j, k) + Ez(i, j, k - 1)) + 0.5 * (By(i, j, k) + By(i, j, k - 1));
    });
}

// S_y through the node plane j over the local columns and the local k in [k0, k1), times h_x h_z
double side_flux_y(const FV Ex, const FV Ez, const FV Bx, const FV Bz, int j, int k0, int k1, int nx, double dA) {
    const int nk = k1 - k0;
    if (nk <= 0) return 0;
    return dA * par_sum(long(nx) * nk, [=] HD(long q) {
        const int i = int(q % nx), k = k0 + int(q / nx);
        return Ez(i, j, k) * 0.5 * (Bx(i, j - 1, k) + Bx(i, j, k)) - Ex(i, j, k) * 0.5 * (Bz(i, j - 1, k) + Bz(i, j, k));
    });
}
// S_z through the node plane k over the local columns and the local j in [j0, j1), times h_x h_y
double side_flux_z(const FV Ex, const FV Ey, const FV Bx, const FV By, int k, int j0, int j1, int nx, double dA) {
    const int nj = j1 - j0;
    if (nj <= 0) return 0;
    return dA * par_sum(long(nx) * nj, [=] HD(long q) {
        const int i = int(q % nx), j = j0 + int(q / nx);
        return Ex(i, j, k) * 0.5 * (By(i, j, k - 1) + By(i, j, k)) - Ey(i, j, k) * 0.5 * (Bx(i, j, k - 1) + Bx(i, j, k));
    });
}
// (1/2) sum (E^2 + B^2) dV over the local columns, j in [j0, j1), k in [k0, k1)
double side_energy(const FV a, const FV b, const FV c, const FV e, const FV f, const FV g, int nx, int j0, int j1, int k0, int k1,
                   double dV) {
    const int nj = j1 - j0, nk = k1 - k0;
    if (nj <= 0 || nk <= 0) return 0;
    return 0.5 * dV * par_sum(long(nx) * nj * nk, [=] HD(long q) {
        const int i = int(q % nx), j = j0 + int((q / nx) % nj), k = k0 + int(q / (long(nx) * nj));
        auto sq = [&](const FV& v) { return v(i, j, k) * v(i, j, k); };
        return sq(a) + sq(b) + sq(c) + sq(e) + sq(f) + sq(g);
    });
}

namespace {
// ------------------------------------------------------------------------------------------------ wake potential
class Wake : public Operator {
public:
    explicit Wake(const WakeConfig& c) : c_(c) {}
    const char* name() const override { return "wake"; }
    Stage stage() const override { return Stage::Diagnostics; }
    void setup(Simulation& s) override {
        if (!s.cfg.window.enabled || s.cfg.window.v != 1 || s.dom.solver != Solver::RipCC)
            throw std::runtime_error("wake " + c_.name + ": needs a moving window at velocity = 1 and numerics.solver = rip_cc "
                                     "(c dt = h_x: a window column is a fixed xi)");
        const Domain& d = s.dom;
        rz_ = s.cfg.rz;
        if (rz_) own_ = true;                      // RZ (decision 105): the probe (x, y) = (c_.y, c_.z); MPI along z only
        else {
            jg_ = int(std::lround((c_.y < 0 ? 0.5 * d.L(1) : c_.y) / d.h[1]));
            kg_ = d.flat ? 0 : int(std::lround((c_.z < 0 ? 0.5 * d.L(2) : c_.z) / d.h[2]));
            const int j = jg_ - d.off[1], k = kg_ - d.off[2];
            own_ = j >= 0 && j < d.n[1] && (d.flat || (k >= 0 && k < d.n[2]));
        }
        acc_.assign(size_t(3 * d.N[0]), 0.0);
        line_.resize(size_t(3 * d.n[0]));
        host_.resize(size_t(3 * d.n[0]));
        last_wx_ = d.wx;
        if (s.comm.rank == 0 && rz_) {
            f_ = open_out(s, "wake_" + (c_.name.empty() ? std::string("probe") : c_.name) + ".txt");
            std::fprintf(f_, "# wake probe (RZ, all modes) at x = %.6g um, y = %.6g um; per output step a block: step t [fs], then\n"
                             "# xi [um]  Ez  Fx = Ex - c By  Fy = Ey + c Bx [V/m]  Vz  Vx  Vy [V] (V = int F c dt at fixed xi = z - c t)\n",
                         c_.y / k0_of(s) * 1e6, c_.z / k0_of(s) * 1e6);
        } else if (s.comm.rank == 0) {
            f_ = open_out(s, "wake_" + (c_.name.empty() ? std::string("probe") : c_.name) + ".txt");
            std::fprintf(f_, "# wake probe at y = %.6g um, z = %.6g um (nodes %d %d); per output step a block: step t [fs], then\n"
                             "# xi [um]  Ex  Fy = Ey - c Bz  Fz = Ez + c By [V/m]  Vx  Vy  Vz [V] (V = int F c dt at fixed xi = x - c t)\n",
                         jg_ * d.h[1] / k0_of(s) * 1e6, kg_ * d.h[2] / k0_of(s) * 1e6, jg_, kg_);
        }
    }
    void apply(Simulation& s) override {
        const Domain& d = s.dom;
        const bool moved = d.wx == last_wx_ + 1;   // one shift this step: the columns are fixed xi
        last_wx_ = d.wx;
        const long step = s.istep + 1;
        const bool out = c_.every > 0 && step % c_.every == 0;
        const bool acc = moved && s.t >= c_.start;
        if (!acc && !out) return;
        if (own_) {
            if (rz_) rz_wake_line(*s.rz, c_.y, c_.z, d.n[0], line_.data());
            else {
                const Fields& F = *s.fields;
                wake_line(F.ex.view(), F.ey.view(), F.ez.view(), F.by.view(), F.bz.view(), jg_ - d.off[1], d.flat ? 0 : kg_ - d.off[2],
                          d.n[0], line_.data());
            }
            line_.to_host(host_.data(), 0, host_.size());
            if (acc)
                for (int i = 0; i < d.n[0]; ++i)
                    for (int c = 0; c < 3; ++c) acc_[size_t(3 * (d.off[0] + i) + c)] += host_[size_t(3 * i + c)] * d.dt();
        }
        if (out) write(s, step);
    }
    void finish(Simulation& s) override { if (s.istep != written_) write(s, s.istep); }   // istep: the steps done
    ~Wake() override { if (f_) std::fclose(f_); }
private:
    void write(Simulation& s, long step) {
        written_ = step;
        const Domain& d = s.dom;
        const int N = d.N[0];
        std::vector<double> v(size_t(6 * N), 0.0);
        if (own_)
            for (int i = 0; i < d.n[0]; ++i)
                for (int c = 0; c < 3; ++c) v[size_t(6 * (d.off[0] + i) + c)] = host_[size_t(3 * i + c)];
        if (own_)                                  // the accumulated part: every owner of the probe adds its columns
            for (int i = 0; i < d.n[0]; ++i)
                for (int c = 0; c < 3; ++c) v[size_t(6 * (d.off[0] + i) + 3 + c)] = acc_[size_t(3 * (d.off[0] + i) + c)];
        s.comm.allreduce_sum(v.data(), int(v.size()));
        if (!f_) return;
        const double k0 = k0_of(s), E0 = e0_of(s);
        std::fprintf(f_, "\n# step %ld t %.8g fs\n", step, s.t / (CL * k0) * 1e15);
        for (int g = 0; g < N; ++g) {
            const double xi = (double(g + d.wx) * d.h[0] - s.t) / k0 * 1e6;
            const double* r = &v[size_t(6 * g)];
            std::fprintf(f_, "%.12g %.12e %.12e %.12e %.12e %.12e %.12e\n", xi, r[0] * E0, r[1] * E0, r[2] * E0, r[3] * E0 / k0,
                         r[4] * E0 / k0, r[5] * E0 / k0);
        }
        std::fflush(f_);
    }
    WakeConfig c_;
    int jg_ = 0, kg_ = 0;
    bool own_ = false, rz_ = false;
    long last_wx_ = 0, written_ = -1;
    std::vector<double> acc_, host_;
    Buf<double> line_;
    FILE* f_ = nullptr;
};

// ------------------------------------------------------------------------------------------------ beam slices
class Slices : public Operator {
public:
    explicit Slices(const SliceDiagConfig& c) : c_(c) {}
    const char* name() const override { return "beam_slices"; }
    Stage stage() const override { return Stage::Diagnostics; }
    void setup(Simulation& s) override {
        idx_ = -1;
        for (size_t i = 0; i < s.cfg.species.size(); ++i)
            if (s.cfg.species[i].name == c_.species) idx_ = int(i);
        if (idx_ < 0) throw std::runtime_error("beam_slices " + c_.name + ": no species " + c_.species);
        if (c_.every <= 0) c_.every = s.cfg.every;
        if (c_.dx <= 0) c_.dx = s.dom.h[0];
        if (s.comm.rank == 0) {
            f_ = open_out(s, "slices_" + (c_.name.empty() ? c_.species : c_.name) + ".txt");
            // the particle axes: Cart3D (x, y, z), RZ (z, x, y) (decision 105)
            const bool rz = s.cfg.rz;
            std::fprintf(f_, "# slices of species %s, width %.6g um along xi = %s - c t; per output step a block: step t [fs], then\n"
                             "# xi [um]  Q [pC]  I [kA]  %s [um]  <Ekin> sigma_E [MeV]  %s [m c]\n",
                         c_.species.c_str(), c_.dx / k0_of(s) * 1e6, rz ? "z" : "x",
                         rz ? "<x> <y> sigma_x sigma_y" : "<y> <z> sigma_y sigma_z", rz ? "<ux> <uy>" : "<uy> <uz>");
        }
    }
    void apply(Simulation& s) override {
        const long step = s.istep + 1;
        if (step % c_.every) return;
        const Domain& d = s.dom;
        const Species& sp = s.species[size_t(idx_)];
        const double xi0 = double(d.wx) * d.h[0] - s.t;          // xi of the left edge of the window
        const int nb = int(std::ceil(d.L(0) / c_.dx));
        // two passes (centred second moments, as beam_diag): w, wy, wz, wg, wuy, wuz, then w dy^2, w dz^2, w dg^2
        const int NV = 9;
        std::vector<double> a(size_t(NV * nb), 0.0);
        const std::vector<double> X = sp.get(Species::X), Y = sp.get(Species::Y), Z = sp.get(Species::Z),
                                  UX = sp.get(Species::UX), UY = sp.get(Species::UY), UZ = sp.get(Species::UZ),
                                  W = sp.get(Species::W);
        std::vector<int> bin(X.size(), -1);
        std::vector<double> gam(X.size(), 0.0);
        for (size_t p = 0; p < X.size(); ++p) {
            if (W[p] == 0) continue;
            const int b = int(std::floor((X[p] - s.t - xi0) / c_.dx));
            if (b < 0 || b >= nb) continue;
            bin[p] = b;
            gam[p] = std::sqrt(1 + UX[p] * UX[p] + UY[p] * UY[p] + UZ[p] * UZ[p]);
            const double w = W[p];
            double* r = &a[size_t(NV * b)];
            r[0] += w; r[1] += w * Y[p]; r[2] += w * Z[p]; r[3] += w * gam[p]; r[4] += w * UY[p]; r[5] += w * UZ[p];
        }
        s.comm.allreduce_sum(a.data(), int(a.size()));
        std::vector<double> c2(size_t(3 * nb), 0.0);
        for (size_t p = 0; p < X.size(); ++p) {
            if (bin[p] < 0) continue;
            const double* r = &a[size_t(NV * bin[p])];
            const double dy = Y[p] - r[1] / r[0], dz = Z[p] - r[2] / r[0], dg = gam[p] - r[3] / r[0];
            double* q = &c2[size_t(3 * bin[p])];
            q[0] += W[p] * dy * dy; q[1] += W[p] * dz * dz; q[2] += W[p] * dg * dg;
        }
        s.comm.allreduce_sum(c2.data(), int(c2.size()));
        for (int b = 0; b < nb; ++b)
            for (int c = 0; c < 3; ++c) a[size_t(NV * b + 6 + c)] = c2[size_t(3 * b + c)];
        if (!f_) return;
        const double k0 = k0_of(s), w0 = CL * k0, nc = EPS0 * ME * w0 * w0 / (QE * QE), mec2 = 0.51099895000;
        const double qpw = std::fabs(sp.q) * QE * nc / (k0 * k0 * k0);   // C per unit weight
        std::fprintf(f_, "\n# step %ld t %.8g fs\n", step, s.t / w0 * 1e15);
        for (int b = 0; b < nb; ++b) {
            const double* r = &a[size_t(NV * b)];
            if (r[0] <= 0) continue;
            const double w = r[0], my = r[1] / w, mz = r[2] / w, mg = r[3] / w;
            const double sy = std::sqrt(r[6] / w), sz = std::sqrt(r[7] / w), sg = std::sqrt(r[8] / w);
            const double Q = w * qpw, I = Q / (c_.dx / k0 / CL);
            const double xi = (xi0 + (b + 0.5) * c_.dx) / k0 * 1e6;
            std::fprintf(f_, "%.8g %.8e %.8e %.8e %.8e %.8e %.8e %.10e %.8e %.8e %.8e\n", xi, Q * 1e12, I * 1e-3, my / k0 * 1e6,
                         mz / k0 * 1e6, sy / k0 * 1e6, sz / k0 * 1e6, (mg - 1) * mec2 * sp.m, sg * mec2 * sp.m, r[4] / w, r[5] / w);
        }
        std::fflush(f_);
    }
    ~Slices() override { if (f_) std::fclose(f_); }
private:
    SliceDiagConfig c_;
    int idx_ = -1;
    FILE* f_ = nullptr;
};

// ------------------------------------------------------------------------------------------------ side flux
class SideFlux : public Operator {
public:
    explicit SideFlux(const SideFluxConfig& c) : c_(c) {}
    const char* name() const override { return "side_flux"; }
    Stage stage() const override { return Stage::Diagnostics; }
    void setup(Simulation& s) override {
        if (s.cfg.rz) throw std::runtime_error("side_flux: Cart3D only");
        const Domain& d = s.dom;
        for (int a = 1; a < 3; ++a) {
            const int L = s.cfg.pml.on[a] ? s.cfg.pml.cells : 0;
            lo_[a] = c_.lo[a] >= 0 ? int(std::lround(c_.lo[a] / d.h[a])) : L + 2;
            hi_[a] = c_.hi[a] >= 0 ? int(std::lround(c_.hi[a] / d.h[a])) : d.N[a] - L - 2;
            if (a == 2 && d.flat) { lo_[2] = 0; hi_[2] = 1; }
            if (hi_[a] <= lo_[a]) throw std::runtime_error("side_flux: empty box");
        }
        if (c_.every <= 0) c_.every = s.cfg.every;
        prev_ = power(s);
        if (s.comm.rank == 0) {
            f_ = open_out(s, "side_flux.txt");
            std::fprintf(f_, "# energy leaving the box %d <= j <= %d, %d <= k <= %d (node planes) through its side faces\n"
                             "# step t[fs]  P_ylo P_yhi P_zlo P_zhi [W]  E_ylo E_yhi E_zlo E_zhi (accumulated) [J]  W_inside [J]\n",
                         lo_[1], hi_[1], lo_[2], hi_[2]);
        }
    }
    void apply(Simulation& s) override {
        const std::vector<double> p = power(s);
        const double dt = s.dom.dt();
        for (int f = 0; f < 4; ++f) { sum_[f] += 0.5 * (prev_[size_t(f)] + p[size_t(f)]) * dt; }
        prev_ = p;
        const long step = s.istep + 1;
        if (step % c_.every) return;
        const Domain& d = s.dom;
        const Fields& F = *s.fields;
        double w = side_energy(F.ex.view(), F.ey.view(), F.ez.view(), F.bx.view(), F.by.view(), F.bz.view(), d.n[0],
                               clampj(d, lo_[1], 1), clampj(d, hi_[1], 1), d.flat ? 0 : clampj(d, lo_[2], 2),
                               d.flat ? 1 : clampj(d, hi_[2], 2), d.cell_volume());
        s.comm.allreduce_sum(&w, 1);
        if (!f_) return;
        const double k0 = k0_of(s), E0 = e0_of(s), wu = EPS0 * E0 * E0 / (k0 * k0 * k0), pu = wu * CL * k0;
        std::fprintf(f_, "%ld %.8g %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n", step, s.t / (CL * k0) * 1e15,
                     p[0] * pu, p[1] * pu, p[2] * pu, p[3] * pu, sum_[0] * wu, sum_[1] * wu, sum_[2] * wu, sum_[3] * wu, w * wu);
        std::fflush(f_);
    }
    ~SideFlux() override { if (f_) std::fclose(f_); }
    double sum_[4] = {0, 0, 0, 0};
private:
    static int clampj(const Domain& d, int g, int a) { const int l = g - d.off[a]; return l < 0 ? 0 : (l > d.n[a] ? d.n[a] : l); }
    // outward power through the four faces (code units), over all ranks
    std::vector<double> power(Simulation& s) {
        const Domain& d = s.dom;
        const Fields& F = *s.fields;
        std::vector<double> p(4, 0.0);
        const int k0 = d.flat ? 0 : clampj(d, lo_[2], 2), k1 = d.flat ? 1 : clampj(d, hi_[2], 2);
        const int j0 = clampj(d, lo_[1], 1), j1 = clampj(d, hi_[1], 1);
        for (int e = 0; e < 2; ++e) {
            const int jl = (e == 0 ? lo_[1] : hi_[1]) - d.off[1];
            if (jl >= 0 && jl < d.n[1])
                p[size_t(e)] = (e == 0 ? -1 : 1) * side_flux_y(F.ex.view(), F.ez.view(), F.bx.view(), F.bz.view(), jl, k0, k1, d.n[0],
                                                               d.h[0] * d.h[2]);
            if (d.flat) continue;
            const int kl = (e == 0 ? lo_[2] : hi_[2]) - d.off[2];
            if (kl >= 0 && kl < d.n[2])
                p[size_t(2 + e)] = (e == 0 ? -1 : 1) * side_flux_z(F.ex.view(), F.ey.view(), F.bx.view(), F.by.view(), kl, j0, j1,
                                                                   d.n[0], d.h[0] * d.h[1]);
        }
        s.comm.allreduce_sum(p.data(), 4);
        return p;
    }
    SideFluxConfig c_;
    int lo_[3] = {0, 0, 0}, hi_[3] = {0, 0, 0};
    std::vector<double> prev_;
    FILE* f_ = nullptr;
};
}  // namespace

std::unique_ptr<Operator> make_wake_operator(const WakeConfig& c) { return std::make_unique<Wake>(c); }
std::unique_ptr<Operator> make_slices_operator(const SliceDiagConfig& c) { return std::make_unique<Slices>(c); }
std::unique_ptr<Operator> make_side_flux_operator(const SideFluxConfig& c) { return std::make_unique<SideFlux>(c); }
