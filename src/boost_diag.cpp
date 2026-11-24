// Lab-frame diagnostics of a boosted run (block lab_diagnostics; docs/specs/boosted_frame.md, decision 87).
//
// Fields (as WarpX's and FBPIC's back-transformed diagnostics): the lab time T is met in the boosted step t'_n on the
// plane x'_T = ((T - T0) / gamma - t'_n) / beta. Every step that plane is interpolated (cubic Lagrange) between the nodes
// around it (all field components sit at integer x, RIP-CC and iRIP), transformed to the lab frame and stored at the lab
// position x = gamma (x'_T + beta t'_n) + X0: the planes of one snapshot are spaced dt' / (gamma beta) in the lab, one per
// boosted step, and cover the window. With the transverse pairs co-located (Ey, Bz; Ez, By; RZ: Er, Bphi; Ephi, Br)
// the transformation needs no transverse interpolation:
//   E1 = gamma (E1' + beta B2'),  E2 = gamma (E2' - beta B1'),  B1 = gamma (B1' - beta E2'),  B2 = gamma (B2' + beta E1'),
// E, B along the boost axis unchanged (1, 2: y, z in Cart3D; r, phi per azimuthal mode in RZ, a linear map per mode).
// Particles: a particle crosses the lab time T in the step where t_lab = gamma (t' + beta x') + T0 passes T; its
// position is interpolated on the straight move of the step (x_old = x - dt u / gamma, the leapfrog move), its momentum
// is the step's u, boosted back: u_x = gamma (u'_x + beta gamma'). Moments with beam_moments (beam_diag.hpp) of the lab
// particles, the particles themselves as float64 records.
// QED (decision 117): photons (kind = photon) are massless: velocity u / |u|, u_x = gamma (u'_x + beta |u'|); species with
// spin get the lab rest-frame spin (s0 s1 s2, the Wigner rotation included: qed_spin_boost), photons their Stokes vector
// (xi1 xi2 xi3, unchanged) and polarization axis (pe0 pe1 pe2, qed_pol_boost) as extra columns of their records.
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "boost.hpp"
#include "fsutil.hpp"
#include "qed.hpp"
#include "sim.hpp"
#if RIP_GPU
#include <cub/cub.cuh>
#endif

namespace {

FILE* open_or_throw(const std::string& p, const char* mode) {
    FILE* f = std::fopen(p.c_str(), mode);
    if (!f) throw std::runtime_error("cannot open " + p + ": " + std::strerror(errno));
    return f;
}

// One plane of the six components (longitudinal, transverse 1, 2 of E, then of B) in the lab frame: out[c n1 n2 +
// k n1 + j] for the owned transverse cells, interpolated between the nodes i and i + 1 with weight w.
struct FV6 { FV f[6]; };              // by value into the kernel
void sample_plane(const FV6 F, int i, double w, int n1, int n2, double g, double b, double* out) {
    const long n = long(n1) * n2;
    par_for(n, [=] HD(long q) {
        const int j = int(q % n1), k = int(q / n1);
        double v[6];
        // cubic Lagrange through the nodes i - 1 .. i + 2 (linear lost 2 % of a carrier at 16 cells per wavelength)
        const double am = -w * (w - 1) * (w - 2) / 6, a0 = (w + 1) * (w - 1) * (w - 2) / 2, a1 = -(w + 1) * w * (w - 2) / 2,
                     a2 = (w + 1) * w * (w - 1) / 6;
        for (int c = 0; c < 6; ++c)
            v[c] = am * F.f[c](i - 1, j, k) + a0 * F.f[c](i, j, k) + a1 * F.f[c](i + 1, j, k) + a2 * F.f[c](i + 2, j, k);
        out[q] = v[0];
        out[n + q] = g * (v[1] + b * v[5]);
        out[2 * n + q] = g * (v[2] - b * v[4]);
        out[3 * n + q] = v[3];
        out[4 * n + q] = g * (v[4] - b * v[2]);
        out[5 * n + q] = g * (v[5] + b * v[1]);
    });
}

// Particles of one species crossing the lab times T[0 .. nt) in the step t0 -> t0 + dt: count per particle (C), then the
// records (8 doubles: snapshot, x, y, z, ux, uy, uz, w; lab frame) from offset O[p].
// sel_u: keep crossings with the lab u along the boost axis >= sel_u; sel_r: and |y|, |z| < sel_r (RZ: the particles' x, y); the
// lab_diagnostics keys select_uz, select_r (FBPIC's BackTransformedParticleDiagnostic select), off by default
constexpr int LAB_MAX = 64;                      // lab times with particle snapshots (Cross is passed by value to the kernels)
// photon: massless (gp = |u|); nx extra columns (0; 3: the spin; 6: the Stokes vector and the polarization axis, decision 117)
struct Cross { BoostConfig B; double t0, dt; int nt; double T[LAB_MAX]; double sel_u = -1e300, sel_r = 1e300, yc = 0, zc = 0; bool flat = false;
               bool photon = false; int nx = 0; };   // yc, zc: Cart3D: the box centre
struct Extra { const double* a[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}; };   // by value into the kernel
HD inline double cross_gamma(const Cross& c, double ux, double uy, double uz) {
    return c.photon ? sqrt(ux * ux + uy * uy + uz * uz) : sqrt(1 + ux * ux + uy * uy + uz * uz);
}
HD inline int crossings(const Cross& c, double x1, double y1, double z1, double ux, double uy, double uz, int out[LAB_MAX], double f[LAB_MAX]) {
    const double gp = cross_gamma(c, ux, uy, uz), x0 = x1 - c.dt * ux / gp;
    if (c.B.gamma * (ux + c.B.beta * gp) < c.sel_u) return 0;
    if (fabs(y1 - c.yc) >= c.sel_r || (!c.flat && fabs(z1 - c.zc) >= c.sel_r)) return 0;   // the end of the step
    const double s0 = c.B.t_lab(x0, c.t0), s1 = c.B.t_lab(x1, c.t0 + c.dt);
    int m = 0;
    for (int k = 0; k < c.nt; ++k)
        if (s0 < c.T[k] && c.T[k] <= s1) { out[m] = k; f[m] = (c.T[k] - s0) / (s1 - s0); ++m; }
    return m;
}
void count_crossings(const Cross c, long n, const double* X, const double* Y, const double* Z, const double* UX, const double* UY,
                     const double* UZ, int* C) {
    par_for(n, [=] HD(long p) {
        int o[LAB_MAX]; double f[LAB_MAX];
        C[p] = crossings(c, X[p], Y[p], Z[p], UX[p], UY[p], UZ[p], o, f);
    });
}
void fill_crossings(const Cross c, long n, const double* X, const double* Y, const double* Z, const double* UX, const double* UY,
                    const double* UZ, const double* W, const Extra A, const int* C, const int* O, double* R) {
    const long rw = 8 + c.nx;
    par_for(n, [=] HD(long p) {
        if (C[p] == 0) return;
        int o[LAB_MAX]; double f[LAB_MAX];
        const int m = crossings(c, X[p], Y[p], Z[p], UX[p], UY[p], UZ[p], o, f);
        const double ux = UX[p], uy = UY[p], uz = UZ[p], gp = cross_gamma(c, ux, uy, uz);
        const double x1 = X[p], y1 = Y[p], z1 = Z[p], dx = c.dt * ux / gp, dy = c.dt * uy / gp, dz = c.dt * uz / gp;
        double ex[6] = {0, 0, 0, 0, 0, 0};                     // the extra columns in the lab (decision 117)
        const double u[3] = {ux, uy, uz};
        if (c.nx == 3) {                                       // spin: the lab rest-frame spin (the frame moves at -beta)
            for (int k = 0; k < 3; ++k) ex[k] = A.a[k][p];
            qed_spin_boost(u, ex, -c.B.beta);
        } else if (c.nx == 6) {                                // photon: Stokes vector unchanged, polarization axis to the lab
            for (int k = 0; k < 6; ++k) ex[k] = A.a[k][p];
            qed_pol_boost(u, ex + 3, -c.B.beta);
        }
        for (int q = 0; q < m; ++q) {
            double* r = R + rw * long(O[p] + q);
            const double a = 1 - f[q];                        // back from the end of the move
            const double xp = x1 - a * dx, tp = c.t0 + f[q] * c.dt;
            r[0] = o[q];
            r[1] = c.B.x_lab(xp, tp); r[2] = y1 - a * dy; r[3] = z1 - a * dz;
            r[4] = c.B.gamma * (ux + c.B.beta * gp); r[5] = uy; r[6] = uz;
            r[7] = W[p];
            for (int k = 0; k < c.nx; ++k) r[8 + k] = ex[k];
        }
    });
}

struct Scratch { Buf<int> cnt, off; Buf<double> rec, plane; Buf<unsigned char> tmp; };
Scratch& scratch() {
    static Scratch* s = new Scratch;
    return *s;
}
template <class T> void ensure(Buf<T>& b, size_t n) { if (b.size() < n) b.resize(n); }

// The extra columns of a species (decision 117): 6 for photons (xi1 xi2 xi3 pe0 pe1 pe2), 3 for a species with spin (s0 s1 s2)
int extra_columns(const Species& sp) { return sp.photon ? 6 : sp.slot("s0") >= 0 ? 3 : 0; }
// The records of the crossings of species sp in the step (host, 8 + c.nx doubles each).
std::vector<double> collect(const Species& sp, Cross c) {
    c.photon = sp.photon;
    c.nx = extra_columns(sp);
    Extra A;
    if (c.nx == 6) for (int k = 0; k < 3; ++k) { A.a[k] = sp.ptr<double>(sp.slot("xi1") + k); A.a[3 + k] = sp.ptr<double>(sp.slot("pe0") + k); }
    else if (c.nx == 3) for (int k = 0; k < 3; ++k) A.a[k] = sp.ptr<double>(sp.slot("s0") + k);
    const long n = long(sp.size());
    if (n == 0) return {};
    Scratch& S = scratch();
    ensure(S.cnt, size_t(n)); ensure(S.off, size_t(n) + 1);
    int* C = S.cnt.data(); int* O = S.off.data();
    const double* X = sp.ptr<double>(Species::X); const double* Y = sp.ptr<double>(Species::Y); const double* Z = sp.ptr<double>(Species::Z);
    const double* UX = sp.ptr<double>(Species::UX); const double* UY = sp.ptr<double>(Species::UY); const double* UZ = sp.ptr<double>(Species::UZ);
    const double* W = sp.ptr<double>(Species::W);
    count_crossings(c, n, X, Y, Z, UX, UY, UZ, C);
    const double tot = par_sum(n, [=] HD(long p) { return double(C[p]); });
    if (tot < 0.5) return {};
#if RIP_GPU
    size_t bytes = 0;
    gpu_check(cub::DeviceScan::ExclusiveSum(nullptr, bytes, C, O, int(n)), "cub scan (size)");
    ensure(S.tmp, bytes);
    gpu_check(cub::DeviceScan::ExclusiveSum(S.tmp.data(), bytes, C, O, int(n)), "cub scan");
#else
    for (long p = 0, t = 0; p < n; ++p) { O[p] = int(t); t += C[p]; }
#endif
    const size_t m = size_t(tot + 0.5), rw = size_t(8 + c.nx);
    ensure(S.rec, rw * m);
    fill_crossings(c, n, X, Y, Z, UX, UY, UZ, W, A, C, O, S.rec.data());
    std::vector<double> h(rw * m);
    S.rec.to_host(h.data(), 0, h.size());
    return h;
}

const double e_si = 1.602176634e-19, me_si = 9.1093837015e-31, eps0_si = 8.8541878128e-12, c_si = 299792458.0;
const double mec2_mev = 0.51099895000;

// ---------------------------------------------------------------- fields
class LabFields : public Operator {
public:
    const char* name() const override { return "lab_fields"; }
    Stage stage() const override { return Stage::Diagnostics; }
    void setup(Simulation& s) override {
        for (double T : s.cfg.labdiag.times) { Snap S; S.T = T; snaps_.push_back(S); }
        if (s.comm.rank == 0) make_dirs(s.cfg.dir + "/lab_fields");
        if (s.restart_step >= 0) restore(s);
    }
    // Checkpoint (decision 116): the lab times, the status of every snapshot (0 open, 1 written, 2 open but incomplete: its
    // first planes were lost in a restart without this state) and of each open snapshot with planes the lab x of all its
    // planes (xl_K) and their values gathered over the ranks (planes_K, [component][n2][n1][plane], planes in step order).
    bool save(Simulation& s, OpState& o) override {
        std::vector<double> st(snaps_.size());
        for (size_t k = 0; k < snaps_.size(); ++k) st[k] = snaps_[k].done ? 1 : snaps_[k].lost ? 2 : 0;
        o.add("times", s.cfg.labdiag.times);
        o.add("status", st);
        for (size_t k = 0; k < snaps_.size(); ++k) {
            const Snap& S = snaps_[k];
            if (S.done || S.xl.empty()) continue;
            std::vector<double> g = assemble(s, S, false);
            if (s.comm.rank != 0) continue;
            o.add("xl_" + std::to_string(k), S.xl);
            o.add("planes_" + std::to_string(k), std::move(g));
        }
        return true;
    }
    void apply(Simulation& s) override {
        const Domain& d = s.dom;
        const BoostConfig& B = s.cfg.boost;
        const double t1 = double(s.istep + 1) * d.dt(), h = d.h[0];
        const double lo = double(d.wx) * h, top = lo + double(d.N[0] - 1) * h;   // global node range [lo, top]
        for (size_t k = 0; k < snaps_.size(); ++k) {
            Snap& S = snaps_[k];
            if (S.done) continue;
            const double xT = ((S.T - B.T0) / B.gamma - t1) / B.beta;
            if (xT < lo) { finalize(s, k, true); continue; }
            if (xT >= top) continue;
            S.xl.push_back(B.x_lab(xT, t1));
            const double fi = std::floor(xT / h);
            const int i = int(long(fi) - d.cell0(0));
            if (i < 0 || i >= d.n[0]) continue;                  // another rank along x
            const int n1 = d.n[1], n2 = d.n[2];
            const size_t n = size_t(n1) * n2;
            Buf<double>& P = scratch().plane;
            ensure(P, 6 * n);
            FV6 F;
            if (s.cfg.rz) {
                const FieldsRZ& f = *s.rz;
                F = {{f.ez.view(), f.er.view(), f.ep.view(), f.bz.view(), f.br.view(), f.bp.view()}};
            } else {
                const Fields& f = *s.fields;
                F = {{f.ex.view(), f.ey.view(), f.ez.view(), f.bx.view(), f.by.view(), f.bz.view()}};
            }
            sample_plane(F, i, xT / h - fi, n1, n2, B.gamma, B.beta, P.data());
            S.mine.push_back(int(S.xl.size()) - 1);
            const size_t o = S.data.size();
            S.data.resize(o + 6 * n);
            P.to_host(S.data.data() + o, 0, 6 * n);
        }
    }
    void finish(Simulation& s) override {
        for (size_t k = 0; k < snaps_.size(); ++k)
            if (!snaps_[k].done) finalize(s, k, false);
    }
private:
    struct Snap {
        double T;
        bool done = false;
        bool lost = false;               // planes from before a restart are missing: written with complete 0
        std::vector<double> xl;          // lab x of every plane so far (all ranks), in step order (decreasing)
        std::vector<int> mine;           // planes sampled here
        std::vector<double> data;        // their values, 6 n1 n2 each (owned transverse cells)
        long npre = 0;                   // the first npre planes come from the checkpoint (rank 0: pre)
        std::vector<double> pre;         // [component][N2][N1][npre], step order (rank 0)
    };
    std::vector<Snap> snaps_;
    // The global array of the planes so far on every rank (a sum over the ranks; the restored planes from rank 0),
    // [component][N2][N1][plane]; planes in increasing lab x (lab_x) or in step order.
    std::vector<double> assemble(Simulation& s, const Snap& S, bool lab_x) {
        const Domain& d = s.dom;
        const long P = long(S.xl.size()), N1 = d.N[1], N2 = d.N[2], n1 = d.n[1], n2 = d.n[2];
        std::vector<double> g(size_t(6 * P * N1 * N2), 0.0);
        auto pos = [&](long j) { return lab_x ? P - 1 - j : j; };   // plane j of the step order
        if (s.comm.rank == 0 && S.npre > 0)
            for (long r = 0; r < 6 * N2 * N1; ++r)
                for (long j = 0; j < S.npre; ++j) g[size_t(r * P + pos(j))] = S.pre[size_t(r * S.npre + j)];
        for (size_t q = 0; q < S.mine.size(); ++q) {
            const long p = pos(S.mine[q]);
            const double* v = S.data.data() + q * 6 * n1 * n2;
            for (int c = 0; c < 6; ++c)
                for (long kk = 0; kk < n2; ++kk)
                    for (long j = 0; j < n1; ++j)
                        g[size_t(((c * N2 + d.off[2] + kk) * N1 + d.off[1] + j) * P + p)] = v[(c * n2 + kk) * n1 + j];
        }
        for (size_t o = 0; o < g.size(); o += size_t(1) << 26)   // in chunks (int counts)
            s.comm.allreduce_sum(g.data() + o, int(std::min(g.size() - o, size_t(1) << 26)));
        return g;
    }
    // After a restart: the state of the checkpoint (save), or, from a checkpoint without it (written before decision 116, or
    // with other lab times), the status from the geometry: written if its plane had left the box at the checkpoint's step,
    // incomplete (complete 0, said in the log) if it had planes then, else open.
    void restore(Simulation& s) {
        const bool r0 = s.comm.rank == 0;
        const size_t nt = snaps_.size();
        OpState o;
        bool have = read_op_state(s, name(), {"times", "status"}, o);
        const char* why = "the checkpoint holds no lab-snapshot state (written before decision 116)";
        if (have && *o.get("times") != s.cfg.labdiag.times) { have = false; why = "lab_diagnostics has other lab times than the checkpoint's run"; }
        if (have && o.get("status")->size() != nt) { have = false; why = "the checkpoint's lab-snapshot state is inconsistent"; }
        if (have) {
            const std::vector<double> st = *o.get("status");
            std::vector<std::string> keys;
            for (size_t k = 0; k < nt; ++k) {
                if (st[k] == 1) continue;
                keys.push_back("xl_" + std::to_string(k));
                if (r0) keys.push_back("planes_" + std::to_string(k));
            }
            read_op_state(s, name(), keys, o);
            const long N12 = long(s.dom.N[1]) * s.dom.N[2];
            for (size_t k = 0; k < nt; ++k) {
                Snap& S = snaps_[k];
                S.done = st[k] == 1;
                S.lost = st[k] == 2;
                if (S.done) continue;
                S.xl = *o.get("xl_" + std::to_string(k));
                S.npre = long(S.xl.size());
                if (r0 && S.npre > 0) {
                    S.pre = *o.get("planes_" + std::to_string(k));
                    if (S.pre.size() != size_t(6 * N12 * S.npre))
                        throw std::runtime_error("checkpoint: the planes of lab snapshot " + std::to_string(k) + " have another size");
                }
                if (r0 && S.npre > 0)
                    std::printf("  lab_diagnostics: field snapshot %zu restored from the checkpoint (%ld planes)%s\n", k, S.npre,
                                S.lost ? ", incomplete since an earlier restart (complete 0)" : "");
            }
            return;
        }
        const Domain& d = s.dom;
        const BoostConfig& B = s.cfg.boost;
        const double t1 = double(s.istep) * d.dt(), h = d.h[0];          // the last step before the checkpoint (apply)
        const double lo = double(d.wx) * h, top = lo + double(d.N[0] - 1) * h;
        for (size_t k = 0; k < nt; ++k) {
            Snap& S = snaps_[k];
            const double xT = ((S.T - B.T0) / B.gamma - t1) / B.beta;
            if (xT < lo) S.done = true;                                   // written before the checkpoint
            else if (xT < top) {
                S.lost = true;
                if (r0)
                    std::fprintf(stderr, "warning: lab_diagnostics: field snapshot %zu (t_lab %.6g fs) began before the checkpoint and "
                                         "cannot be restored (%s): it is written with complete 0\n", k,
                                 S.T / (c_si * 2 * M_PI / s.cfg.lambda0) * 1e15, why);
            }
        }
    }
    // All planes to rank 0 (a sum over the ranks of the global array), written in increasing lab x:
    // dir/lab_fields/snap_KKK.bin, float64 [component][transverse 2][transverse 1][x], and snap_KKK.txt.
    void finalize(Simulation& s, size_t k, bool complete) {
        Snap& S = snaps_[k];
        S.done = true;
        complete = complete && !S.lost;
        const Domain& d = s.dom;
        const long P = long(S.xl.size()), N1 = d.N[1], N2 = d.N[2];
        if (P == 0) return;
        const std::vector<double> g = assemble(s, S, true);
        S.data.clear(); S.data.shrink_to_fit();
        S.pre.clear(); S.pre.shrink_to_fit();
        if (s.comm.rank != 0) return;
        char base[64];
        std::snprintf(base, sizeof base, "/lab_fields/snap_%03zu", k);
        const std::string p = s.cfg.dir + base;
        FILE* f = open_or_throw(p + ".bin", "wb");
        std::fwrite(g.data(), sizeof(double), g.size(), f);
        std::fclose(f);
        const BoostConfig& B = s.cfg.boost;
        const double k0 = 2 * M_PI / s.cfg.lambda0;
        FILE* m = open_or_throw(p + ".txt", "w");
        std::fprintf(m, "# lab-frame fields of a boosted run (docs/specs/boosted_frame.md); code units (1/k0, 1/omega0, m c omega0 / e)\n");
        std::fprintf(m, "t_lab %.12g\nt_lab_fs %.10g\ncomplete %d\ngeometry %s\nmodes %d\n", S.T, S.T / (c_si * k0) * 1e15,
                     complete ? 1 : 0, s.cfg.rz ? "rz" : (s.cfg.flat ? "cartesian2d" : "cartesian"), s.cfg.modes);
        std::fprintf(m, "nx %ld\nx0 %.12g\ndx %.12g\n", P, S.xl.back(), (S.xl.front() - S.xl.back()) / std::max(1.0, double(P - 1)));
        std::fprintf(m, "n1 %ld\nn2 %ld\nd1 %.12g\nd2 %.12g\n", N1, N2, d.h[1], d.h[2]);
        std::fprintf(m, "components %s\nlayout float64 [component][n2][n1][nx], x fastest\n",
                     s.cfg.rz ? "Ez Er Ephi Bz Br Bphi (per mode slot: 0 = m 0, 2m-1, 2m = Re, Im of F_m; field at phi = F_0 + 2 Re sum F_m e^(i m phi))" : "Ex Ey Ez Bx By Bz");
        std::fprintf(m, "stagger as the grid (src/grid.hpp, src/rz.hpp): transverse positions of each component unchanged\n");
        std::fprintf(m, "gamma_boost %.12g\nlambda0_m %.6g\n", B.gamma, s.cfg.lambda0);
        std::fclose(m);
    }
};

// ---------------------------------------------------------------- particles
class LabParticles : public Operator {
public:
    const char* name() const override { return "lab_particles"; }
    Stage stage() const override { return Stage::Fields; }
    void setup(Simulation& s) override {
        const LabDiagConfig& L = s.cfg.labdiag;
        if (L.times.size() > size_t(LAB_MAX)) throw std::runtime_error("lab_diagnostics: at most " + std::to_string(LAB_MAX) + " lab times with species");
        for (const std::string& n : L.species) {
            int idx = -1;
            for (size_t i = 0; i < s.species.size(); ++i)
                if (s.species[i].name == n) idx = int(i);
            if (idx < 0) throw std::runtime_error("lab_diagnostics.species: no species " + n);
            idx_.push_back(idx);
            rw_.push_back(7 + size_t(extra_columns(s.species[size_t(idx)])));   // record width (decision 117)
        }
        done_.assign(L.times.size(), 0);
        lost_.assign(L.times.size(), 0);
        rec_.assign(L.times.size() * idx_.size(), {});
        if (s.comm.rank == 0) make_dirs(s.cfg.dir + "/lab_particles");
        if (s.restart_step >= 0) { restore(s); return; }
        if (s.comm.rank == 0) {
            beams_ = open_or_throw(s.cfg.dir + "/lab_beams.txt", "w");
            std::fprintf(beams_, "# lab-frame beams of a boosted run at the lab times of lab_diagnostics, gamma >= %g\n"
                                 "# snapshot t_lab[fs] species complete N_macro Q[pC] E_mean[MeV] dE/E_rms E_max[MeV] eps_ny[um] "
                                 "eps_nz[um] x_mean[um] sigma_x[um] sigma_y[um] sigma_z[um]\n", L.gamma_min);
            std::fflush(beams_);
        }
    }
    // Checkpoint (decision 116): the lab times, the species, the status of every snapshot (0 open, 1 written, 2 incomplete),
    // the length of lab_beams.txt, the number of ranks and of each open snapshot and species the records so far, gathered in
    // rank order (rec_K_Q, widths[Q] doubles each: 7, + 3 with spin, + 6 for photons) with the count of every rank (cnt_K_Q).
    bool save(Simulation& s, OpState& o) override {
        const int R = s.comm.size;
        const size_t ns = idx_.size();
        std::vector<double> st(done_.size()), sp(idx_.begin(), idx_.end()), wd(rw_.begin(), rw_.end());
        for (size_t k = 0; k < done_.size(); ++k) st[k] = done_[k] ? 1 : lost_[k] ? 2 : 0;
        o.add("times", s.cfg.labdiag.times);
        o.add("species", sp);
        o.add("widths", wd);
        o.add("status", st);
        o.add("ranks", {double(R)});
        if (beams_) { std::fflush(beams_); o.add("beams_bytes", {double(std::ftell(beams_))}); }
        for (size_t k = 0; k < done_.size(); ++k) {
            if (done_[k]) continue;
            for (size_t q = 0; q < ns; ++q) {
                const std::vector<double>& v = rec_[k * ns + q];
                const double rw = double(rw_[q]), nloc = double(v.size() / rw_[q]);
                std::vector<double> cnt(static_cast<size_t>(R));
                s.comm.allgather(&nloc, 1, cnt.data());
                double before = 0, tot = 0;
                for (int r = 0; r < R; ++r) { if (r < s.comm.rank) before += cnt[size_t(r)]; tot += cnt[size_t(r)]; }
                if (tot == 0) continue;
                std::vector<double> all(size_t(rw * tot), 0.0);
                std::copy(v.begin(), v.begin() + long(rw * nloc), all.begin() + long(rw * before));
                for (size_t a = 0; a < all.size(); a += size_t(1) << 26)
                    s.comm.allreduce_sum(all.data() + a, int(std::min(all.size() - a, size_t(1) << 26)));
                if (s.comm.rank != 0) continue;
                const std::string kq = std::to_string(k) + "_" + std::to_string(q);
                o.add("rec_" + kq, std::move(all));
                o.add("cnt_" + kq, cnt);
            }
        }
        return true;
    }
    void apply(Simulation& s) override {
        const Domain& d = s.dom;
        const BoostConfig& B = s.cfg.boost;
        const LabDiagConfig& L = s.cfg.labdiag;
        Cross c;
        c.B = B; c.t0 = double(s.istep) * d.dt(); c.dt = d.dt(); c.nt = int(L.times.size());
        c.sel_u = L.select_uz; c.sel_r = L.select_r; c.flat = s.cfg.flat;
        if (!s.cfg.rz) { c.yc = 0.5 * s.cfg.N[1] * s.cfg.h[1]; c.zc = 0.5 * s.cfg.N[2] * s.cfg.h[2]; }
        for (int k = 0; k < c.nt; ++k) c.T[k] = done_[size_t(k)] ? -1e300 : L.times[size_t(k)];   // done: never crossed again
        for (size_t q = 0; q < idx_.size(); ++q) {
            const std::vector<double> r = collect(s.species[size_t(idx_[q])], c);
            const size_t w = rw_[q] + 1;                  // + the snapshot index
            for (size_t p = 0; p < r.size(); p += w) {
                std::vector<double>& v = rec_[size_t(r[p]) * idx_.size() + q];
                v.insert(v.end(), r.begin() + long(p) + 1, r.begin() + long(p + w));
            }
        }
        // done once the lab time at the box's left edge (one cell beyond: the particles kept there) has passed T
        const double xl = double(d.wx - 1) * d.h[0], t1 = c.t0 + c.dt;
        for (int k = 0; k < c.nt; ++k)
            if (!done_[size_t(k)] && B.t_lab(xl, t1) > L.times[size_t(k)]) finalize(s, size_t(k), true);
    }
    void finish(Simulation& s) override {
        for (size_t k = 0; k < done_.size(); ++k)
            if (!done_[k]) finalize(s, k, false);
        if (beams_) { std::fclose(beams_); beams_ = nullptr; }
    }
    ~LabParticles() override { if (beams_) std::fclose(beams_); }
private:
    std::vector<int> idx_;
    std::vector<size_t> rw_;                 // record width per species: 7, + 3 (spin), + 6 (photons; decision 117)
    std::vector<char> done_;
    std::vector<char> lost_;                 // records from before a restart are missing: written with complete 0
    std::vector<std::vector<double>> rec_;   // [snapshot][species]: rw_ doubles (x y z ux uy uz w [extra]) per particle, this rank
    FILE* beams_ = nullptr;
    // After a restart: the state of the checkpoint (save; the records back to their ranks if the number of ranks is the same,
    // else all to rank 0; lab_beams.txt cut to its length at the checkpoint), or, from a checkpoint without it, the status
    // from the geometry (written if the lab time had left the box at the checkpoint's step, incomplete if a particle in the
    // box could have crossed it, said in the log) and lab_beams.txt without the lines of the snapshots not written then.
    void restore(Simulation& s) {
        const bool r0 = s.comm.rank == 0;
        const LabDiagConfig& L = s.cfg.labdiag;
        const size_t nt = done_.size(), ns = idx_.size();
        const std::string bpath = s.cfg.dir + "/lab_beams.txt";
        const std::vector<double> sp(idx_.begin(), idx_.end());
        OpState o;
        bool have = read_op_state(s, name(), {"times", "species", "status", "ranks", "beams_bytes", "widths"}, o);
        const char* why = "the checkpoint holds no lab-particle state (written before decision 116)";
        if (have && (*o.get("times") != L.times || *o.get("species") != sp)) {
            have = false;
            why = "lab_diagnostics has other lab times or species than the checkpoint's run";
        }
        if (have) {                                      // record widths (decision 117; absent before it: 7 for all)
            const std::vector<double> wd = *o.get("widths");
            const std::vector<double> mine(rw_.begin(), rw_.end());
            if (wd.empty() ? std::any_of(rw_.begin(), rw_.end(), [](size_t w) { return w != 7; }) : wd != mine) {
                have = false;
                why = "the checkpoint's lab-particle records have other columns (spin, photons) than this run's species";
            }
        }
        if (have && (o.get("status")->size() != nt || o.get("ranks")->size() != 1)) { have = false; why = "the checkpoint's lab-particle state is inconsistent"; }
        if (have) {
            const std::vector<double> st = *o.get("status");
            const int R = int(o.get("ranks")->at(0));
            const bool same = R == s.comm.size;
            const std::vector<double> bb = *o.get("beams_bytes");
            std::vector<std::string> keys;
            for (size_t k = 0; k < nt; ++k)
                for (size_t q = 0; q < ns && st[k] != 1; ++q) {
                    const std::string kq = std::to_string(k) + "_" + std::to_string(q);
                    if (same || r0) { keys.push_back("rec_" + kq); keys.push_back("cnt_" + kq); }
                }
            read_op_state(s, name(), keys, o);
            for (size_t k = 0; k < nt; ++k) {
                done_[k] = st[k] == 1;
                lost_[k] = st[k] == 2;
                if (done_[k]) continue;
                double nrec = 0;
                for (size_t q = 0; q < ns && (same || r0); ++q) {
                    const std::string kq = std::to_string(k) + "_" + std::to_string(q);
                    const std::vector<double>& all = *o.get("rec_" + kq);
                    const std::vector<double>& cnt = *o.get("cnt_" + kq);
                    if (all.empty()) continue;
                    if (cnt.size() != size_t(R)) throw std::runtime_error("checkpoint: lab particle counts of another number of ranks");
                    double before = 0, n = 0;
                    for (int r = 0; r < R; ++r) n += cnt[size_t(r)];
                    if (same) {
                        for (int r = 0; r < s.comm.rank; ++r) before += cnt[size_t(r)];
                        n = cnt[size_t(s.comm.rank)];
                    }
                    const double rw = double(rw_[q]);
                    if (size_t(rw * (before + n)) > all.size()) throw std::runtime_error("checkpoint: lab particle records cut short");
                    rec_[k * ns + q].assign(all.begin() + long(rw * before), all.begin() + long(rw * (before + n)));
                    nrec += double(all.size() / rw_[q]);
                }
                if (r0 && (nrec > 0 || lost_[k]))
                    std::printf("  lab_diagnostics: particle snapshot %zu restored from the checkpoint (%.0f records%s)%s\n", k, nrec,
                                same ? "" : ", all on rank 0: another number of ranks", lost_[k] ? ", incomplete since an earlier restart (complete 0)" : "");
            }
            if (r0) {                                    // lab_beams.txt as at the checkpoint
                std::string txt = read_text(bpath);
                const size_t n = bb.empty() ? txt.size() : size_t(bb[0]);
                if (bb.empty() || txt.size() < n)
                    std::fprintf(stderr, "warning: lab_diagnostics: %s is shorter than at the checkpoint (%zu < %zu bytes): continued as it is\n",
                                 bpath.c_str(), txt.size(), n);
                else txt.resize(n);
                beams_ = open_or_throw(bpath, "w");
                std::fputs(txt.c_str(), beams_);
                std::fflush(beams_);
            }
            return;
        }
        const Domain& d = s.dom;
        const BoostConfig& B = s.cfg.boost;
        const double t1 = double(s.istep) * d.dt(), h = d.h[0];          // the last step before the checkpoint (apply)
        const double xl = double(d.wx - 1) * h, xr = double(d.wx + d.N[0] + 16) * h;   // beyond the injection plane
        for (size_t k = 0; k < nt; ++k) {
            const double T = L.times[k];
            if (B.t_lab(xl, t1) > T) done_[k] = 1;                       // written before the checkpoint
            else if (B.t_lab(xr, t1) > T) {
                lost_[k] = 1;
                if (r0)
                    std::fprintf(stderr, "warning: lab_diagnostics: particle snapshot %zu (t_lab %.6g fs) may have records from before the "
                                         "checkpoint that cannot be restored (%s): it is written with complete 0\n", k,
                                 T / (c_si * 2 * M_PI / s.cfg.lambda0) * 1e15, why);
            }
        }
        if (r0) {                                        // keep the lines of the snapshots written before the checkpoint
            const std::string txt = read_text(bpath);
            std::string keep, line;
            for (char ch : txt) {
                line.push_back(ch);
                if (ch != '\n') continue;
                const long k = std::strtol(line.c_str(), nullptr, 10);
                if (line[0] == '#' || (k >= 0 && size_t(k) < nt && done_[size_t(k)])) keep += line;
                line.clear();
            }
            if (keep.empty())
                keep = "# lab-frame beams of a boosted run at the lab times of lab_diagnostics (continued after a restart)\n";
            beams_ = open_or_throw(bpath, "w");
            std::fputs(keep.c_str(), beams_);
            std::fflush(beams_);
        }
    }
    static std::string read_text(const std::string& path) {
        std::string t;
        if (FILE* f = std::fopen(path.c_str(), "rb")) {
            char b[65536];
            for (size_t n; (n = std::fread(b, 1, sizeof b, f)) > 0;) t.append(b, n);
            std::fclose(f);
        }
        return t;
    }
    void finalize(Simulation& s, size_t k, bool complete) {
        done_[k] = 1;
        complete = complete && !lost_[k];
        const LabDiagConfig& L = s.cfg.labdiag;
        const double k0 = 2 * M_PI / s.cfg.lambda0, w0 = c_si * k0, um = 1e6 / k0;
        const double n_c = eps0_si * me_si * w0 * w0 / (e_si * e_si);
        for (size_t q = 0; q < idx_.size(); ++q) {
            std::vector<double>& v = rec_[k * idx_.size() + q];
            const Species& sp = s.species[size_t(idx_[q])];
            Species lab;                                         // the lab particles of this rank
            lab.q = sp.q; lab.m = sp.m;
            const size_t rw = rw_[q];
            std::vector<double> h[Species::NC];
            for (size_t p = 0; p + rw <= v.size(); p += rw)
                for (int a = 0; a < Species::NC; ++a) h[a].push_back(v[p + size_t(a)]);
            lab.append(h);
            const BeamMoments m = beam_moments(lab, L.gamma_min, s.comm);
            const double qpc = sp.q * e_si * n_c / (k0 * k0 * k0) * 1e12, mev = sp.m * mec2_mev;
            if (s.comm.rank == 0) {
                const double emean = m.wsum > 0 ? (m.gmean - 1) * mev : 0.0;
                std::fprintf(beams_, "%zu %.10g %s %d %.0f %.10g %.10g %.6g %.10g %.6g %.6g %.10g %.6g %.6g %.6g\n", k,
                             L.times[k] / w0 * 1e15, sp.name.c_str(), complete ? 1 : 0, m.count, m.wsum * std::abs(qpc), emean,
                             emean > 0 ? m.gsig * mev / emean : 0.0, m.count > 0 ? (m.gmax - 1) * mev : 0.0, m.eps_y * um,
                             m.eps_z * um, m.xmean * um, m.sx * um, m.sy * um, m.sz * um);
                std::fflush(beams_);
            }
            if (L.particles) {                                   // all records to rank 0 (in rank order)
                const double nloc = double(v.size() / rw), dw = double(rw);
                std::vector<double> cnt(size_t(s.comm.size));
                s.comm.allgather(&nloc, 1, cnt.data());
                double before = 0, tot = 0;
                for (int r = 0; r < s.comm.size; ++r) { if (r < s.comm.rank) before += cnt[size_t(r)]; tot += cnt[size_t(r)]; }
                std::vector<double> all(size_t(dw * tot), 0.0);
                std::copy(v.begin(), v.begin() + long(dw * nloc), all.begin() + long(dw * before));
                for (size_t o = 0; o < all.size(); o += size_t(1) << 26)
                    s.comm.allreduce_sum(all.data() + o, int(std::min(all.size() - o, size_t(1) << 26)));
                if (s.comm.rank == 0) {
                    char base[96];
                    std::snprintf(base, sizeof base, "/lab_particles/%s_%03zu", sp.name.c_str(), k);
                    FILE* f = open_or_throw(s.cfg.dir + base + ".bin", "wb");
                    std::fwrite(all.data(), sizeof(double), all.size(), f);
                    std::fclose(f);
                    FILE* m2 = open_or_throw(s.cfg.dir + base + ".txt", "w");
                    const char* extra = rw == 13 ? " xi1 xi2 xi3 pe0 pe1 pe2" : rw == 10 ? " s0 s1 s2" : "";   // decision 117
                    std::fprintf(m2, "t_lab %.12g\nt_lab_fs %.10g\ncomplete %d\ncount %.0f\ncolumns x y z ux uy uz w%s (float64, lab frame, "
                                     "code units)\nlength_um %.10g\nweight_pC %.10g\ncharge %g\nmass %g\nlambda0_m %.6g\n",
                                 L.times[k], L.times[k] / w0 * 1e15, complete ? 1 : 0, tot, extra, um, qpc, sp.q, sp.m, s.cfg.lambda0);
                    if (sp.photon) std::fprintf(m2, "photon 1 (massless: the energy is |u|; Stokes vector in the basis (pe, n x pe))\n");
                    std::fclose(m2);
                }
            }
            v.clear(); v.shrink_to_fit();
        }
    }
};

}  // namespace

std::unique_ptr<Operator> make_lab_field_operator() { return std::make_unique<LabFields>(); }
std::unique_ptr<Operator> make_lab_particle_operator() { return std::make_unique<LabParticles>(); }
