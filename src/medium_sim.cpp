// O7: the media in the Simulation (medium.hpp, docs/specs/linear_medium.md, decision 84): checks, setup, dir/medium.txt.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <stdexcept>
#include "fsutil.hpp"
#include "sim.hpp"

void Config::check_media() const {
    auto bad = [](const std::string& w) { throw std::runtime_error("medium: " + w + " (docs/specs/linear_medium.md)"); };
    if (solver != Solver::IRip) bad("numerics.solver = irip");
    if (o5) bad("numerics.closure = box (the work field of o5 is not the box closure's)");
    for (const LaserConfig& L : all_lasers())
        if (L.enabled && L.tfsf_x >= 0) bad("no TFSF laser yet");
    if (side.on) bad("no side tube yet");
    if (!(medium_tol > 0) || medium_maxit < 1 || medium_restart < 1) bad("medium_tol > 0, medium_maxit >= 1, medium_restart >= 1");
    if (rz && mover_implicit) bad("geometry = rz: not with numerics.mover = implicit yet (Cart3D only)");
    if (!(medium_bound_tol > 0) || medium_bound_maxit < 1) bad("medium_bound_tol > 0, medium_bound_maxit >= 1");
    int nb = 0;
    for (const MediumConfig& m : media) nb += m.bound ? 1 : 0;
    if (nb > 1) bad("one block with model = bound (yet)");
}

// RZ: the volumes of the current positions per component (Jz, Jphi at the nodes: 2 pi A_j dz; Jr at the half nodes: 2 pi B_j dz)
static std::vector<double> rz_weights(const FieldsRZ& F, const Domain& d) {
    const RadialGrid& g = F.grid();
    const int nr = d.n[1];
    std::vector<double> w(size_t(3 * nr));
    for (int j = 0; j < nr; ++j) {
        w[size_t(j)] = 2 * M_PI * g.A[size_t(j)] * d.h[0];
        w[size_t(nr + j)] = 2 * M_PI * g.B[size_t(j)] * d.h[0];
        w[size_t(2 * nr + j)] = 2 * M_PI * g.A[size_t(j)] * d.h[0];
    }
    return w;
}

void Simulation::setup_medium() {
    medium = std::make_unique<Medium>();
    Medium& M = *medium;
    M.tol = cfg.medium_tol; M.maxit = cfg.medium_maxit; M.restart = cfg.medium_restart;
    M.btol = cfg.medium_bound_tol; M.bmaxit = cfg.medium_bound_maxit;
    double mx;
    if (cfg.rz) {
        const RadialGrid& g = rz->grid();
        mx = medium_build(M, dom, cfg.media, 2, g.r.data(), g.rh.data());
        M.weights = rz_weights(*rz, dom);
        if (M.bound) {                               // the atoms on the azimuthal samples (medium_rz_bound.cpp)
            int np = 0;
            for (const MediumConfig& m : cfg.media) if (m.bound) np = m.nphi;
            medium_bound_rz_setup(M, dom, g.view(), cfg.modes, np > 0 ? np : 4 * cfg.modes);
            const std::vector<double> nb = M.bn.host();     // no atoms at the last radial node (Et_phi from r_{j+1}), nor in the absorber
            const int jmax = dom.n[1] - 1 - std::max(0, cfg.rz_abs_cells);
            double bad = 0;
            for (int j = std::max(0, jmax); j < dom.n[1]; ++j)
                for (int i = 0; i < dom.n[0]; ++i)
                    if (nb[M.bn.idx(i, j, 0)] != 0) bad = 1;
            if (comm.allreduce_max(bad) > 0) throw std::runtime_error("medium (bound, rz): atoms in the last radial cells or the absorber");
        }
    } else mx = medium_build(M, dom, cfg.media, cfg.flat ? 1 : 0);
    medium_wp2_max = comm.allreduce_max(mx);
    if (cfg.rz && cfg.rz_abs_cells > 0) {            // the radial absorber damps the fields only: keep the media out of it
        double in = 0;
        for (int c = 0; c < 3; ++c) {
            const std::vector<double> w = M.wp2[c].host();
            for (int j = std::max(0, dom.n[1] - cfg.rz_abs_cells - 1); j < dom.n[1]; ++j)
                for (int i = 0; i < dom.n[0]; ++i)
                    if (w[M.wp2[c].idx(i, j, 0)] != 0) in = 1;
        }
        if (comm.allreduce_max(in) > 0) throw std::runtime_error("medium: inside the radial absorber (keep one cell between them)");
    }
    // the CPML layers must stay free of media (the work field there is not the box closure's)
    if (!cfg.rz && (cfg.pml.on[1] || cfg.pml.on[2])) {
        double in = 0;
        for (int c = 0; c < 3; ++c) {
            const std::vector<double> w = M.wp2[c].host();
            for (int k = 0; k < dom.n[2]; ++k)
                for (int j = 0; j < dom.n[1]; ++j)
                    for (int i = 0; i < dom.n[0]; ++i) {
                        const int jj = j + dom.off[1], kk = k + dom.off[2], L = cfg.pml.cells + 1;
                        const bool lay = (cfg.pml.on[1] && (jj < L || jj >= dom.N[1] - L)) ||
                                         (cfg.pml.on[2] && !dom.flat && (kk < L || kk >= dom.N[2] - L));
                        if (lay && w[M.wp2[c].idx(i, j, k)] != 0) in = 1;
                    }
        }
        if (comm.allreduce_max(in) > 0) throw std::runtime_error("medium: inside a pml layer (keep one cell between the media and the layers)");
    }
    for (const MediumConfig& m : cfg.media)            // ionization into species: they must exist (electrons q < 0, ions q > 0)
        if (m.bound && m.ionize && !m.electrons.empty()) {
            int ok = 0;
            for (const SpeciesConfig& sp : cfg.species) {
                if (sp.name == m.electrons && sp.q < 0) ++ok;
                if (sp.name == m.ions && sp.q > 0) ++ok;
            }
            if (ok != 2) throw std::runtime_error("medium (bound): electrons = " + m.electrons + " (charge < 0) and ions = " + m.ions +
                                                 " (charge > 0) must be species of the deck");
        }
    if (cfg.rz) rz->medium = &M;
    else fields->medium = &M;
}

double Simulation::medium_energy() {
    double k = medium->energy(dom, cfg.rz ? &medium->weights : nullptr);
    if (cfg.rz && medium->bound) k += medium_bound_rz_energy(*medium, dom, rz->grid().view());   // the atoms on the angles
    comm.allreduce_sum(&k, 1);
    return k;
}

void medium_diagnostics(Simulation& s, bool root, double wtot) {
    const double K = s.medium_energy();
    double emitted = s.medium->emitted;                // ionization: the pairs emitted, summed over the ranks (collective)
    s.comm.allreduce_sum(&emitted, 1);
    if (!root) return;
    const Medium& M = *s.medium;
    if (!s.medium_f) {
        bool fresh;
        s.medium_f = open_log(s.cfg.dir + "/medium.txt", s.restart_step, fresh);
        if (fresh)
            std::fprintf(s.medium_f, "# step  t  K_medium  W_dissipated  W_field+W_kin+K+W_dissipated  solves  iterations  "
                                     "max_iterations  unconverged  last_residual%s\n",
                         M.bound ? (M.bc.ion ? "  newton_total  newton_max  newton_unconverged (model = bound: outer iterations)  ionized  emitted"
                                             : "  newton_total  newton_max  newton_unconverged (model = bound: outer iterations)") : "");
    }
    const MediumStats& st = M.stats;
    std::fprintf(s.medium_f, "%ld %.6e %.12e %.12e %.12e %ld %ld %ld %ld %.3e", s.istep, s.t, K, M.dissipated, wtot + K + M.dissipated,
                 st.solves, st.iterations, st.max_iterations, st.unconverged, st.last_residual);
    if (M.bound) std::fprintf(s.medium_f, " %ld %ld %ld", st.newton, st.newton_max, st.newton_unconverged);
    if (M.bound && M.bc.ion) std::fprintf(s.medium_f, " %.12e %.12e", M.ionized, emitted);
    std::fprintf(s.medium_f, "\n");
    std::fflush(s.medium_f);
}

void medium_summary(FILE* f, const Config& c) {
    for (const MediumConfig& m : c.media) {
        if (m.bound) {                               // weak-field chi(omega0) = n_a sum f_j / (w_j^2 - 1) (decision 85)
            double chi = 0;
            for (int j = 0; j < m.lines; ++j) chi += m.density * m.bf[j] / (m.bw[j] * m.bw[j] - 1);
            std::fprintf(f, "  medium%s%s: bound (V-type Maxwell-Bloch atom), n_a = %g n_c, dipole along %c, %d line(s):", m.name.empty() ? "" : " ",
                         m.name.c_str(), m.density, 'x' + m.axis, m.lines);
            for (int j = 0; j < m.lines; ++j)
                std::fprintf(f, " %.6g omega0 (f %.3g, d %.3g e a0, omega dt %.3g)", m.bw[j], m.bf[j], m.bd[j], m.bw[j] * c.dt());
            std::fprintf(f, "; weak-field chi(omega0) = %.4g; supersample %d, profile %s\n", chi, m.supersample,
                         m.profile.empty() ? "1" : m.profile.text().c_str());
            continue;
        }
        const double wp2 = m.density, w02 = m.w0 * m.w0;
        const std::complex<double> eps = 1.0 + wp2 / std::complex<double>(w02 - 1, -m.nu);   // eps(omega0) = 1 + wp2 / (w0^2 - 1 - i nu)
        std::fprintf(f, "  medium%s%s: %s, wp^2 = %g omega0^2 (omega_p dt = %.3g), nu = %g omega0%s, eps(omega0) = %.4g%+.4gi, "
                        "supersample %d, profile %s\n", m.name.empty() ? "" : " ", m.name.c_str(), m.w0 > 0 ? "lorentz" : "drude", wp2,
                     std::sqrt(wp2) * c.dt(), m.nu, m.w0 > 0 ? (", resonance " + std::to_string(m.w0) + " omega0").c_str() : "",
                     eps.real(), eps.imag(), m.supersample, m.profile.empty() ? "1" : m.profile.text().c_str());
    }
    if (!c.media.empty())
        std::fprintf(f, "  medium solve: GMRES(%d), tol %g, at most %d iterations per step (decision 84)\n", c.medium_restart, c.medium_tol,
                     c.medium_maxit);
}

// model = bound with ionization = adk and the species electrons, ions: the ionized density accumulated per atom position becomes an
// electron and an ion at rest at that position once it reaches the quantum n_a / ionization_ppc (weight = the accumulated density
// times the cell volume); the pair deposits no charge, so the Gauss law is untouched; below the quantum the density waits (the
// atoms' polarization already lost it). After the field step, before the migration.
namespace {
class MediumIonize : public Operator {
public:
    const char* name() const override { return "medium_ionize"; }
    Stage stage() const override { return Stage::Migrate; }
    void apply(Simulation& s) override { medium_emit(s); }
};
}  // namespace

std::unique_ptr<Operator> make_medium_ionize_operator() { return std::make_unique<MediumIonize>(); }

void medium_emit(Simulation& s) {
    Medium& M = *s.medium;
    const MediumConfig* mc = nullptr;
    for (const MediumConfig& m : s.cfg.media) if (m.bound) mc = &m;
    if (!mc || !mc->ionize || mc->electrons.empty()) return;
    int ie = -1, ii = -1;
    for (size_t q = 0; q < s.species.size(); ++q) {
        if (s.species[q].name == mc->electrons) ie = int(q);
        if (s.species[q].name == mc->ions) ii = int(q);
    }
    if (ie < 0 || ii < 0) throw std::runtime_error("medium (bound): no species " + (ie < 0 ? mc->electrons : mc->ions));
    const Domain& d = s.dom;
    if (s.cfg.rz) {                                    // RZ: the atoms on the angles (medium_rz_bound.cpp), particles (z, r cos, r sin)
        std::vector<double> acc = M.ra[RA_ACC].host();
        const std::vector<double> nn = M.bn.host();
        const RadialGrid& g = s.rz->grid();
        const int nz = d.n[0], nr = d.n[1], np = M.nphi;
        std::vector<double> h[Species::NC];
        bool any = false;
        for (int k = 0; k < np; ++k)
            for (int j = 0; j < nr; ++j)
                for (int i = 0; i < nz; ++i) {
                    const size_t q = (size_t(k) * nr + j) * nz + i;
                    const double n = nn[M.bn.idx(i, j, 0)], a = acc[q];
                    if (n == 0.0 || !(a >= n / mc->ion_ppc)) continue;
                    const double ph = 2 * M_PI * k / np, r = g.rh[size_t(j)];
                    const double w = a * 2 * M_PI * g.B[size_t(j)] * d.h[0] / np;
                    h[Species::X].push_back(d.lo(0) + (i + 0.5) * d.h[0]);
                    h[Species::Y].push_back(r * std::cos(ph));
                    h[Species::Z].push_back(r * std::sin(ph));
                    h[Species::UX].push_back(0); h[Species::UY].push_back(0); h[Species::UZ].push_back(0);
                    h[Species::W].push_back(w);
                    acc[q] = 0;
                    any = true;
                    M.emitted += w;
                }
        if (!any) return;
        M.ra[RA_ACC].from_host(acc.data(), 0, acc.size());
        s.species[size_t(ie)].append(h);
        s.species[size_t(ii)].append(h);
        return;
    }
    std::vector<double> acc = M.bacc.host();
    const std::vector<double> nn = M.bn.host();
    const int c = M.bax;
    const double ox = c == 0 ? 0.0 : 0.5, oy = c == 1 ? 0.5 : 0.0, oz = d.flat || c == 2 ? 0.5 : 0.0, dv = d.cell_volume();
    std::vector<double> h[Species::NC];
    bool any = false;
    for (int k = 0; k < d.n[2]; ++k)
        for (int j = 0; j < d.n[1]; ++j)
            for (int i = 0; i < d.n[0]; ++i) {
                const size_t q = M.bacc.idx(i, j, k);
                const double n = nn[q], a = acc[q];
                if (n == 0.0 || !(a >= n / mc->ion_ppc)) continue;
                h[Species::X].push_back(d.lo(0) + (i + ox) * d.h[0]);
                h[Species::Y].push_back(d.lo(1) + (j + oy) * d.h[1]);
                h[Species::Z].push_back(d.lo(2) + (k + oz) * d.h[2]);
                h[Species::UX].push_back(0); h[Species::UY].push_back(0); h[Species::UZ].push_back(0);
                h[Species::W].push_back(a * dv);
                acc[q] = 0;
                any = true;
                M.emitted += a * dv;
            }
    if (!any) return;
    M.bacc.upload(acc);
    s.species[size_t(ie)].append(h);
    s.species[size_t(ii)].append(h);
}

// Moving window (after ++dom.wx, before the iRIP entry cell): every array of the media state moves one cell towards -x like the
// fields (the coefficients wp2, w02, nu, J, P; the atoms' density matrix, n_a, the last Et, their polarization, the ionization
// accumulator; RZ: the lagged atom current and the polarization per mode, the atoms on the angles), and the entering column (and
// the x guards beyond it) is rebuilt from the profiles at the new position: linear media at rest (J = P = 0), atoms in the ground
// state with no polarization.
void Simulation::shift_window_medium() {
    Medium& M = *medium;
    std::vector<Field*> f;
    for (int c = 0; c < 3; ++c) { f.push_back(&M.wp2[c]); f.push_back(&M.w02[c]); f.push_back(&M.nu[c]); f.push_back(&M.J[c]); f.push_back(&M.P[c]); }
    std::vector<Field*> st;                            // the state of the atoms (reset in the entering column)
    if (M.bound && cfg.rz) st = {&M.bLr, &M.bLp, &M.bPr, &M.bPp};
    else if (M.bound) {
        for (int q = 0; q < BS_N; ++q) st.push_back(&M.bs[q]);
        st.push_back(&M.bE); st.push_back(&M.bP);
        if (M.bc.ion) st.push_back(&M.bacc);
    }
    if (M.bound) f.push_back(&M.bn);
    for (Field* x : st) f.push_back(x);
    comm.exchange_x(dom, f);
    shift_fields_x(dom, f, work_);
    const bool right = dom.off[0] + dom.n[0] == dom.N[0];
    const int nx = dom.n[0], g = dom.ng, sy = dom.n[1] + 2 * g, sz = dom.n[2] + 2 * dom.gw(2);
    if (right) {                                       // the entering column and the guards beyond: zero, then from the profiles
        for (Field* x : f) {
            const FV v = x->view();
            const int gz = dom.gw(2);
            par_for(long(sy) * sz, [=] HD(long q) {
                const int j = int(q % sy) - g, k = int(q / sy) - gz;
                for (int i = nx - 1; i < nx + g; ++i) v(i, j, k) = 0.0;
            });
        }
        if (M.bound && !cfg.rz) {
            const FV p0 = M.bs[BS_P0].view();
            for_owned(dom, [=] HD(int i, int j, int k) { if (i == nx - 1) p0(i, j, k) = 1.0; });
        }
        if (cfg.rz) {
            const RadialGrid& gr = rz->grid();
            medium_build(M, dom, cfg.media, 2, gr.r.data(), gr.rh.data(), nx - 1);
        } else medium_build(M, dom, cfg.media, cfg.flat ? 1 : 0, nullptr, nullptr, nx - 1);
    }
    comm.exchange_x(dom, f);
    if (M.bound && cfg.rz) {                           // the atoms on the angles: raw arrays [q][k][j][i] of the owned cells
        const int nr = dom.n[1], np = M.nphi;
        const int qs[] = {BS_P0, BS_P1, BS_P2, BS_R01, BS_I01, BS_R02, BS_I02, BS_R12, BS_I12, RA_E, RA_ACC};
        const int nq = int(sizeof(qs) / sizeof(qs[0]));
        std::vector<std::vector<double>> h(static_cast<size_t>(nq));
        std::vector<unsigned char> send[2], recv[2];
        const size_t col = size_t(nr) * np;
        std::vector<double> c0(col * nq);
        for (int a = 0; a < nq; ++a) {
            h[size_t(a)] = M.ra[size_t(qs[a])].host();
            for (size_t q = 0; q < col; ++q) c0[a * col + q] = h[size_t(a)][q * nx];      // the column i = 0
        }
        send[0].resize(c0.size() * 8);
        std::memcpy(send[0].data(), c0.data(), send[0].size());
        comm.exchange_bytes(0, send, recv, 7300);      // the left neighbour gets this column 0 as its new last column
        const bool got = recv[0].size() == c0.size() * 8;
        std::vector<double> cr(got ? c0.size() : 0);
        if (got) std::memcpy(cr.data(), recv[0].data(), recv[0].size());
        for (int a = 0; a < nq; ++a) {
            std::vector<double>& v = h[size_t(a)];
            for (size_t q = 0; q < col; ++q) {
                double* row = v.data() + q * nx;
                for (int i = 0; i + 1 < nx; ++i) row[i] = row[i + 1];
                row[nx - 1] = got ? cr[a * col + q] : (qs[a] == BS_P0 ? 1.0 : 0.0);   // the entering atoms: ground state
            }
            M.ra[size_t(qs[a])].from_host(v.data(), 0, v.size());
        }
    }
}
