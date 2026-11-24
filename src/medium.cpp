// O7: linear-response media for iRIP (medium.hpp, docs/specs/linear_medium.md, decision 84).
#include "env.hpp"
#include "medium.hpp"
#include "deck.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

MediumConfig medium_block(const Deck& d, const std::string& b, double k0, bool rz) {
    MediumConfig m;
    m.name = b.size() > 7 ? b.substr(7) : "";
    const std::string model = d.str(b, "model", "drude");
    if (model != "drude" && model != "lorentz" && model != "bound") throw std::runtime_error(b + ".model: drude, lorentz or bound");
    if (model == "bound") {                          // a V-type Maxwell-Bloch atom (bloch.hpp, decision 85)
        for (const char* k : {"collision_frequency", "resonance", "permittivity"})
            if (d.has(b, k)) throw std::runtime_error(b + "." + k + ": not with model = bound (lines: atom or wavelength1, strength1, ...)");
        m.bound = true;
        double lam[2] = {0, 0}, f[2] = {0, 0};
        std::string atom = d.str(b, "atom", "");
        for (char& ch : atom) ch = char(std::tolower(static_cast<unsigned char>(ch)));
        if (atom == "rb") {                          // D. A. Steck, Rb D-line data: 5S - 5P_1/2 (D1), 5S - 5P_3/2 (D2)
            lam[0] = 794.979e-9; f[0] = 0.342; lam[1] = 780.241e-9; f[1] = 0.695; m.lines = 2;
        } else if (!atom.empty()) throw std::runtime_error(b + ".atom: only Rb is built in (else wavelength1, strength1[, wavelength2, strength2])");
        for (int j = 0; j < 2; ++j) {
            const std::string w = "wavelength" + std::to_string(j + 1), s = "strength" + std::to_string(j + 1);
            if (d.has(b, w) != d.has(b, s)) throw std::runtime_error(b + ": " + w + " and " + s + " together");
            if (d.has(b, w)) { lam[j] = d.num(b, w, Kind::Length); f[j] = d.num(b, s, Kind::None); m.lines = std::max(m.lines, j + 1); }
        }
        if (m.lines == 0 || (m.lines == 2 && !(lam[0] > 0))) throw std::runtime_error(b + ": atom = Rb, or wavelength1, strength1[, wavelength2, strength2]");
        for (int j = 0; j < m.lines; ++j) {
            if (!(lam[j] > 0 && f[j] > 0)) throw std::runtime_error(b + ": wavelengths and strengths must be > 0");
            m.bw[j] = 2 * M_PI / lam[j] / k0;
            m.bf[j] = f[j];
            // d^2 / (e a0)^2 = f hbar / (2 m omega a0^2) = f E_h / (2 hbar omega), hbar omega [eV] = 1.239842e-6 m / lambda
            m.bd[j] = std::sqrt(f[j] * 27.211386245988 / (2 * 1.23984198e-6 / lam[j]));
        }
        std::string ds = d.str(b, "density");
        const size_t pc = ds.find("cm^-3");
        if (pc != std::string::npos) {               // atoms per cm^3
            double fc = 0;
            unit_to_code("cm^-3", k0, fc);
            m.density = std::stod(ds.substr(0, pc)) * fc;
        } else m.density = d.num(b, "density", Kind::None);
        if (!(m.density > 0)) throw std::runtime_error(b + ".density (atoms, n_a / n_c or '<n> cm^-3') must be > 0");
        const std::string pol = d.str(b, "polarization", rz ? "x" : "z");
        if (pol != "x" && pol != "y" && pol != "z") throw std::runtime_error(b + ".polarization (the dipole axis): x, y or z");
        m.nphi = int(d.integer(b, "n_phi", 0));        // RZ: azimuthal samples of the atoms (0: 4 modes)
        m.axis = pol[0] - 'x';
        if (rz && m.axis != 0) throw std::runtime_error(b + ".polarization: x in RZ (the laser polarization; dipole along x)");
        m.a0k0 = 5.29177210903e-11 * k0;
        const double ck0 = 299792458.0 * k0;           // seconds -> 1 / omega0
        m.bT1 = d.num(b, "lifetime", Kind::Time, 0.0) * ck0;
        m.bT2 = d.num(b, "dephasing_time", Kind::Time, 0.0) * ck0;
        if (m.bT1 < 0 || m.bT2 < 0) throw std::runtime_error(b + ": lifetime, dephasing_time >= 0 (0: none)");
        const std::string ion = d.str(b, "ionization", "none");
        if (ion != "none" && ion != "adk") throw std::runtime_error(b + ".ionization: none or adk");
        m.ionize = ion == "adk";
        if (m.ionize) {
            if (atom == "rb") { m.ip[0] = 4.177128; m.ip[1] = 4.177128 - 1.559591; m.ip[2] = 4.177128 - 1.589049; }   // NIST: 5S, 5P_1/2, 5P_3/2
            if (d.has(b, "ionization_potentials")) {          // eV: the ground state, then every excited state
                std::string t = d.str(b, "ionization_potentials");
                for (char& ch : t) if (ch == ',') ch = ' ';
                std::vector<double> v;
                size_t p = 0;
                while (true) {
                    const size_t a = t.find_first_not_of(" 	", p);
                    if (a == std::string::npos) break;
                    const size_t e = t.find_first_of(" 	", a);
                    const std::string tok = t.substr(a, e == std::string::npos ? std::string::npos : e - a);
                    if (tok != "eV") v.push_back(std::stod(tok));
                    if (e == std::string::npos) break;
                    p = e;
                }
                if (int(v.size()) != m.lines + 1) throw std::runtime_error(b + ".ionization_potentials: the ground state and every excited state (eV)");
                for (int k = 0; k <= m.lines; ++k) m.ip[k] = v[size_t(k)];
            }
            for (int k = 0; k <= m.lines; ++k)
                if (!(m.ip[k] > 0)) throw std::runtime_error(b + ".ionization_potentials needed (eV; Rb is built in)");
            m.electrons = d.str(b, "electrons", "");
            m.ions = d.str(b, "ions", "");
            if (m.electrons.empty() != m.ions.empty()) throw std::runtime_error(b + ": electrons and ions (species names) together");
            m.ion_ppc = int(d.integer(b, "ionization_ppc", 8));
            if (m.ion_ppc < 1) throw std::runtime_error(b + ".ionization_ppc >= 1");
        }
    }
    m.nu = d.num(b, "collision_frequency", Kind::None, 0.0);
    if (model == "bound") {
    } else if (model == "lorentz") {
        m.w0 = d.num(b, "resonance", Kind::None);
        if (d.has(b, "permittivity") == d.has(b, "density")) throw std::runtime_error(b + ": density or permittivity (one of them)");
        if (d.has(b, "permittivity")) {
            if (std::abs(m.w0 - 1) < 1e-6) throw std::runtime_error(b + ".permittivity: the resonance must differ from omega0");
            m.density = (d.num(b, "permittivity", Kind::None) - 1) * (m.w0 * m.w0 - 1);
        } else m.density = d.num(b, "density", Kind::None);
        if (!(m.w0 > 0)) throw std::runtime_error(b + ".resonance must be > 0 (model = drude for a free plasma)");
    } else {
        if (d.has(b, "resonance") || d.has(b, "permittivity")) throw std::runtime_error(b + ": resonance, permittivity need model = lorentz");
        m.density = d.num(b, "density", Kind::None);
    }
    if (!(m.density > 0) || m.nu < 0) throw std::runtime_error(b + ": density (wp^2 / omega0^2) > 0 and collision_frequency >= 0");
    if (d.has(b, "profile")) {
        std::string t = d.str(b, "profile");
        if (t.size() >= 2 && t.front() == '"' && t.back() == '"') t = t.substr(1, t.size() - 2);
        const UnitFn u = [k0](const std::string& n, double& f) { return unit_to_code(n, k0, f); };
        try { m.profile = rz ? Expr(t, {"z", "r"}, u) : Expr(t, {"x", "y", "z"}, u); }
        catch (const std::exception& ex) { throw std::runtime_error(b + ".profile: " + ex.what()); }
    }
    m.supersample = int(d.integer(b, "supersample", 4));
    if (m.supersample < 1 || m.supersample > 16) throw std::runtime_error(b + ".supersample: 1 .. 16");
    return m;
}

void Medium::resize(const Domain& d) {
    for (int c = 0; c < 3; ++c) {
        wp2[c].resize(d); w02[c].resize(d); nu[c].resize(d); J[c].resize(d); P[c].resize(d); u[c].resize(d); jt[c].resize(d);
    }
    sx.resize(d); sy.resize(d); gx.resize(d); gy.resize(d);
}

namespace {
// the sub-cell average of the profile around the current position of component c (code units, absolute). geom 0: Cart3D,
// 1: cartesian2d (z not sampled), 2: RZ (z, r; volume-weighted in r between the neighbouring half nodes / nodes).
struct BuildC {
    int geom, S, ncode;
    double h[3], lo[3];
    double base, nu, w02;
    int i0;                      // only the cells i >= i0 (the entering column of a moving window)
};
void build_kernel(const Domain& d, const BuildC c, const Expr::Ins* code, int comp, const FV W, const FV N, const FV O,
                  const double* r, const double* rh, int nr) {
    for_owned(d, [=] HD(int i, int j, int k) {
        if (i < c.i0) return;
        if (c.geom == 2 && k != 0) return;                       // RZ: one coefficient per (z, r), copied to the slots below
        double ox, oy, oz;                                       // the position of the component in cells
        if (c.geom == 2) { ox = comp == 0 ? 0.0 : 0.5; oy = 0; oz = 0; }
        else { ox = comp == 0 ? 0.0 : 0.5; oy = comp == 1 ? 0.5 : 0.0; oz = comp == 2 ? 0.5 : 0.0; }
        const int S = c.S, Sz = c.geom == 0 ? S : 1;
        double acc = 0, wsum = 0;
        double ra = 0, rb = 0;                                   // RZ: the radial extent of the sampling cell
        if (c.geom == 2) {
            if (comp == 1) { ra = r[j]; rb = r[j + 1]; }        // Jr at the half node rh[j]: between the nodes
            else { ra = j > 0 ? rh[j - 1] : 0.0; rb = j < nr ? rh[j] : r[j]; }   // Jz, Jphi at the node r[j]
        }
        for (int a = 0; a < S; ++a)
            for (int b = 0; b < S; ++b)
                for (int e = 0; e < Sz; ++e) {
                    const double fa = (a + 0.5) / S - 0.5, fb = (b + 0.5) / S - 0.5;
                    double v[3], wt = 1;
                    v[0] = c.lo[0] + (i + ox + fa) * c.h[0];
                    if (c.geom == 2) {
                        const double rr = ra + (b + 0.5) / S * (rb - ra);
                        v[1] = rr; v[2] = 0; wt = rr > 0 ? rr : 0.0;
                        if (rb <= ra) wt = 1;
                    } else {
                        v[1] = c.lo[1] + (j + oy + fb) * c.h[1];
                        v[2] = c.geom == 0 ? c.lo[2] + (k + oz + (e + 0.5) / Sz - 0.5) * c.h[2] : 0.0;
                    }
                    const double f = c.ncode > 0 ? expr_eval(code, c.ncode, v) : 1.0;
                    acc += wt * f; wsum += wt;
                }
        const double frac = wsum > 0 ? acc / wsum : 0.0;
        const double add = c.base * frac;
        if (add <= 0) return;
        const double w = W(i, j, k);                             // overlapping blocks: wp2-weighted w02 and nu
        N(i, j, k) = (N(i, j, k) * w + c.nu * add) / (w + add);
        O(i, j, k) = (O(i, j, k) * w + c.w02 * add) / (w + add);
        W(i, j, k) = w + add;
    });
}
void copy_slots(const Domain& d, const FV F) {
    for_owned(d, [=] HD(int i, int j, int k) { if (k > 0) F(i, j, k) = F(i, j, 0); });
}
}  // namespace

double medium_build(Medium& M, const Domain& d, const std::vector<MediumConfig>& cfg, int geom, const double* r, const double* rh,
                    int i0) {
    if (i0 == 0) M.resize(d);
    M.rz = geom == 2;
    Buf<Expr::Ins> code;
    Buf<double> rb, rhb;
    if (geom == 2) {
        rb.resize(size_t(d.n[1] + 1)); rhb.resize(size_t(d.n[1]));
        rb.from_host(r, 0, size_t(d.n[1] + 1)); rhb.from_host(rh, 0, size_t(d.n[1]));
    }
    for (const MediumConfig& m : cfg) {
        if (!m.profile.empty() && m.profile.max_stack() > Expr::MAX_STACK) throw std::runtime_error("medium: profile too long");
        const int ncode = int(m.profile.program().size());
        code.resize(size_t(std::max(ncode, 1)));
        if (ncode > 0) code.from_host(m.profile.program().data(), 0, size_t(ncode));
        BuildC c{geom, std::max(1, m.supersample), ncode, {d.h[0], d.h[1], d.h[2]}, {d.lo(0), d.lo(1), d.lo(2)}, m.density, m.nu,
                 m.w0 * m.w0, i0};
        if (m.bound && i0 > 0) {                     // a moving window: n_a of the entering column (its state is reset by the caller)
            build_kernel(d, c, code.data(), geom == 2 ? 1 : m.axis, M.bn.view(), M.bu0.view(), M.bsg.view(), rb.data(), rhb.data(), d.n[1]);
            M.bu0.fill(0); M.bsg.fill(0);
            continue;
        }
        if (m.bound) {                               // the atoms: n_a at the positions of the dipole component, all in the ground state
            if (M.bound) throw std::runtime_error("medium: one block with model = bound (yet)");
            M.bound = true;
            M.bax = m.axis;
            M.bc = bloch_constants(m.lines, m.bw, m.bd, m.a0k0, d.dt());
            bloch_set_relaxation(M.bc, m.bT1, m.bT2, d.dt());
            if (m.ionize) { bloch_set_adk(M.bc, m.ip, m.a0k0 / 5.29177210903e-11); M.bacc.resize(d); }
            for (int j = 0; j < m.lines; ++j)
                if (!(m.bw[j] * d.dt() < 2.0)) throw std::runtime_error("medium (bound): the line needs omega_j dt < 2 (a finer time step)");
            for (int q = 0; q < BS_N; ++q) M.bs[q].resize(d);
            for (Field* f : {&M.bn, &M.bE, &M.bEs, &M.bu0, &M.bsg, &M.bU, &M.bP}) f->resize(d);
            M.bs[BS_P0].fill(1.0);
            build_kernel(d, c, code.data(), geom == 2 ? 1 : m.axis, M.bn.view(), M.bu0.view(), M.bsg.view(), rb.data(), rhb.data(), d.n[1]);   // RZ: the Jr positions
            M.bu0.fill(0); M.bsg.fill(0);
            continue;
        }
        for (int comp = 0; comp < 3; ++comp)
            build_kernel(d, c, code.data(), comp, M.wp2[comp].view(), M.nu[comp].view(), M.w02[comp].view(), rb.data(),
                         rhb.data(), d.n[1]);
    }
    double mx = 0;
    for (int comp = 0; comp < 3; ++comp) {
        if (geom == 2) {
            const FV W = M.wp2[comp].view();
            if (comp == 2) for_owned(d, [=] HD(int i, int j, int k) { if (j == 0 && k == 0) W(i, j, k) = 0; });   // no Jphi on the axis
            copy_slots(d, M.wp2[comp].view()); copy_slots(d, M.nu[comp].view()); copy_slots(d, M.w02[comp].view());
        }
        const FV W = M.wp2[comp].view();
        const int nx = d.n[0], ny = d.n[1];
        mx = std::max(mx, par_max_abs(d.owned(), [=] HD(long q) {
            return W(int(q % nx), int((q / nx) % ny), int(q / (long(nx) * ny)));
        }));
    }
    return mx;
}

// Cart3D: Et0 (the work field without the medium current, decision 81 / src/irip.cpp irip_work_field), then u and jt.
void medium_current_cart(Medium& M, const Domain& d, const FV Ey, const FV Ez, const FV By, const FV Bz, const FV Jx,
                         const FV Jy, const FV Jz, const FV Exs, const FV Bxs) {
    const double D = d.h[0], iy = 1.0 / d.h[1], iz = 1.0 / d.h[2];
    const FV W0 = M.wp2[0].view(), W1 = M.wp2[1].view(), W2 = M.wp2[2].view();
    const FV O0 = M.w02[0].view(), O1 = M.w02[1].view(), O2 = M.w02[2].view();
    const FV N0 = M.nu[0].view(), N1 = M.nu[1].view(), N2 = M.nu[2].view();
    const FV J0 = M.J[0].view(), J1 = M.J[1].view(), J2 = M.J[2].view();
    const FV P0 = M.P[0].view(), P1 = M.P[1].view(), P2 = M.P[2].view();
    const FV U0 = M.u[0].view(), U1 = M.u[1].view(), U2 = M.u[2].view();
    const FV T0 = M.jt[0].view(), T1 = M.jt[1].view(), T2 = M.jt[2].view();
    const bool bnd = M.bound;
    const int c = M.bax;
    const FV BN = bnd ? M.bn.view() : W0, BE = bnd ? M.bE.view() : W0, BEs = bnd ? M.bEs.view() : W0;
    const FV B0 = bnd ? M.bu0.view() : W0, BS = bnd ? M.bsg.view() : W0, BU = bnd ? M.bU.view() : W0;
    for_owned(d, [=] HD(int i, int j, int k) {
        auto mux = [&](const FV& f, int a, int b, int c) { return 0.5 * (f(a, b, c) + f(a + 1, b, c)); };
        auto dxm = [&](int b, int c) { return (Exs(i + 1, b, c) - Exs(i, b, c)) / D; };
        auto cur = [&](double w, double o, double n, double J, double P, double E, double kap) {
            return w == 0.0 ? 0.0 : (2 * J + D * w * E - D * o * P) / (2 + D * n + 0.5 * D * D * o + 0.5 * kap * D * D * w);
        };
        const double psiy = (mux(Bxs, i, j, k) - mux(Bxs, i, j, k - 1)) * iz - Jy(i, j, k);
        const double psiz = -(mux(Bxs, i, j, k) - mux(Bxs, i, j - 1, k)) * iy - Jz(i, j, k);
        const double ex = Exs(i, j, k);
        const double ey = 0.5 * (Ey(i, j, k) + Bz(i, j, k) + Ey(i + 1, j, k) - Bz(i + 1, j, k)) + 0.5 * D * psiy -
                          0.25 * D * D * (dxm(j + 1, k) - dxm(j, k)) * iy;
        const double ez = 0.5 * (Ez(i, j, k) - By(i, j, k) + Ez(i + 1, j, k) + By(i + 1, j, k)) + 0.5 * D * psiz -
                          0.25 * D * D * (dxm(j, k + 1) - dxm(j, k)) * iz;
        const double ux = cur(W0(i, j, k), O0(i, j, k), N0(i, j, k), J0(i, j, k), P0(i, j, k), ex, 0.0);
        const double uy = cur(W1(i, j, k), O1(i, j, k), N1(i, j, k), J1(i, j, k), P1(i, j, k), ey, 1.0);
        const double uz = cur(W2(i, j, k), O2(i, j, k), N2(i, j, k), J2(i, j, k), P2(i, j, k), ez, 1.0);
        U0(i, j, k) = ux; U1(i, j, k) = uy; U2(i, j, k) = uz;
        T0(i, j, k) = Jx(i, j, k) + ux; T1(i, j, k) = Jy(i, j, k) + uy; T2(i, j, k) = Jz(i, j, k) + uz;
        if (!bnd || BN(i, j, k) == 0.0) {
            if (bnd) { BE(i, j, k) = 0; BU(i, j, k) = 0; }
            return;
        }
        // model = bound: the atoms' linearized current u_b = u* + s (Et - Et*) together with a linear medium on the same
        // component, Et = Et0 - kap (dt/2) (u_l + u_b) solved locally (as the elimination in cur)
        const FV& W = c == 0 ? W0 : c == 1 ? W1 : W2;
        const FV& O = c == 0 ? O0 : c == 1 ? O1 : O2;
        const FV& N = c == 0 ? N0 : c == 1 ? N1 : N2;
        const FV& Jm = c == 0 ? J0 : c == 1 ? J1 : J2;
        const FV& Pm = c == 0 ? P0 : c == 1 ? P1 : P2;
        const double e0 = c == 0 ? ex : c == 1 ? ey : ez, kap = c == 0 ? 0.0 : 1.0;
        const double w = W(i, j, k), o = O(i, j, k), den = 2 + D * N(i, j, k) + 0.5 * D * D * o;
        const double al = w == 0.0 ? 0.0 : (2 * Jm(i, j, k) - D * o * Pm(i, j, k)) / den, bl = w == 0.0 ? 0.0 : D * w / den;
        const double sg = BS(i, j, k), ab = B0(i, j, k) - sg * BEs(i, j, k);
        const double et = (e0 - 0.5 * kap * D * (al + ab)) / (1 + 0.5 * kap * D * (bl + sg));
        const double ul = al + bl * et, ub = ab + sg * et;
        const double jx = c == 0 ? Jx(i, j, k) : c == 1 ? Jy(i, j, k) : Jz(i, j, k);
        (c == 0 ? U0 : c == 1 ? U1 : U2)(i, j, k) = ul;
        (c == 0 ? T0 : c == 1 ? T1 : T2)(i, j, k) = jx + ul + ub;
        BE(i, j, k) = et; BU(i, j, k) = ub;
    });
}

namespace {
double commit_comp(const Domain& d, double D, const FV W, const FV N, const FV U, const FV J, const FV P, const double* wt,
                   int c, bool rz) {
    const int nx = d.n[0], ny = d.n[1];
    const double dv = d.cell_volume();
    return par_sum(d.owned(), [=] HD(long q) {
        const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
        const double w = W(i, j, k), u = U(i, j, k);
        J(i, j, k) = 2 * u - J(i, j, k);
        P(i, j, k) += D * u;
        if (w == 0.0) return 0.0;
        const double vol = rz ? wt[c * ny + j] * (k == 0 ? 1.0 : 2.0) : dv;
        return D * N(i, j, k) * u * u / w * vol;
    });
}
double energy_comp(const Domain& d, const FV W, const FV O, const FV J, const FV P, const double* wt, int c, bool rz) {
    const int nx = d.n[0], ny = d.n[1];
    const double dv = d.cell_volume();
    return par_sum(d.owned(), [=] HD(long q) {
        const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
        const double w = W(i, j, k);
        if (w == 0.0) return 0.0;
        const double a = J(i, j, k), p = P(i, j, k);
        const double vol = rz ? wt[c * ny + j] * (k == 0 ? 1.0 : 2.0) : dv;
        return 0.5 * (a * a + O(i, j, k) * p * p) / w * vol;
    });
}
}  // namespace

// RZ: weights[c * nr + j] = the volume of component c at radial index j (the slots of m >= 1
// carry Re, Im with the weight 2 of the field energy of FieldsRZ: |F_0|^2 + 2 sum |F_m|^2)
void medium_commit(Medium& M, const Domain& d, Comm& comm, const std::vector<double>* weights) {
    Buf<double> wt;
    if (weights) { wt.resize(weights->size()); wt.from_host(weights->data(), 0, weights->size()); }
    double diss = 0;
    for (int c = 0; c < 3; ++c)
        diss += commit_comp(d, d.dt(), M.wp2[c].view(), M.nu[c].view(), M.u[c].view(), M.J[c].view(), M.P[c].view(), wt.data(), c,
                            weights != nullptr);
    comm.allreduce_sum(&diss, 1);
    M.dissipated += diss;
}

double Medium::energy(const Domain& d, const std::vector<double>* weights) const {
    Buf<double> wt;
    if (weights) { wt.resize(weights->size()); wt.from_host(weights->data(), 0, weights->size()); }
    double k = 0;
    for (int c = 0; c < 3; ++c)
        k += energy_comp(d, wp2[c].view(), w02[c].view(), J[c].view(), P[c].view(), wt.data(), c, weights != nullptr);
    if (bound) {                                      // the atoms: ef n_a sum_j w_j' p_j (bloch.hpp)
        const BlochC L = bc;
        const FV N = bn.view(), p1 = bs[BS_P1].view(), p2 = bs[BS_P2].view();
        const int nx = d.n[0], ny = d.n[1];
        const double dv = d.cell_volume();
        k += par_sum(d.owned(), [=] HD(long q) {
            const int i = int(q % nx), j = int((q / nx) % ny), kk = int(q / (long(nx) * ny));
            const double n = N(i, j, kk);
            if (n == 0.0) return 0.0;
            double s[BS_N] = {0};
            s[BS_P1] = p1(i, j, kk); s[BS_P2] = p2(i, j, kk);
            return n * bloch_energy(L, s) * dv;
        });
    }
    return k;
}

// rho_med = -div P with the stencil of the RIP-CC continuity at (i+1/2, j, k): Px at (i, j, k), Py at (i+1/2, j+1/2, k),
// Pz at (i+1/2, j, k+1/2); P exchanged first (the lower guards in y, z and the upper one in x)
void medium_add_rho_cart(const Medium& M, const Domain& d, Field& rho) {
    const double ix = 1.0 / d.h[0], iy = 1.0 / d.h[1], iz = 1.0 / d.h[2];
    const FV P0 = M.P[0].view(), P1 = M.P[1].view(), P2 = M.P[2].view(), R = rho.view();
    for_owned(d, [=] HD(int i, int j, int k) {
        R(i, j, k) -= (P0(i + 1, j, k) - P0(i, j, k)) * ix + (P1(i, j, k) - P1(i, j - 1, k)) * iy + (P2(i, j, k) - P2(i, j, k - 1)) * iz;
    });
    if (M.bound) {                                    // the atoms' polarization on its component (bP exchanged by the caller)
        const FV B = M.bP.view();
        const int c = M.bax;
        for_owned(d, [=] HD(int i, int j, int k) {
            R(i, j, k) -= c == 0 ? (B(i + 1, j, k) - B(i, j, k)) * ix : c == 1 ? (B(i, j, k) - B(i, j - 1, k)) * iy
                                                                              : (B(i, j, k) - B(i, j, k - 1)) * iz;
        });
    }
}

// ---------------------------------------------------------------- the linear solve (restarted GMRES, right preconditioning)
namespace {
void pack2(const Domain& d, const Field& a, const Field& b, double* v) {
    const long N = d.owned();
    const int nx = d.n[0], ny = d.n[1];
    const FV A = a.view(), B = b.view();
    par_for(N, [=] HD(long q) {
        const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
        v[q] = A(i, j, k); v[N + q] = B(i, j, k);
    });
}
void unpack2(const Domain& d, const double* v, double s, Field& a, Field& b) {
    const long N = d.owned();
    const int nx = d.n[0], ny = d.n[1];
    const FV A = a.view(), B = b.view();
    par_for(N, [=] HD(long q) {
        const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
        A(i, j, k) = s * v[q]; B(i, j, k) = s * v[N + q];
    });
}
double dot(Comm& comm, long n, const double* a, const double* b) {
    double s = par_sum(n, [=] HD(long q) { return a[q] * b[q]; });
    comm.allreduce_sum(&s, 1);
    return s;
}
void axpy(long n, double a, const double* x, double* y) { par_for(n, [=] HD(long q) { y[q] += a * x[q]; }); }
void lin(long n, double a, const double* x, double b, const double* y, double* o) {   // o = a x + b y
    par_for(n, [=] HD(long q) { o[q] = a * x[q] + b * y[q]; });
}
// the local diagonal of the Ex equation (RZ: Ez) at the positions of component 0
void diag_kernel(const Domain& d, double D, const FV W, const FV O, const FV N, double* a) {
    const long Nn = d.owned();
    const int nx = d.n[0], ny = d.n[1];
    par_for(Nn, [=] HD(long q) {
        const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
        a[q] = 1.0 / (1 + D * D * W(i, j, k) / (2 * (2 + D * N(i, j, k) + 0.5 * D * D * O(i, j, k))));
        a[Nn + q] = 1.0;
    });
}
void scale_vec(long n, const double* a, const double* x, double* y) { par_for(n, [=] HD(long q) { y[q] = a[q] * x[q]; }); }
}  // namespace

void medium_solve(Medium& M, const Domain& d, Comm& comm, const MediumOps& ops, const Field& X0, const Field& Y0, double t) {
    const long N2 = 2 * d.owned();
    const int m = std::max(1, M.restart);
    if (M.x.size() != size_t(N2)) {
        M.x.resize(size_t(N2)); M.w.resize(size_t(N2)); M.z.resize(size_t(N2)); M.adiag.resize(size_t(N2));
        M.have_last = false;
    }
    if (int(M.V.size()) != m + 2) {
        M.V.clear();
        for (int q = 0; q < m + 2; ++q) M.V.emplace_back(size_t(N2));   // V[0 .. m] the basis, V[m + 1] G(0)
    }
    const double D = d.dt();
    diag_kernel(d, D, M.wp2[0].view(), M.w02[0].view(), M.nu[0].view(), M.adiag.data());
    double* x = M.x.data();
    double* w = M.w.data();
    double* z = M.z.data();
    double* g0 = M.V[size_t(m + 1)].data();
    const double* ad = M.adiag.data();
    // G(0) and its norm (the scale of the right-hand side)
    M.sx.fill(0); M.sy.fill(0);
    ops.apply(M.sx, M.sy, M.gx, M.gy);
    pack2(d, M.gx, M.gy, g0);
    double bn = std::sqrt(dot(comm, N2, g0, g0));
    // the scale of the right-hand side, at least 1e-4 of the field state: with Ex, Bx ~ 0 (s-pol along x) |G(0)| is round-off of the
    // transport and the relative residual would stall at eps |F| / |G(0)|
    double scale = std::max(bn, 1e-4 * M.fnorm);
    if (!(scale > 0)) scale = 1.0;
    if (!(M.have_last && M.last_t == t)) pack2(d, X0, Y0, x);   // the start: the old fields (or the last solution of this step)
    auto prec = [&](const double* v, double* o) {                // o = M^-1 v: the Ex part scaled by 1/diag, the vacuum solves
        scale_vec(N2, ad, v, o);
        unpack2(d, o, 1.0, M.gx, M.gy);
        ops.precond(M.gx, 0);
        ops.precond(M.gy, 1);
        pack2(d, M.gx, M.gy, o);
    };
    auto matvec = [&](const double* v, double* o) {              // o = A v = v - (G(s v) - G(0)) / s
        unpack2(d, v, scale, M.sx, M.sy);
        ops.apply(M.sx, M.sy, M.gx, M.gy);
        pack2(d, M.gx, M.gy, o);
        lin(N2, 1.0 / scale, g0, -1.0 / scale, o, o);             // (G(0) - G(s v)) / s
        axpy(N2, 1.0, v, o);
    };
    std::vector<double> H(static_cast<size_t>((m + 1) * m)), cs(static_cast<size_t>(m)), sn(static_cast<size_t>(m)),
        g(static_cast<size_t>(m + 1)), y(static_cast<size_t>(m));
    int its = 0;
    double res = 0;
    for (;;) {
        // r = G(x) - x
        unpack2(d, x, 1.0, M.sx, M.sy);
        ops.apply(M.sx, M.sy, M.gx, M.gy);
        double* v0 = M.V[0].data();
        pack2(d, M.gx, M.gy, v0);
        axpy(N2, -1.0, x, v0);
        const double beta = std::sqrt(dot(comm, N2, v0, v0));
        res = beta / scale;
        if (vlpl_getenv("VLPL_MEDIUM_DEBUG") && comm.rank == 0) std::fprintf(stderr, "medium solve it %d res %.3e\n", its, res);
        if (res < M.tol || its >= M.maxit || beta == 0) break;
        lin(N2, 1.0 / beta, v0, 0.0, v0, v0);
        std::fill(g.begin(), g.end(), 0.0);
        g[0] = beta;
        int kk = 0;
        for (int j = 0; j < m && its < M.maxit; ++j) {
            prec(M.V[size_t(j)].data(), z);
            double* vj1 = M.V[size_t(j + 1)].data();
            matvec(z, vj1);
            for (int i = 0; i <= j; ++i) {                       // modified Gram-Schmidt
                const double h = dot(comm, N2, vj1, M.V[size_t(i)].data());
                H[size_t(i * m + j)] = h;
                axpy(N2, -h, M.V[size_t(i)].data(), vj1);
            }
            const double hn = std::sqrt(dot(comm, N2, vj1, vj1));
            H[size_t((j + 1) * m + j)] = hn;
            if (hn > 0) lin(N2, 1.0 / hn, vj1, 0.0, vj1, vj1);
            for (int i = 0; i < j; ++i) {                        // the previous rotations
                const double a = H[size_t(i * m + j)], b = H[size_t((i + 1) * m + j)];
                H[size_t(i * m + j)] = cs[size_t(i)] * a + sn[size_t(i)] * b;
                H[size_t((i + 1) * m + j)] = -sn[size_t(i)] * a + cs[size_t(i)] * b;
            }
            const double a = H[size_t(j * m + j)], b = H[size_t((j + 1) * m + j)], rr = std::hypot(a, b);
            cs[size_t(j)] = rr > 0 ? a / rr : 1.0; sn[size_t(j)] = rr > 0 ? b / rr : 0.0;
            H[size_t(j * m + j)] = rr; H[size_t((j + 1) * m + j)] = 0;
            g[size_t(j + 1)] = -sn[size_t(j)] * g[size_t(j)];
            g[size_t(j)] = cs[size_t(j)] * g[size_t(j)];
            ++its; kk = j + 1;
            if (std::fabs(g[size_t(j + 1)]) / scale < 0.5 * M.tol || hn == 0) break;
        }
        for (int i = kk - 1; i >= 0; --i) {                      // back substitution
            double s = g[size_t(i)];
            for (int q = i + 1; q < kk; ++q) s -= H[size_t(i * m + q)] * y[size_t(q)];
            y[size_t(i)] = H[size_t(i * m + i)] != 0 ? s / H[size_t(i * m + i)] : 0.0;
        }
        M.w.zero();
        for (int i = 0; i < kk; ++i) axpy(N2, y[size_t(i)], M.V[size_t(i)].data(), w);
        prec(w, z);
        axpy(N2, 1.0, z, x);
    }
    unpack2(d, x, 1.0, M.sx, M.sy);
    M.have_last = true;
    M.last_t = t;
    MediumStats& st = M.stats;
    ++st.solves; st.iterations += its;
    st.max_iterations = std::max<long>(st.max_iterations, its);
    if (res >= M.tol) ++st.unconverged;
    st.last_residual = res;
}

// ---------------------------------------------------------------- model = bound (bloch.hpp, decision 85)
namespace {
struct BState {                                       // the views of the density matrix
    FV s[BS_N];
};
BState bstate(const Medium& M) {
    BState b;
    for (int q = 0; q < BS_N; ++q) b.s[q] = M.bs[q].view();
    return b;
}
// u* = n_a (P(rho(Et*)) - P(rho^n)) / dt and its tangent by central differences (the node step is smooth in E; the tangent only
// steers the iteration, the converged Et does not depend on it)
void bound_linearize(Medium& M, const Domain& d) {
    const BlochC L = M.bc;
    const BState b = bstate(M);
    const FV N = M.bn.view(), Es = M.bEs.view(), U0 = M.bu0.view(), SG = M.bsg.view();
    const double D = d.dt();
    for_owned(d, [=] HD(int i, int j, int k) {
        const double n = N(i, j, k);
        if (n == 0.0) return;
        double s[BS_N], o[BS_N];
        for (int q = 0; q < BS_N; ++q) s[q] = b.s[q](i, j, k);
        const double p0 = bloch_P(L, s), e = Es(i, j, k), de = 1e-6 * (fabs(e) + 1e-3);
        bloch_step(L, s, e, o);
        U0(i, j, k) = n * (bloch_P(L, o) - p0) / D;
        bloch_step(L, s, e + de, o);
        const double up = bloch_P(L, o);
        bloch_step(L, s, e - de, o);
        SG(i, j, k) = n * (up - bloch_P(L, o)) / (2 * de * D);
    });
}
}  // namespace

void medium_bound_solve(Medium& M, const Domain& d, Comm& comm, double t, const std::function<void()>& solve,
                        const std::function<void()>& prepare) {
    if (M.bt != t) { M.bEs.swap(M.bE); M.bE.fill(0); M.bt = t; }    // a new step: linearize at the last step's Et
    const int nx = d.n[0], ny = d.n[1];
    const FV E = M.bE.view(), Es = M.bEs.view(), N = M.bn.view();
    int it = 0;
    bool conv = false;
    for (; it < M.bmaxit; ++it) {
        bound_linearize(M, d);
        solve();
        prepare();                                    // bE: the Et of the solution with this linearization
        double v[2] = {
            par_max_abs(d.owned(), [=] HD(long q) {
                const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
                return N(i, j, k) == 0.0 ? 0.0 : E(i, j, k) - Es(i, j, k);
            }),
            par_max_abs(d.owned(), [=] HD(long q) {
                const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
                return N(i, j, k) == 0.0 ? 0.0 : E(i, j, k);
            })};
        v[0] = comm.allreduce_max(v[0]); v[1] = comm.allreduce_max(v[1]);
        if (vlpl_getenv("VLPL_MEDIUM_DEBUG") && comm.rank == 0) std::fprintf(stderr, "bound newton %d: |dEt| %.3e |Et| %.3e\n", it, v[0], v[1]);
        if (v[0] <= M.btol * std::max(v[1], 1e-300) || v[1] == 0) { conv = true; ++it; break; }
        M.bEs.swap(M.bE);                             // linearize at the new Et (bE is recomputed by the next prepare)
    }
    MediumStats& st = M.stats;
    ++st.newton_steps; st.newton += it;
    st.newton_max = std::max<long>(st.newton_max, it);
    if (!conv) ++st.newton_unconverged;
}

void medium_bound_commit(Medium& M, const Domain& d, Comm& comm) {
    const BlochC L = M.bc;
    const BState b = bstate(M);
    const FV N = M.bn.view(), E = M.bE.view(), U = M.bU.view(), P = M.bP.view();
    const FV A = L.ion ? M.bacc.view() : N, Dn = M.bu0.view();     // ionization: the accumulated density; its change (scratch)
    const double D = d.dt(), dv = d.cell_volume();
    const int nx = d.n[0], ny = d.n[1];
    // with relaxation or ionization the field's work dt Et u minus the change of the atoms' energy is dissipated (exact bookkeeping)
    double diss = par_sum(d.owned(), [=] HD(long q) {
        const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
        const double n = N(i, j, k);
        if (n == 0.0) return 0.0;
        double s[BS_N];
        for (int r = 0; r < BS_N; ++r) s[r] = b.s[r](i, j, k);
        const double k0 = bloch_energy(L, s), tr0 = s[BS_P0] + s[BS_P1] + s[BS_P2];
        bloch_step(L, s, E(i, j, k), s);
        if (L.ion) {                                  // the atoms ionized in this step (the trace is the neutral fraction)
            const double dn = n * (tr0 - s[BS_P0] - s[BS_P1] - s[BS_P2]);
            A(i, j, k) += dn;
            Dn(i, j, k) = dn;
        }
        for (int r = 0; r < BS_N; ++r) b.s[r](i, j, k) = s[r];
        P(i, j, k) += D * U(i, j, k);                 // the polarization the fields saw (Gauss law); = n_a P(rho) to the tolerance
        return L.loss || L.ion ? (D * E(i, j, k) * U(i, j, k) - n * (bloch_energy(L, s) - k0)) * dv : 0.0;
    });
    if (L.loss || L.ion) { comm.allreduce_sum(&diss, 1); M.dissipated += diss; }
    if (L.ion) {
        double ion = par_sum(d.owned(), [=] HD(long q) {
            const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
            return N(i, j, k) == 0.0 ? 0.0 : Dn(i, j, k) * dv;
        });
        comm.allreduce_sum(&ion, 1);
        M.ionized += ion;
    }
}
