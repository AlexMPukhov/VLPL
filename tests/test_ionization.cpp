// Field ionization (decision 86, docs/specs/field_ionization.md, src/ionization.hpp): rates, the Monte-Carlo scheme against the
// rate equations, the charge-state distribution after a pulse against tools/ionization_ref.py, the energy balance of the
// ionization current, the Gauss law with electrons created, the charge per particle in the push, deck checks.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>
#include "common.hpp"
#include "deck.hpp"
#include "atomic.hpp"
#include "ionization.hpp"
#include "sim.hpp"

namespace {
const double LAM = 0.8e-6, DX = 2 * M_PI / 40;          // the test grids: dx = lambda0 / 40, dt = dx
Config deck_config(const std::string& text, const char* path) {
    if (FILE* f = std::fopen(path, "w")) { std::fputs(text.c_str(), f); std::fclose(f); }
    Deck dk = Deck::read(path);
    return Config::from_deck(dk);
}
// a periodic 8 x 4 x 4 box, dx = lambda0 / 40, dy = dz = 2 dx; species: the ion block (ion lines) and immobile or mobile electrons
std::string box_deck(const std::string& ion_lines, bool mobile_e, const std::string& numerics = "") {
    return "grid {\n lambda0 = 0.8 um\n nx = 8\n ny = 4\n nz = 4\n dx = 0.02 um\n dy = 0.04 um\n dz = 0.04 um\n t_end = 1000 fs\n}\n"
           "numerics {\n seed = 7\n" + numerics + "}\n"
           "species ion {\n" + ion_lines + " ionization_electrons = e\n}\n"
           "species e {\n charge = -1\n mass = 1\n density = 0\n" + std::string(mobile_e ? "" : " mobile = false\n") + "}\n"
           "output {\n dir = ion_tmp_out\n}\n";
}
int idx(Simulation& s, const char* name) {             // species index by name (the deck orders the blocks by name)
    for (size_t i = 0; i < s.cfg.species.size(); ++i)
        if (s.cfg.species[i].name == name) return int(i);
    return -1;
}
std::vector<double> states(Simulation& s, int sp, int zmax, Comm& comm) {
    Species& S = s.species[size_t(sp)];
    const std::vector<int> z = S.get_as<int>(S.zslot);
    std::vector<double> c(size_t(zmax) + 1, 0.0);
    for (int v : z) c[size_t(v)] += 1;
    comm.allreduce_sum(c.data(), zmax + 1);
    return c;
}
// the rate equations dN_k/dt = w_{k-1} N_{k-1} - w_k N_k (RK4, 2000 steps per unit of w_max T)
std::vector<double> chain(std::vector<double> N, const std::vector<double>& w, double T) {
    double wm = 0;
    for (double x : w) wm = std::max(wm, x);
    const int n = std::max(2000, int(2000 * wm * T));
    const double h = T / n;
    auto f = [&](const std::vector<double>& x) {
        std::vector<double> d(x.size(), 0.0);
        for (size_t k = 0; k + 1 < x.size(); ++k) { const double r = w[k] * x[k]; d[k] -= r; d[k + 1] += r; }
        return d;
    };
    for (int i = 0; i < n; ++i) {
        std::vector<double> a = f(N), t(N.size());
        for (size_t k = 0; k < N.size(); ++k) t[k] = N[k] + 0.5 * h * a[k];
        std::vector<double> b = f(t);
        for (size_t k = 0; k < N.size(); ++k) t[k] = N[k] + 0.5 * h * b[k];
        std::vector<double> c = f(t);
        for (size_t k = 0; k < N.size(); ++k) t[k] = N[k] + h * c[k];
        std::vector<double> d = f(t);
        for (size_t k = 0; k < N.size(); ++k) N[k] += h / 6 * (a[k] + 2 * b[k] + 2 * c[k] + d[k]);
    }
    return N;
}
// the field at which level (ip, zafter, l) has the rate w (code units), by bisection
double field_for_rate(double ip, int zafter, int l, double w) {
    double lo = 1e-4, hi = 1;
    const IonLevel L = ion_level(ip, zafter, l, 0, false, 6, LAM, DX);
    for (int it = 0; it < 200; ++it) {
        const double mid = std::sqrt(lo * hi);
        (ion_rate(L, mid) < w ? lo : hi) = mid;
    }
    return std::sqrt(lo * hi);
}
double field_energy(Simulation& s, Comm& comm) {
    double v = s.rz ? s.rz->energy() : s.fields->energy();
    comm.allreduce_sum(&v, 1);
    return v;
}
double energy(Simulation& s, Comm& comm) {
    double v[2] = {s.rz ? s.rz->energy() : s.fields->energy(), 0};
    for (const Species& sp : s.species) v[1] += sp.kinetic_energy();
    comm.allreduce_sum(v, 2);
    return v[0] + v[1];
}
}  // namespace

// I1. Rates: (a) hydrogen-like I = 1/2 Hartree, l = 0: the Landau rate 4/F exp(-2/(3F)) [a.u.] exactly; (b) the code-unit tables
//     (IonLevel) equal the SI formula; (c) against the independent Python implementation (tools/ionization_ref.py rates, l from
//     the configuration, l* = n* - 1); (d) beyond the peak of the expression the rate stays at its maximum.
static void ion_rates() {
    const double Wa = 4.1341373335e16, Ea = 5.14220674763e11, IH = 13.605693122994;
    double e1 = 0;
    for (double F : {0.02, 0.05, 0.1, 0.2}) {
        const double landau = Wa * 4 / F * std::exp(-2 / (3 * F));
        e1 = std::max(e1, std::abs(ion_rate_si(IH, 1, 0, 0, false, 6, F * Ea) / landau - 1));
    }
    check("ionization I1a: ADK of I = 1/2 Hartree, l = 0 = the Landau rate 4/F exp(-2/(3F))", e1 < 1e-13, fmt("max rel. error %.1e", e1));
    const double E0 = 2 * M_PI * 0.51099895e6 / LAM, w0 = 2 * M_PI * 299792458.0 / LAM;
    double e2 = 0;
    struct Lv { double ip; int z, l; bool tl; };
    for (const Lv& v : {Lv{13.598434599702, 1, 0, false}, Lv{15.7596119, 1, 1, false}, Lv{143.4567, 8, 1, true}, Lv{552.06741, 6, 0, false},
                        Lv{2311.0, 30, 2, true}}) {
        const IonLevel L = ion_level(v.ip, v.z, v.l, 0, v.tl, 6, LAM, DX);
        for (double E : {0.003, 0.01, 0.03, 0.1, 0.3, 1.0}) {
            const double a = ion_rate(L, E), b = ion_rate_si(v.ip, v.z, v.l, 0, v.tl, 6, std::min(E, L.epeak) * E0) / w0;
            if (b > 1e-200 && E > L.emin) e2 = std::max(e2, std::abs(a / b - 1));
        }
    }
    check("ionization I1b: code-unit level tables = the SI formula (5 levels, 6 fields, with and without Tong-Lin)", e2 < 1e-11,
          fmt("max rel. error %.1e", e2));
    // python3 tools/ionization_ref.py rates
    struct Ref { double ip; int z, l; double E, w, wtl; };
    const Ref ref[] = {{13.598434599702, 1, 0, 1e10, 1.1317085433e+04, 8.9588259827e+03},
                       {13.598434599702, 1, 0, 5e10, 1.8006623383e+15, 5.5977669493e+14},
                       {24.587389011, 1, 0, 1e11, 1.8458960595e+14, 1.0848098821e+14},
                       {15.7596119, 1, 1, 5e10, 9.2903284822e+14, 4.1409340259e+14},
                       {143.4567, 8, 0, 1e12, 1.1863409873e+18, 1.8942038841e+16},
                       {552.06741, 6, 0, 5e12, 2.3174780217e+12, 1.5526423674e+12}};
    double e3 = 0;
    for (const Ref& r : ref) {
        e3 = std::max(e3, std::abs(ion_rate_si(r.ip, r.z, r.l, 0, false, 6, r.E) / r.w - 1));
        e3 = std::max(e3, std::abs(ion_rate_si(r.ip, r.z, r.l, 0, true, 6, r.E) / r.wtl - 1));
    }
    check("ionization I1c: rates = the independent Python implementation (H, He, Ar0 (l = 1), Ar7+, N5+, plain and Tong-Lin)", e3 < 1e-9,
          fmt("max rel. error %.1e", e3));
    const IonLevel L = ion_level(13.598434599702, 1, 0, 0, false, 6, LAM, DX), T = ion_level(13.598434599702, 1, 0, 0, true, 6, LAM, DX);
    const bool cap = ion_rate(L, 2 * L.epeak) == ion_rate(L, L.epeak) && ion_rate(L, 0.99 * L.epeak) < ion_rate(L, L.epeak) &&
                     ion_rate(T, 3 * T.epeak) == ion_rate(T, T.epeak) && T.epeak < L.epeak;
    check("ionization I1d: the rate stays at its maximum beyond the peak of the expression (ADK, Tong-Lin)", cap,
          fmt("peak ADK %.3f", L.epeak) + fmt(", Tong-Lin %.3f (m c omega0 / e)", T.epeak));
}

// I2. The Monte-Carlo scheme in a constant uniform field (E_y, periodic box, ions and electrons at rest, no ionization current):
//     (a) one level (H, w dt = 0.02): the neutral fraction against exp(-w t); (b) four levels of Ar with w_1 dt = 0.5 (Ar0 -> Ar+
//     faster still, several levels per step) against the rate equations (RK4); (c) the result does not depend on the order of
//     the particles (the random numbers are keyed by the particle id: any number of threads, ranks or a GPU give the same).
static void ion_constant_field(Comm& comm) {
    {
        const double ip = 13.598434599702, w = 0.02 / DX, E = field_for_rate(ip, 1, 0, w);
        Config c = deck_config(box_deck(" element = H\n ionization = adk\n ionization_current = false\n mobile = false\n density = 1e-3\n"
                                        " ppc_x = 4\n ppc_y = 8\n ppc_z = 8\n", false), "ion_tmp.deck");
        Simulation s(c, comm);
        s.init();
        const double n = states(s, idx(s, "ion"), 1, comm)[0];
        double worst = 0;
        std::string info;
        for (int step = 1; step <= 100; ++step) {
            s.fields->ey.fill(E);
            s.step();
            if (step == 25 || step == 50 || step == 100) {
                const double f = states(s, idx(s, "ion"), 1, comm)[0] / n, p = std::exp(-w * step * DX), sig = std::sqrt(p * (1 - p) / n);
                worst = std::max(worst, std::abs(f - p) / sig);
                info += fmt(" t = %.0f dt: %.4f", step, f) + fmt(" (exp(-w t) %.4f);", p);
            }
        }
        check("ionization I2a: H in a constant field (w dt = 0.02, 32768 ions): neutral fraction = exp(-w t) within 4 sigma", worst < 4,
              fmt("max deviation %.2f sigma;", worst) + info);
    }
    double counts[2][5] = {};
    for (int perm = 0; perm < 2; ++perm) {
        const std::vector<double> ip = {15.7596119, 27.62967, 40.735, 59.58};
        const double E = field_for_rate(ip[1], 2, 1, 0.5 / DX);
        std::vector<double> w(5, 0.0);
        for (int k = 0; k < 4; ++k) w[size_t(k)] = ion_rate(ion_level(ip[size_t(k)], k + 1, 1, 0, false, 6, LAM, DX), E);
        Config c = deck_config(box_deck(" element = Ar\n ionization = adk\n max_charge_state = 4\n ionization_current = false\n mobile = false\n"
                                        " density = 1e-3\n ppc_x = 4\n ppc_y = 8\n ppc_z = 8\n", false), "ion_tmp.deck");
        Simulation s(c, comm);
        s.init();
        if (perm) {                                       // reversed order of the ions
            Species& S = s.species[size_t(idx(s, "ion"))];
            std::vector<int> p(S.size());
            std::iota(p.rbegin(), p.rend(), 0);
            S.permute(p);
        }
        const double n = states(s, idx(s, "ion"), 4, comm)[0];
        double worst = 0;
        std::string info;
        int done = 0;
        for (int step : {1, 3, 10}) {
            for (; done < step; ++done) { s.fields->ey.fill(E); s.step(); }
            const std::vector<double> st = states(s, idx(s, "ion"), 4, comm);
            const std::vector<double> ref = chain({1, 0, 0, 0, 0}, w, step * DX);
            for (int k = 0; k <= 4; ++k) {
                const double p = ref[size_t(k)], sig = std::sqrt(std::max(p * (1 - p), 1e-12) / n);
                worst = std::max(worst, std::abs(st[size_t(k)] / n - p) / sig);
            }
            info += fmt(" %.0f dt:", step);
            for (int k = 0; k <= 4; ++k) info += fmt(" %.4f", st[size_t(k)] / n) + fmt("/%.4f", ref[size_t(k)]);
            info += ";";
            if (step == 10) for (int k = 0; k <= 4; ++k) counts[perm][k] = st[size_t(k)];
        }
        if (!perm)
            check("ionization I2b: Ar0..Ar4+ in a constant field (w dt = " + fmt("%.3g", w[0] * DX) + fmt(", %.2f", w[1] * DX) +
                      fmt(", %.3f", w[2] * DX) + fmt(", %.2e", w[3] * DX) + ") = the rate equations within 4 sigma",
                  worst < 4, fmt("max deviation %.2f sigma; MC/reference", worst) + info);
    }
    bool same = true;
    for (int k = 0; k <= 4; ++k) same = same && counts[0][k] == counts[1][k];
    check("ionization I2c: the same ionizations with the ions in reversed order (random numbers keyed by the particle id)", same,
          fmt("Ar counts after 10 steps: %.0f", counts[0][0]) + fmt(" %.0f", counts[0][1]) + fmt(" %.0f", counts[0][2]) +
              fmt(" %.0f", counts[0][3]) + fmt(" %.0f", counts[0][4]));
}

// I3. The charge-state distribution of carbon after a Gaussian pulse E_y = E0 exp(-((t - t0)/tau)^2) cos(t - t0), E0 = 0.055,
//     tau = 4 cycles, t0 = 3 tau (the field imposed uniformly every step), against the rate equations integrated in Python
//     (tools/ionization_ref.py pulse: continuous time; the field held over each step differs by `disc` below).
static void ion_pulse(Comm& comm) {
    const double ref[5] = {0, 0, 0, 0.560844, 0.439155};       // python3 tools/ionization_ref.py pulse (continuous, n = 400000)
    const double disc[5] = {0, 0, 0, 0.560852, 0.439148};      // ... pulse --discrete dt (the field of the MC scheme)
    Config c = deck_config(box_deck(" element = C\n ionization = adk\n max_charge_state = 4\n ionization_current = false\n mobile = false\n"
                                    " density = 1e-3\n ppc_x = 8\n ppc_y = 8\n ppc_z = 8\n", false), "ion_tmp.deck");
    Simulation s(c, comm);
    s.init();
    const double tau = 4 * 2 * M_PI, t0 = 3 * tau, E0 = 0.055;
    const int steps = int(std::lround(6 * tau / DX));
    const double n = states(s, idx(s, "ion"), 4, comm)[0];
    for (int i = 0; i < steps; ++i) {
        const double t = i * DX;
        s.fields->ey.fill(E0 * std::exp(-std::pow((t - t0) / tau, 2)) * std::cos(t - t0));
        s.step();
    }
    const std::vector<double> st = states(s, idx(s, "ion"), 4, comm);
    double worst = 0;
    std::string info;
    for (int k = 0; k <= 4; ++k) {
        const double p = ref[k], sig = std::sqrt(std::max(p * (1 - p), 1e-12) / n) + std::abs(ref[k] - disc[k]);
        worst = std::max(worst, std::abs(st[size_t(k)] / n - p) / sig);
        info += fmt(" %.4f", st[size_t(k)] / n) + fmt("/%.4f", p);
    }
    check("ionization I3: C0 -> C4+ after a Gaussian pulse (E0 = 0.055, tau = 4 cycles, 65536 ions) = the rate equations (Python) "
          "within 4 sigma", worst < 4, fmt("max deviation %.2f sigma; fractions C0..C4+ MC/reference", worst) + info);
}

// I4. The energy balance of the ionization current: H in a constant field E_y (periodic box, density 0.05 n_c, 30 steps, w dt =
//     0.05). (a) Ions and electrons at rest: the field energy lost = the work of the displacement currents (to the time centring
//     of E) = sum I_k w; (b) mobile electrons, the run with the current against the one without (the same random draws): the
//     current takes sum I_k w from W_field + W_kin. (The balance of one run alone is dominated by the half-step offset of
//     W_kin in diag: the electrons gain ~60 I_k in the static field.)
static void ion_energy(Comm& comm) {
    const double ip = 13.598434599702, E = field_for_rate(ip, 1, 0, 0.05 / DX);
    double rel[3] = {0, 0, 0}, wend[3] = {0, 0, 0}, eion[3] = {0, 0, 0};
    for (int v = 0; v < 3; ++v) {
        Config c = deck_config(box_deck(std::string(" element = H\n ionization = adk\n mobile = false\n density = 0.05\n") +
                                            (v == 2 ? " ionization_current = false\n" : "") + " ppc_x = 2\n ppc_y = 2\n ppc_z = 2\n",
                                        v > 0), "ion_tmp.deck");
        Simulation s(c, comm);
        s.init();
        s.fields->ey.fill(E);
        const double w0 = energy(s, comm), wf0 = field_energy(s, comm);
        for (int i = 0; i < 30; ++i) s.step();
        const IonizationTotals T = ionization_totals(s);
        const double dw = energy(s, comm) - w0;
        rel[v] = T.e_ion / wf0;
        wend[v] = dw;
        eion[v] = T.e_ion;
        if (v == 0)
            check("ionization I4a: ions and electrons at rest: field energy lost = work of the ionization current = sum I_k w",
                  std::abs(-dw / T.e_cur - 1) < 2e-3 && std::abs(T.e_cur / T.e_ion - 1) < 1e-12 && T.capped == 0,
                  fmt("-dW / E_cur - 1 = %.2e", -dw / T.e_cur - 1) + fmt(", E_cur / E_ion - 1 = %.1e", T.e_cur / T.e_ion - 1) +
                      fmt(", E_ion / W_field = %.3f", rel[0]) + fmt(", %.0f events", double(T.events)));
    }
    // the run without the current scaled to the same number of ionizations (the current lowers E: 2 % fewer events)
    const double d = (wend[1] - wend[2] * eion[1] / eion[2]) / eion[1];
    check("ionization I4b: mobile electrons: the current takes sum I_k w from W_field + W_kin (run with - run without the current)",
          std::abs(d + 1) < 1e-2, fmt("[d(W + K)_with - d(W + K)_without E_ion,with / E_ion,without] / E_ion,with = %.4f", d) + fmt(", E_ion with / without - 1 = %.1e",
          eion[1] / eion[2] - 1) + fmt(", E_ion / W_field = %.3f", rel[1]) + fmt(", d(W + K) / E_ion without the current = %.2f", wend[2] / eion[2]));
}

// I5. The Gauss law at round-off while electrons are created (charge-neutral at the ion, then displaced by the ionization
//     current), mobile H ions (charge per particle in push and deposit) and electrons, shape 2: Cart3D, 2D and RZ (modes 0, 1,
//     a uniform E_z (m = 0) and E_x (m = 1) at the start).
static void ion_gauss(Comm& comm) {
    for (int g = 0; g < 3; ++g) {
        std::string t;
        const char* ion = " element = H\n ionization = adk\n density = 0.05\n";
        if (g < 2)
            t = std::string("grid {\n lambda0 = 0.8 um\n") + (g == 1 ? " geometry = cartesian2d\n" : " nz = 6\n dz = 0.04 um\n") +
                " nx = 16\n ny = 8\n dx = 0.02 um\n dy = 0.04 um\n t_end = 1000 fs\n}\n"
                "numerics {\n shape = 2\n}\n"
                "species ion {\n" + ion + " ppc_x = 2\n ppc_y = 2\n" + (g == 0 ? " ppc_z = 2\n" : "") +
                " xmin = 0.06 um\n xmax = 0.26 um\n ionization_electrons = e\n}\n"
                "species e {\n charge = -1\n mass = 1\n density = 0\n}\noutput {\n dir = ion_tmp_out\n}\n";
        else
            t = std::string("grid {\n geometry = rz\n modes = 2\n lambda0 = 0.8 um\n nz = 16\n nr = 10\n dz = 0.02 um\n dr = 0.04 um\n t_end = 1000 fs\n}\n") +
                "numerics {\n shape = 2\n}\n"
                "species ion {\n" + ion + " ppc_r = 2\n ppc_theta = 8\n zmin = 0.06 um\n zmax = 0.26 um\n rmax = 0.25 um\n ionization_electrons = e\n}\n"
                "species e {\n charge = -1\n mass = 1\n density = 0\n ppc_theta = 8\n}\noutput {\n dir = ion_tmp_out\n}\n";
        Config c = deck_config(t, "ion_tmp.deck");
        Simulation s(c, comm);
        s.init();
        const double E = 0.02;
        if (g < 2) { s.fields->ey.fill(E); s.fields->ex.fill(0.5 * E); }
        else {
            H ez(s.rz->ez), er(s.rz->er), ep(s.rz->ep);
            const Domain& d = s.dom;
            for (int i = -d.ng; i < d.n[0] + d.ng; ++i)
                for (int j = 0; j < d.n[1]; ++j) { ez(i, j, 0) = 0.5 * E; er(i, j, 1) = 0.5 * E; ep(i, j, 2) = 0.5 * E; }
            ez.put(s.rz->ez); er.put(s.rz->er); ep.put(s.rz->ep);
        }
        const double w0 = energy(s, comm);
        double gd = 0;
        for (int i = 0; i < 20; ++i) { s.step(); gd = std::max(gd, s.gauss_drift()); }
        const IonizationTotals T = ionization_totals(s);
        const std::vector<double> st = states(s, idx(s, "ion"), 1, comm);
        const double frac = st[1] / std::max(1.0, st[0] + st[1]);
        const char* nm[3] = {"Cart3D", "2D", "RZ (m = 0, 1)"};
        check(std::string("ionization I5: Gauss law at round-off with electrons created and displaced, ") + nm[g], gd < 1e-12 && frac > 0.3,
              fmt("Gauss %.1e", gd) + fmt(", ionized fraction %.3f", frac) + fmt(", %.0f events", double(T.events)) +
                  fmt(", (W + K + E_cur)(20 dt) / W(0) - 1 = %.1e", (energy(s, comm) + T.e_cur) / w0 - 1));
    }
}

// I6. The charge per particle in the push: C4+ (no ionization in E = 0.01: the next level needs 392 eV) mobile in a uniform E_y,
//     10 steps: u_y = 4 E t / m exactly (the push of a pure E field).
static void ion_push(Comm& comm) {
    Config c = deck_config(box_deck(" element = C\n charge_state = 4\n ionization = adk\n density = 1e-3\n", false), "ion_tmp.deck");
    Simulation s(c, comm);
    s.init();
    const int ii = idx(s, "ion");
    const double E = 0.01, m = s.species[size_t(ii)].m;
    for (int i = 0; i < 10; ++i) { s.fields->ey.fill(E); s.fields->bx.fill(0); s.fields->by.fill(0); s.fields->bz.fill(0); s.step(); }
    const std::vector<double> uy = s.species[size_t(ii)].get(Species::UY);
    double err = 0;
    for (double u : uy) err = std::max(err, std::abs(u / (4 * E * 10 * DX / m) - 1));
    err = comm.allreduce_max(err);
    check("ionization I6: C4+ in a uniform E: u_y = Z E t / m (charge state per particle in the push)", err < 1e-12 && m > 21800 && m < 21900,
          fmt("max rel. error %.1e", err) + fmt(", m = %.1f m_e", m));
}

// I7. Deck checks: unknown element, max_charge_state above Z, charge with ionization, ionization keys without ionization, the
//     implicit mover, the electron species missing or the ion species itself, an unknown BSI model.
static void ion_deck(Comm& comm) {
    const std::string base = box_deck(" element = N\n ionization = adk\n density = 1e-3\n", true);
    deck_config(base, "ion_tmp.deck");
    int rejected = 0;
    for (const char* o : {"species ion.element=Xx", "species ion.max_charge_state=8", "species ion.charge=2", "species ion.ionization=none",
                          "numerics.mover=implicit", "species ion.ionization_electrons=ion", "species ion.ionization_electrons=none",
                          "species ion.ionization_bsi=posthumus"}) {
        Deck x = Deck::read("ion_tmp.deck");
        x.override_from(o);
        try {
            Config c = Config::from_deck(x);
            Simulation s(c, comm);
            s.init();
        } catch (const std::exception&) { ++rejected; }
    }
    Config c = deck_config(base, "ion_tmp.deck");
    size_t ci = 0;
    while (ci < c.species.size() && c.species[ci].name != "ion") ++ci;
    const IonizationConfig& I = c.species[ci].ion;
    const bool ok = I.on && I.zmax == 7 && I.z0 == 0 && std::abs(c.species[ci].m / 1822.888486209 - 14.0069) < 1e-9 && c.species[ci].q == 1 &&
                    I.l[0] == 1 && I.l[5] == 0 && std::abs(I.ip_eV[6] - 667.0461377) < 1e-9;
    check("ionization I7: deck (N: Z = 7, mass, NIST energies, l), 8 invalid combinations rejected", ok && rejected == 8,
          fmt("rejected %.0f of 8", rejected));
    if (comm.rank == 0) std::remove("ion_tmp.deck");
    // the table: H .. Lr complete (and D), energies rising with k, l in 0 .. 3; spot values (NIST ASD 5.12)
    bool tab = elements().size() == 104;
    int z = 0;
    for (const Element& e : elements()) {
        if (e.symbol != "D") tab = tab && e.Z == ++z;
        for (int k = 0; k < e.Z; ++k) tab = tab && e.l[size_t(k)] >= 0 && e.l[size_t(k)] <= 3 && (k == 0 || e.ip_eV[size_t(k)] > e.ip_eV[size_t(k) - 1]);
    }
    const Element *U = find_element("U"), *Lr = find_element("lr"), *Au = find_element("Au");
    tab = tab && U && Lr && Au && Lr->Z == 103 && std::abs(U->mass_u - 238.029) < 1e-3 && std::abs(Au->ip_eV[78] - 93254.62) < 1e-6 &&
          Au->l[0] == 0 && Au->l[1] == 2;
    check("ionization I7b: the NIST table: every charge state of H .. Lr and D, energies rising with the charge state", tab,
          fmt("%.0f elements", double(elements().size())) + fmt(", %.0f levels of Lr", Lr ? double(Lr->ip_eV.size()) : 0.0));
}

void test_ionization(Comm& comm) {
    ion_rates();
    ion_constant_field(comm);
    ion_pulse(comm);
    ion_energy(comm);
    ion_gauss(comm);
    ion_push(comm);
    ion_deck(comm);
}
