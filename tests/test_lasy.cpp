// Laser from LASY envelope files (docs/specs/laser_file.md). The test writes the files in LASY's layout
// (checked against files of LASY 0.7.0) and compares with the analytic pulses.
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include "common.hpp"
#include "sim.hpp"
#ifdef RIP_HDF5
#include "h5.hpp"

namespace {
using cd = std::complex<double>;
const double lambda0 = 0.8e-6, c_si = 299792458.0, k0 = 2 * M_PI / lambda0, w0si = c_si * k0;
const double E0 = 9.1093837015e-31 * c_si * w0si / 1.602176634e-19;   // m c omega0 / e [V/m]

// A LASY file: env in code units (E [m c omega0 / e] or a), geometry "cartesian" (axes t, y, x: LASY's
// x = our y, LASY's y = our z) or "thetaMode" (components, t, r), SI grids.
struct LasyFile {
    std::string path, geometry = "cartesian", field = "electric_field";
    double omega = 1, pxr = 1, pxi = 0, pyr = 0, pyi = 0;
    int nt = 0, n1 = 1, n2 = 1, ncomp = 1;           // cartesian: n1 = ny (LASY y), n2 = nx (LASY x); thetaMode: n2 = nr
    double t0 = 0, dt = 1, a0 = 0, da = 1, b0 = 0, db = 1;   // code units: t; LASY y (a), LASY x or r (b)
    std::function<cd(int comp, double t, double a, double b)> f;
    void write() const {
        h5::Id file = h5::create_file(path);
        for (auto kv : {std::make_pair("openPMD", "1.1.0"), std::make_pair("basePath", "/data/%T/"),
                        std::make_pair("meshesPath", "meshes/"), std::make_pair("iterationEncoding", "fileBased"),
                        std::make_pair("iterationFormat", "lasy_%05T"), std::make_pair("software", "lasy")})
            h5::attr(file, kv.first, kv.second);
        h5::attr(file, "openPMDextension", 0u);
        h5::Id it = h5::group(file, "data/0");
        h5::attr(it, "time", 0.0); h5::attr(it, "dt", 1.0); h5::attr(it, "timeUnitSI", 1.0);
        h5::Id me = h5::group(it, "meshes");
        const bool cart = geometry == "cartesian";
        const std::vector<hsize_t> dims = cart ? std::vector<hsize_t>{hsize_t(nt), hsize_t(n1), hsize_t(n2)}
                                               : std::vector<hsize_t>{hsize_t(ncomp), hsize_t(nt), hsize_t(n2)};
        std::vector<double> v;
        const double unit = field == "electric_field" ? E0 : 1.0;
        if (cart)
            for (int l = 0; l < nt; ++l)
                for (int a = 0; a < n1; ++a)
                    for (int b = 0; b < n2; ++b) {
                        const cd e = unit * f(0, t0 + l * dt, a0 + a * da, b0 + b * db);
                        v.push_back(e.real()); v.push_back(e.imag());
                    }
        else
            for (int c = 0; c < ncomp; ++c)
                for (int l = 0; l < nt; ++l)
                    for (int b = 0; b < n2; ++b) {
                        const cd e = unit * f(c, t0 + l * dt, 0, b0 + b * db);
                        v.push_back(e.real()); v.push_back(e.imag());
                    }
        const hid_t ct = H5Tcreate(H5T_COMPOUND, 16);       // as h5py writes complex128
        H5Tinsert(ct, "r", 0, H5T_NATIVE_DOUBLE);
        H5Tinsert(ct, "i", 8, H5T_NATIVE_DOUBLE);
        const hid_t sp = H5Screate_simple(3, dims.data(), nullptr);
        const hid_t ds = H5Dcreate2(me, "laserEnvelope", ct, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, ct, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
        h5::attr(ds, "geometry", geometry);
        h5::attr(ds, "dataOrder", "C");
        h5::attr(ds, "axisLabels", cart ? std::vector<std::string>{"t", "y", "x"} : std::vector<std::string>{"t", "r"});
        const double T = 1 / w0si, L = 1 / k0;
        h5::attr(ds, "gridSpacing", cart ? std::vector<double>{dt * T, da * L, db * L} : std::vector<double>{dt * T, db * L});
        h5::attr(ds, "gridGlobalOffset", cart ? std::vector<double>{t0 * T, a0 * L, b0 * L} : std::vector<double>{t0 * T, b0 * L});
        h5::attr(ds, "position", cart ? std::vector<double>{0, 0, 0} : std::vector<double>{0, 0});
        h5::attr(ds, "gridUnitSI", 1.0);
        h5::attr(ds, "unitSI", 1.0);
        h5::attr(ds, "angularFrequency", omega * w0si);
        h5::attr(ds, "envelopeField", field);
        if (pxi == 0 && pyi == 0) h5::attr(ds, "polarization", std::vector<double>{pxr, pyr});
        else {                                              // complex attribute (compound r, i)
            const double p[4] = {pxr, pxi, pyr, pyi};
            const hsize_t two = 2;
            const hid_t as = H5Screate_simple(1, &two, nullptr);
            const hid_t a = H5Acreate2(ds, "polarization", ct, as, H5P_DEFAULT, H5P_DEFAULT);
            H5Awrite(a, ct, p);
            H5Aclose(a); H5Sclose(as);
        }
        H5Dclose(ds); H5Sclose(sp); H5Tclose(ct);
    }
};

double gauss_t(double t, double tau) { return std::exp(-1.3862943611198906 * t * t / (tau * tau)); }   // 2 ln 2
}  // namespace

// L4. Cartesian file of the built-in Gaussian at its focus (env = i a0 e^{-r^2/w^2} g(t): E = a0 ... sin),
//     both polarizations, against LaserConfig::field at random points; the interpolation error of the
//     file grid (dt = 0.2, dy = 0.25) is the only difference.
static void lasy_cartesian(Comm&) {
    const double W = 5, tau = 9, delay = 40, A = 1.3;
    LasyFile F;
    F.path = "test_lasy_cart.h5";
    F.nt = 241; F.t0 = -24; F.dt = 0.2;
    F.n1 = F.n2 = 97; F.a0 = F.b0 = -12; F.da = F.db = 0.25;
    F.f = [&](int, double t, double a, double b) { return cd(0, A) * std::exp(-(a * a + b * b) / (W * W)) * gauss_t(t, tau); };
    F.write();
    LaserFileConfig c;
    c.path = F.path; c.delay = delay; c.t_center = 0; c.yc = 3; c.zc = -2;
    FileLaser L(c, lambda0, -20, 20, -20, 20);
    const FileLaserView v = L.host_view();
    LaserConfig B;
    B.enabled = true; B.a0 = A; B.tau = tau; B.tc = delay; B.w0 = W; B.w0z = W; B.yc = 3; B.zc = -2;
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> u(-1, 1);
    double worst = 0, other = 0;
    for (int n = 0; n < 2000; ++n) {
        const double t = delay + 15 * u(rng), y = 3 + 9 * u(rng), z = -2 + 9 * u(rng);
        worst = std::max(worst, std::abs(v.field(0, t, y, z) - B.field(t, y, z)));
        other = std::max(other, std::abs(v.field(1, t, y, z)));
    }
    check("lasy: cartesian file = built-in Gaussian (interpolation of the file grid)", worst < 2e-3 * A && other == 0,
          fmt("max |E_y - E_built-in| / a0 %.1e", worst / A) + fmt(", max |E_z| %.1e", other));
}

// L5. thetaMode file with m = 0 and m = 1: a Laguerre-Gauss (0, 1) pulse, env = i f(r) e^{-i theta}
//     (C_1 = i f, S_1 = f), against the built-in LG mode (0, 1) at its focus; and a cartesian file of a
//     normalized vector potential with circular polarization (1, i)/sqrt 2, against
//     E = Re[p (i omega a - da/dt) e^{-i omega u}] written out.
static void lasy_modes(Comm&) {
    const double W = 5, tau = 9, delay = 40, A = 0.9;
    LasyFile F;
    F.path = "test_lasy_theta.h5";
    F.geometry = "thetaMode"; F.ncomp = 3;
    F.nt = 241; F.t0 = -24; F.dt = 0.2;
    F.n2 = 81; F.b0 = 0; F.db = 0.2;
    auto fr = [&](double t, double r) { return A * std::sqrt(2.0) * r / W * std::exp(-r * r / (W * W)) * gauss_t(t, tau); };
    F.f = [&](int comp, double t, double, double r) { return comp == 1 ? cd(0, fr(t, r)) : (comp == 2 ? cd(fr(t, r), 0) : cd(0)); };
    F.write();
    LaserFileConfig c;
    c.path = F.path; c.delay = delay; c.t_center = 0;
    FileLaser L(c, lambda0, -20, 20, -20, 20);
    LaserConfig B;
    B.enabled = true; B.a0 = A; B.tau = tau; B.tc = delay; B.w0 = W; B.w0z = W; B.l = 1;
    std::mt19937_64 rng(6);
    std::uniform_real_distribution<double> u(-1, 1);
    double lg = 0;
    for (int n = 0; n < 2000; ++n) {
        const double t = delay + 15 * u(rng), y = 9 * u(rng), z = 9 * u(rng);
        lg = std::max(lg, std::abs(L.host_view().field(0, t, y, z) - B.field(t, y, z)));
    }
    // vector potential, circular
    const double s = 1 / std::sqrt(2.0), om = 1.1;
    LasyFile V;
    V.path = "test_lasy_circ.h5";
    V.field = "normalized_vector_potential"; V.omega = om;
    V.pxr = s; V.pyi = s;
    V.nt = 481; V.t0 = -24; V.dt = 0.1;
    V.n1 = V.n2 = 1;                                            // uniform in y and z
    auto a = [&](double t) { return A * gauss_t(t, tau); };
    V.f = [&](int, double t, double, double) { return cd(a(t), 0); };
    V.write();
    c.path = V.path;
    FileLaser C(c, lambda0, -20, 20, -20, 20);
    double circ = 0;
    for (int n = 0; n < 2000; ++n) {
        const double t = delay + 15 * u(rng), uu = t - delay;
        const double dadt = -2 * 1.3862943611198906 * uu / (tau * tau) * a(uu);
        const cd E = cd(0, om) * a(uu) - dadt, ph = std::polar(1.0, -om * uu);
        const double ey = (cd(s, 0) * E * ph).real(), ez = (cd(0, s) * E * ph).real();
        circ = std::max({circ, std::abs(C.host_view().field(0, t, 1, 2) - ey), std::abs(C.host_view().field(1, t, -3, 4) - ez)});
    }
    check("lasy: thetaMode Laguerre-Gauss (0, 1) = built-in; vector potential, circular = E written out",
          lg < 2e-3 * A && circ < 2e-3 * A * om, fmt("LG max error / a0 %.1e", lg / A) + fmt(", circular %.1e", circ / (A * om)));
}

// L6. Exact propagation: a short pulse (tau = 12, ~20 % bandwidth) uniform in z, at its focus (W = 24)
//     in the file at the plane x_f, propagated back to the entrance plane, against the spectrally
//     resolved paraxial beam: every frequency omega of the Gaussian spectrum G(omega) of g(u) e^{-iu}
//     propagated with its own q = -x_f - i z_R(omega), z_R = omega W^2 / 2 (slab: sqrt(q0 / q)
//     e^{i omega y^2 / (2 q)}), summed numerically. Agreement to O(1/(k W)^2) ~ 2e-3.
static void lasy_propagation(Comm&) {
    const double W = 24, tau = 12, delay = 60, A = 1, xf = 0.5 * W * W;
    LasyFile F;
    F.path = "test_lasy_prop.h5";
    F.nt = 161; F.t0 = -32; F.dt = 0.4;
    F.n1 = 1; F.n2 = 129; F.b0 = -96; F.db = 1.5;
    F.f = [&](int, double t, double, double b) { return cd(0, A) * std::exp(-b * b / (W * W)) * gauss_t(t, tau); };
    F.write();
    LaserFileConfig c;
    c.path = F.path; c.delay = delay; c.t_center = 0; c.file_x = xf;
    FileLaser L(c, lambda0, -120, 120, -1, 1);
    // reference: g(u) e^{-iu} = int dw/2pi G(w) e^{-iwu}, G = sqrt(pi/a) e^{-(w-1)^2/(4a)}, a = 2 ln2 / tau^2
    const double al = 1.3862943611198906 / (tau * tau), sw = std::sqrt(2 * al);   // spectral rms width
    const int nw = 801;
    const double wlo = 1 - 10 * sw, dw = 20 * sw / (nw - 1);
    auto ref = [&](double t, double y) {
        const double u = t - delay;
        cd env = 0;
        for (int q = 0; q < nw; ++q) {
            const double w = wlo + q * dw, zr = 0.5 * w * W * W;
            const cd q0(0, -zr), qq(-xf, -zr);
            const cd mono = std::sqrt(q0 / qq) * std::exp(cd(0, w * y * y) / (2.0 * qq));
            env += std::sqrt(M_PI / al) * std::exp(-(w - 1) * (w - 1) / (4 * al)) * mono * std::polar(1.0, -w * u) * dw / (2 * M_PI);
        }
        return (cd(0, A) * env).real();
    };
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> u(-1, 1);
    double worst = 0, peak = 0;
    for (int n = 0; n < 1500; ++n) {
        const double t = delay + 20 * u(rng), y = 60 * u(rng), r = ref(t, y);
        peak = std::max(peak, std::abs(r));
        worst = std::max(worst, std::abs(L.host_view().field(0, t, y, 0.3) - r));
    }
    check("lasy: short pulse at its focus x_f = z_R, propagated exactly to x = 0 = spectral paraxial beam",
          worst < 5e-3 * peak && peak > 0.7, fmt("max difference / peak %.1e", worst / peak) + fmt(" (peak %.3f)", peak));
}

// L7. In the PIC loop: the cartesian file of L4 injected through the boundary against the built-in laser,
//     the fields in the box after the pulse has entered.
static void lasy_run(Comm& comm) {
    Config c;
    const double dx = 2 * M_PI / 16;
    c.N[0] = 48; c.N[1] = 32; c.N[2] = 32;
    c.h[0] = dx; c.h[1] = 2 * dx; c.h[2] = 2 * dx;
    c.periodic[0] = false;
    c.t_end = 40 + 10 + c.N[0] * dx * 0.5;
    c.check();
    Config cb = c;
    LaserFileConfig F;
    F.name = "laser file"; F.path = "test_lasy_cart.h5"; F.delay = 40; F.t_center = 0;
    F.yc = 0.5 * c.N[1] * c.h[1] + 3; F.zc = 0.5 * c.N[2] * c.h[2] - 2;
    c.file_lasers.push_back(F);
    LaserConfig& B = cb.laser;
    B.enabled = true; B.a0 = 1.3; B.tau = 9; B.tc = 40; B.w0 = 5; B.w0z = 5; B.yc = F.yc; B.zc = F.zc; B.pol = 'y';
    Simulation a(c, comm), b(cb, comm);
    a.init(); b.init();
    while (!a.finished()) { a.step(); b.step(); }
    double d = 0, m = 0;
    const std::vector<double> ea = a.fields->ey.host(), eb = b.fields->ey.host(), za = a.fields->ez.host(), zb = b.fields->ez.host();
    for (size_t q = 0; q < ea.size(); ++q) { d = std::max({d, std::abs(ea[q] - eb[q]), std::abs(za[q] - zb[q])}); m = std::max(m, std::abs(eb[q])); }
    d = comm.allreduce_max(d); m = comm.allreduce_max(m);
    check("lasy: injected through the boundary = built-in laser, fields in the box", m > 0.5 && d < 2e-3 * m,
          fmt("max |E| %.3f", m) + fmt(", max difference / max %.1e", d / m));
}

// ---- RZ, 2D and the boosted antenna (decision 97) ----

// L8. RZ (RzFileLaser): a thetaMode file of the built-in Gaussian (env = i a0 e^{-r^2/W^2} g, pol along the particles' x) and of
//     a Laguerre-Gauss (0, 1) pulse (as L5), and a cartesian file of the Gaussian polarized along y (polar resampling), against
//     LaserConfig::field at the particles' (x, y) = (r cos phi, r sin phi); 2 modes kept.
static void lasy_rz_view(Comm&) {
    const double W = 5, tau = 9, delay = 40, A = 1.3;
    auto gauss = [&](double t, double r) { return cd(0, A) * std::exp(-r * r / (W * W)) * gauss_t(t, tau); };
    LasyFile F;
    F.path = "test_lasy_rz_theta.h5"; F.geometry = "thetaMode"; F.ncomp = 1;
    F.nt = 241; F.t0 = -24; F.dt = 0.2; F.n2 = 81; F.b0 = 0; F.db = 0.2;
    F.f = [&](int, double t, double, double r) { return gauss(t, r); };
    F.write();
    LasyFile G = F;
    G.path = "test_lasy_rz_lg.h5"; G.ncomp = 3;
    auto fr = [&](double t, double r) { return A * std::sqrt(2.0) * r / W * std::exp(-r * r / (W * W)) * gauss_t(t, tau); };
    G.f = [&](int comp, double t, double, double r) { return comp == 1 ? cd(0, fr(t, r)) : (comp == 2 ? cd(fr(t, r), 0) : cd(0)); };
    G.write();
    LasyFile C;
    C.path = "test_lasy_rz_cart.h5"; C.pxr = 0; C.pyr = 1;
    C.nt = 241; C.t0 = -24; C.dt = 0.2; C.n1 = C.n2 = 97; C.a0 = C.b0 = -12; C.da = C.db = 0.25;
    C.f = [&](int, double t, double a, double b) { return gauss(t, std::hypot(a, b)); };
    C.write();
    LaserFileConfig c;
    c.delay = delay; c.t_center = 0;
    LaserConfig B;
    B.enabled = true; B.a0 = A; B.tau = tau; B.tc = delay; B.w0 = W; B.w0z = W; B.pol = 'y';
    std::mt19937_64 rng(8);
    std::uniform_real_distribution<double> u(-1, 1);
    double err[3] = {0, 0, 0}, other = 0;
    for (int k = 0; k < 3; ++k) {
        c.path = k == 0 ? F.path : (k == 1 ? G.path : C.path);
        RzFileLaser L(c, lambda0, 2, 15);
        const RzFileLaserView v = L.host_view();
        LaserConfig R = B;
        if (k == 1) R.l = 1;
        if (k == 2) R.pol = 'z';
        for (int n = 0; n < 2000; ++n) {
            const double t = delay + 15 * u(rng), r = 9 * std::abs(u(rng)), ph = M_PI * u(rng);
            double ex, ey;
            v.field(t, r, std::cos(ph), std::sin(ph), ex, ey);
            const double ref = R.field(t, r * std::cos(ph), r * std::sin(ph));
            err[k] = std::max(err[k], std::abs((k == 2 ? ey : ex) - ref));
            other = std::max(other, std::abs(k == 2 ? ex : ey));
        }
    }
    check("lasy RZ: thetaMode Gaussian, Laguerre-Gauss (0, 1), cartesian file (polar resampling) = built-in pulses",
          err[0] < 2e-3 * A && err[1] < 2e-3 * A && err[2] < 4e-3 * A && other < 1e-12,
          fmt("max error / a0: Gaussian %.1e, LG %.1e", err[0] / A, err[1] / A) + fmt(", cartesian %.1e", err[2] / A) +
              fmt("; other component %.1e", other));
}

// L9. RZ propagation (Hankel transform per mode + FFT): a short round pulse (tau = 12, W = 24) at its focus x_f = z_R in the
//     file, propagated to z = 0, against the spectrally resolved paraxial beam (every omega with its own q, round beam: q0 / q
//     e^{i omega r^2 / (2 q)}), as L6.
static void lasy_rz_propagation(Comm&) {
    const double W = 24, tau = 12, delay = 60, A = 1, xf = 0.5 * W * W;
    LasyFile F;
    F.path = "test_lasy_rz_prop.h5"; F.geometry = "thetaMode"; F.ncomp = 1;
    F.nt = 161; F.t0 = -32; F.dt = 0.4; F.n2 = 81; F.b0 = 0; F.db = 1.5;
    F.f = [&](int, double t, double, double r) { return cd(0, A) * std::exp(-r * r / (W * W)) * gauss_t(t, tau); };
    F.write();
    LaserFileConfig c;
    c.path = F.path; c.delay = delay; c.t_center = 0; c.file_x = xf;
    RzFileLaser L(c, lambda0, 1, 110);
    const double al = 1.3862943611198906 / (tau * tau), sw = std::sqrt(2 * al);
    const int nw = 801;
    const double wlo = 1 - 10 * sw, dw = 20 * sw / (nw - 1);
    auto ref = [&](double t, double r) {
        const double uu = t - delay;
        cd env = 0;
        for (int q = 0; q < nw; ++q) {
            const double w = wlo + q * dw, zr = 0.5 * w * W * W;
            const cd q0(0, -zr), qq(-xf, -zr);
            const cd mono = (q0 / qq) * std::exp(cd(0, w * r * r) / (2.0 * qq));
            env += std::sqrt(M_PI / al) * std::exp(-(w - 1) * (w - 1) / (4 * al)) * mono * std::polar(1.0, -w * uu) * dw / (2 * M_PI);
        }
        return (cd(0, A) * env).real();
    };
    std::mt19937_64 rng(9);
    std::uniform_real_distribution<double> u(-1, 1);
    double worst = 0, peak = 0;
    for (int n = 0; n < 1500; ++n) {
        const double t = delay + 20 * u(rng), r = 60 * std::abs(u(rng)), ph = M_PI * u(rng), rf = ref(t, r);
        double ex, ey;
        L.host_view().field(t, r, std::cos(ph), std::sin(ph), ex, ey);
        peak = std::max(peak, std::abs(rf));
        worst = std::max(worst, std::abs(ex - rf));
    }
    check("lasy RZ: short pulse at its focus x_f = z_R, propagated exactly per mode to z = 0 = spectral paraxial beam",
          worst < 5e-3 * peak && peak > 0.5, fmt("max difference / peak %.1e", worst / peak) + fmt(" (peak %.3f)", peak));
}

namespace {
Config lasy_deck(const std::string& text, Comm& comm) {     // rank 0 writes, the others read after it (MPI)
    if (comm.rank == 0) { std::ofstream f("lasy_test_tmp.deck"); f << text; }
    comm.barrier();
    Config c = Config::from_deck(Deck::read("lasy_test_tmp.deck"));
    comm.barrier();                                           // nobody rewrites the file before all have read it
    return c;
}
// a file of the built-in Gaussian (a0 1, waist 2 um, 3 fs, at its focus) for the PIC runs: thetaMode (RZ) or a slab (2D)
void write_pic_file(const std::string& path, bool theta, double pxr, double pyr) {
    const double W = 2e-6 * k0, tau = 3e-15 * w0si;
    LasyFile F;
    F.path = path; F.pxr = pxr; F.pyr = pyr;
    F.nt = 301; F.t0 = -30; F.dt = 0.2;
    if (theta) { F.geometry = "thetaMode"; F.ncomp = 1; F.n2 = 161; F.b0 = 0; F.db = 0.25; }
    else { F.n1 = 1; F.n2 = 321; F.b0 = -40; F.db = 0.25; }
    F.f = [&](int, double t, double, double b) { return cd(0, 1) * std::exp(-b * b / (W * W)) * gauss_t(t, tau); };
    F.write();
}
}  // namespace

// L10. In the PIC loop: RZ lab (2 modes, polarization y: the m = 1 components), RZ boosted (gamma 2, antenna, polarization x),
//      2D lab and 2D boosted (antenna, Cartesian sheet): the file of the built-in Gaussian against the built-in laser on the same
//      grid, the transverse fields in the box at the end (boosted: gamma (1 - beta) = 0.27 of the lab field; RZ: the m = 1 slots hold
//      half of E_x); the boosted runs also keep the Gauss law at round-off.
static void lasy_rz_runs(Comm& comm) {
    if (comm.rank == 0) { write_pic_file("test_lasy_pic_theta_y.h5", true, 0, 1); write_pic_file("test_lasy_pic_theta_x.h5", true, 1, 0);
                          write_pic_file("test_lasy_pic_slab.h5", false, 1, 0); }
    comm.barrier();
    const char* name[4] = {"RZ lab", "RZ boosted antenna", "2D lab", "2D boosted antenna"};
    for (int k = 0; k < 4; ++k) {
        const bool rz = k < 2, boost = k % 2 == 1;
        std::ostringstream o;
        o << "grid {\n lambda0 = 0.8 um\n";
        if (rz) o << " geometry = rz\n modes = 2\n nz = 96\n nr = 48\n dz = 0.05 um\n dr = 0.125 um\n boundary_z = open\n";
        else o << " geometry = cartesian2d\n nx = 96\n ny = 96\n dx = 0.05 um\n dy = 0.125 um\n boundary_x = open\n";
        o << " t_end = " << (boost ? 30 : 20) << " fs\n}\nnumerics {\n solver = " << (rz || boost ? "irip" : "rip_cc") << "\n}\n";
        // the window starts when the pulse (peak at 8 fs, +-5 fs) is inside: the box covers t - z in [start - 16 fs, start]
        if (boost) o << "window {\n start = 17 fs\n}\nboost {\n gamma = 2\n}\n";
        o << "output {\n dir = test_out_lasy_rz\n every = 100000\n}\n";
        const std::string pol = rz ? (boost ? "x" : "y") : "y";
        const std::string file = rz ? (boost ? "test_lasy_pic_theta_x.h5" : "test_lasy_pic_theta_y.h5") : "test_lasy_pic_slab.h5";
        const std::string yc = rz ? "" : " y_center = 6 um\n";
        Config cf = lasy_deck(o.str() + "laser {\n file = " + file + "\n delay = 8 fs\n t_center = 0 fs\n" + yc + "}\n", comm);
        Config cb = lasy_deck(o.str() + "laser {\n a0 = 1\n polarization = " + pol + "\n duration = 3 fs\n delay = 8 fs\n waist = 2 um\n" +
                              yc + (boost ? " injection = antenna\n" : "") + "}\n", comm);
        Simulation a(cf, comm), b(cb, comm);
        a.init(); b.init();
        double g = 0;
        while (!a.finished()) {
            a.step(); b.step();
            if (boost && a.istep % 10 == 0) {
                if (rz) { double w, gg, r, d; a.rz_diagnostics(w, gg, r, d); g = std::max(g, gg); }
                else g = std::max(g, a.gauss_drift());
            }
        }
        double d = 0, m = 0;
        const Field* fa[2] = {rz ? &a.rz->er : &a.fields->ey, rz ? &a.rz->ep : &a.fields->ez};
        const Field* fb[2] = {rz ? &b.rz->er : &b.fields->ey, rz ? &b.rz->ep : &b.fields->ez};
        for (int c = 0; c < 2; ++c) {
            const std::vector<double> ea = fa[c]->host(), eb = fb[c]->host();
            for (size_t q = 0; q < ea.size(); ++q) { d = std::max(d, std::abs(ea[q] - eb[q])); m = std::max(m, std::abs(eb[q])); }
        }
        d = comm.allreduce_max(d); m = comm.allreduce_max(m);
        check(std::string("lasy ") + name[k] + ": file laser = built-in laser, fields in the box" + (boost ? ", Gauss law" : ""),
              m > (boost ? (rz ? 0.05 : 0.1) : 0.3) && d < 3e-3 * m && (!boost || (a.antenna && g < 1e-12)),
              fmt("max |E| %.3f", m) + fmt(", max difference / max %.1e", d / m) + (boost ? fmt(", max|dGauss| %.1e", g) : ""));
    }
}

void test_lasy(Comm& comm) {
    if (comm.rank == 0) { lasy_cartesian(comm); lasy_modes(comm); lasy_propagation(comm); lasy_rz_view(comm); lasy_rz_propagation(comm); }
    comm.barrier();
    lasy_run(comm);
    lasy_rz_runs(comm);
}
#else
void test_lasy(Comm&) { std::printf("lasy: skipped (built without HDF5)\n"); }
#endif
