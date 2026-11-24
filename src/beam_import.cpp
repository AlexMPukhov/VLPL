// Beams from openPMD particle files (src/beam_import.hpp, docs/specs/structure3d.md).
#include "beam_import.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#ifdef RIP_HDF5
#include "h5.hpp"
#endif

namespace {
constexpr double QE = 1.602176634e-19, ME = 9.1093837015e-31, CL = 299792458.0, EPS0 = 8.8541878128e-12;

#ifdef RIP_HDF5
std::string join(const std::vector<std::string>& v) {
    std::string s;
    for (const auto& x : v) s += (s.empty() ? "" : ", ") + x;
    return s.empty() ? "none" : s;
}

struct File {
    std::string path;
    h5::Id f;
    explicit File(const std::string& p) : path(p), f(h5::open_file(p)) {}
    [[noreturn]] void fail(const std::string& m) const { throw std::runtime_error(path + ": " + m); }
    bool exists(const std::string& p) const { return h5::exists(f, p); }
    bool group(const std::string& p) const { return h5::is_group(f, p); }
    double attr1(const std::string& p, const std::string& a, double def) const {
        h5::Id o(H5Oopen(f, p.c_str(), H5P_DEFAULT), ("no object " + p).c_str());
        return h5::has_attr(o, a) ? h5::attr_vec<double>(o, a).at(0) : def;
    }
    std::string attr_s(const std::string& p, const std::string& a) const {
        h5::Id o(H5Oopen(f, p.c_str(), H5P_DEFAULT), ("no object " + p).c_str());
        return h5::has_attr(o, a) ? h5::attr_string(o, a) : std::string();
    }
    size_t count(const std::string& p) const {      // dataset size or the shape of a constant record
        if (group(p)) {
            h5::Id o(H5Oopen(f, p.c_str(), H5P_DEFAULT), ("no object " + p).c_str());
            const std::vector<double> s = h5::attr_vec<double>(o, "shape");
            return s.empty() ? 0 : size_t(s[0]);
        }
        h5::Id d = h5::open_dataset(f, p);
        size_t n = 1;
        for (hsize_t x : h5::dims(d)) n *= size_t(x);
        return n;
    }
    // record component (dataset or constant group with "value"), times unitSI; zeros if absent and not required
    std::vector<double> component(const std::string& p, size_t n, bool required = true) const {
        if (!exists(p)) {
            if (required) fail("missing record component " + p);
            return std::vector<double>(n, 0.0);
        }
        const double u = attr1(p, "unitSI", 1.0);
        std::vector<double> v;
        if (group(p)) v.assign(n, attr1(p, "value", 0.0));
        else {
            h5::Id d = h5::open_dataset(f, p);
            if (count(p) != n) fail(p + " has " + std::to_string(count(p)) + " entries, expected " + std::to_string(n));
            v.resize(n);
            if (n) h5::read<double>(d, v.data(), {0}, {hsize_t(n)});
        }
        for (double& x : v) x *= u;
        return v;
    }
};
#endif
}  // namespace

BeamParticles import_beam(const BeamFileConfig& c, double lambda0, unsigned long seed) {
#ifndef RIP_HDF5
    (void)lambda0; (void)seed;
    throw std::runtime_error(c.path + ": beams from files need an HDF5 build (make HDF5=1)");
#else
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    const File F(c.path);
    BeamParticles B;
    if (!F.exists("/") || F.attr_s("/", "openPMD").empty()) F.fail("not an openPMD file (no attribute openPMD)");
    B.format = "openPMD " + F.attr_s("/", "openPMD");
    std::string base = F.attr_s("/", "basePath");
    if (base.empty()) base = "/data/%T/";
    const std::string root = base.substr(0, base.find("%T"));
    std::vector<long> its;
    {
        h5::Id g = h5::open_group(F.f, root.empty() ? "/" : root);
        for (const std::string& ch : h5::children(g)) {
            char* end = nullptr;
            const long v = std::strtol(ch.c_str(), &end, 10);
            if (end != ch.c_str() && *end == '\0') its.push_back(v);
        }
    }
    if (its.empty()) F.fail("no iteration under " + root);
    std::sort(its.begin(), its.end());
    long it = c.iteration < 0 ? its.back() : c.iteration;
    if (!std::binary_search(its.begin(), its.end(), it)) F.fail("iteration " + std::to_string(it) + " not in the file");
    B.iteration = it;
    std::string pp = F.attr_s("/", "particlesPath");
    if (pp.empty()) pp = "particles/";
    if (pp.back() != '/') pp += '/';
    const std::string ip = root + std::to_string(it) + "/" + pp;
    std::vector<std::string> sps;
    { h5::Id g = h5::open_group(F.f, ip); sps = h5::children(g); }
    std::string sp = c.species;
    if (sp.empty()) {                               // the only species with particles
        std::vector<std::string> full;
        for (const auto& s : sps)
            if (F.exists(ip + s + "/position/x") && F.count(ip + s + "/position/x") > 0) full.push_back(s);
        if (full.size() != 1) F.fail("give file_species: the file holds the species " + join(sps));
        sp = full[0];
    } else if (std::find(sps.begin(), sps.end(), sp) == sps.end()) F.fail("species '" + sp + "' not found (species: " + join(sps) + ")");
    B.species = sp;
    const std::string g = ip + sp + "/";
    const size_t n = F.count(g + "position/x");
    if (n == 0) F.fail("species '" + sp + "' holds no particles");
    const double q = F.exists(g + "charge") ? F.component(g + "charge", n)[0] : -QE;
    const double m = F.exists(g + "mass") ? F.component(g + "mass", n)[0] : ME;
    if (std::fabs(std::fabs(q) / QE - 1) > 1e-3 || std::fabs(m / ME - 1) > 1e-3)
        F.fail("species '" + sp + "' is not electrons or positrons (charge " + std::to_string(q) + " C, mass " + std::to_string(m) + " kg)");
    B.sign = q < 0 ? -1 : 1;
    std::vector<double> w = F.exists(g + "weighting") ? F.component(g + "weighting", n) : std::vector<double>(n, 1.0);
    std::vector<double> pos[3], mom[3];
    const char* ax[3] = {"x", "y", "z"};
    for (int a = 0; a < 3; ++a) {
        pos[a] = F.component(g + "position/" + ax[a], n, a != 1 || F.exists(g + "position/y"));
        const std::vector<double> off = F.component(g + "positionOffset/" + ax[a], n, false);
        for (size_t k = 0; k < n; ++k) pos[a][k] += off[k];
        mom[a] = F.component(g + "momentum/" + ax[a], n, a != 1 || F.exists(g + "momentum/y"));
    }
    // macroWeighted = 1: the record holds the sum over the macroparticle, w^weightingPower times the single particle
    const bool mw = F.exists(g + "momentum") && F.attr1(g + "momentum", "macroWeighted", 0.0) != 0.0;
    const double wpow = F.exists(g + "momentum") ? F.attr1(g + "momentum", "weightingPower", 1.0) : 1.0;
    for (size_t k = 0; k < n; ++k) {
        const double f = mw ? std::pow(w[k], -wpow) / (ME * CL) : 1.0 / (ME * CL);
        for (int a = 0; a < 3; ++a) mom[a][k] *= f;
    }
    // optional resampling (systematic, probabilities ~ weight, equal weights; GEM-PIC-RZ)
    double W = 0;
    for (double x : w) W += x;
    if (c.macroparticles > 0 && size_t(c.macroparticles) != n) {
        const long M = c.macroparticles;
        std::mt19937_64 rng(seed * 0x9e3779b97f4a7c15ULL + 12345);
        std::uniform_real_distribution<double> ud(0.0, 1.0);
        std::normal_distribution<double> nd(0.0, 1.0);
        double sd[6] = {0, 0, 0, 0, 0, 0};
        if (c.smoothing > 0) {
            double m1[6] = {0, 0, 0, 0, 0, 0}, m2[6] = {0, 0, 0, 0, 0, 0};
            for (size_t k = 0; k < n; ++k) {
                const double v[6] = {pos[0][k], pos[1][k], pos[2][k], mom[0][k], mom[1][k], mom[2][k]};
                for (int a = 0; a < 6; ++a) { m1[a] += w[k] / W * v[a]; m2[a] += w[k] / W * v[a] * v[a]; }
            }
            for (int a = 0; a < 6; ++a) sd[a] = c.smoothing * std::sqrt(std::max(0.0, m2[a] - m1[a] * m1[a]));
        }
        std::vector<double> np[3], nm[3], nw(size_t(M), W / double(M));
        for (int a = 0; a < 3; ++a) { np[a].resize(size_t(M)); nm[a].resize(size_t(M)); }
        const double u0 = ud(rng);
        double cum = 0;
        size_t j = 0;
        for (long k = 0; k < M; ++k) {
            const double target = (k + u0) / double(M) * W;
            while (j + 1 < n && cum + w[j] < target) cum += w[j++];
            for (int a = 0; a < 3; ++a) {
                np[a][size_t(k)] = pos[a][j] + (c.smoothing > 0 ? sd[a] * nd(rng) : 0.0);
                nm[a][size_t(k)] = mom[a][j] + (c.smoothing > 0 ? sd[3 + a] * nd(rng) : 0.0);
            }
        }
        for (int a = 0; a < 3; ++a) { pos[a].swap(np[a]); mom[a].swap(nm[a]); }
        w.swap(nw);
    }
    const size_t N = w.size();
    if (c.charge > 0) {                             // rescale to the total charge
        const double f = c.charge / (W * QE);
        for (double& x : w) x *= f;
    }
    // to code units: lengths k0, weight w k0^3 / n_c
    const double k0 = 2 * M_PI / lambda0, w0 = CL * k0, nc = EPS0 * ME * w0 * w0 / (QE * QE), wf = k0 * k0 * k0 / nc;
    double zc = 0, Wt = 0;
    for (size_t k = 0; k < N; ++k) { zc += w[k] * pos[2][k]; Wt += w[k]; }
    zc /= Wt;
    B.x.resize(N); B.y.resize(N); B.z.resize(N); B.ux.resize(N); B.uy.resize(N); B.uz.resize(N); B.w.resize(N);
    for (size_t k = 0; k < N; ++k) {
        B.x[k] = c.x_center + (pos[2][k] - zc) * k0;
        B.y[k] = c.y_center + c.y_offset + pos[0][k] * k0;
        B.z[k] = c.z_center + c.z_offset + pos[1][k] * k0;
        B.ux[k] = mom[2][k]; B.uy[k] = mom[0][k]; B.uz[k] = mom[1][k];
        B.w[k] = w[k] * wf;
    }
    return B;
#endif
}

FileMoments file_moments(const BeamParticles& b, double lambda0) {
    FileMoments M;
    const double k0 = 2 * M_PI / lambda0, w0 = CL * k0, nc = EPS0 * ME * w0 * w0 / (QE * QE), mec2 = 0.51099895000;
    const size_t n = b.size();
    M.n = long(n);
    double W = 0, g1 = 0, r1[3] = {0, 0, 0};
    for (size_t k = 0; k < n; ++k) {
        const double g = std::sqrt(1 + b.ux[k] * b.ux[k] + b.uy[k] * b.uy[k] + b.uz[k] * b.uz[k]);
        W += b.w[k]; g1 += b.w[k] * g;
        r1[0] += b.w[k] * b.x[k]; r1[1] += b.w[k] * b.y[k]; r1[2] += b.w[k] * b.z[k];
    }
    if (W <= 0) return M;
    g1 /= W;
    for (double& r : r1) r /= W;
    // centred second moments (two passes)
    double gg = 0, rr[3] = {0, 0, 0}, uu[2] = {0, 0}, ru[2] = {0, 0}, um[2] = {0, 0};
    for (size_t k = 0; k < n; ++k) { um[0] += b.w[k] * b.uy[k]; um[1] += b.w[k] * b.uz[k]; }
    um[0] /= W; um[1] /= W;
    for (size_t k = 0; k < n; ++k) {
        const double g = std::sqrt(1 + b.ux[k] * b.ux[k] + b.uy[k] * b.uy[k] + b.uz[k] * b.uz[k]);
        const double d[3] = {b.x[k] - r1[0], b.y[k] - r1[1], b.z[k] - r1[2]}, du[2] = {b.uy[k] - um[0], b.uz[k] - um[1]};
        gg += b.w[k] * (g - g1) * (g - g1);
        for (int a = 0; a < 3; ++a) rr[a] += b.w[k] * d[a] * d[a];
        for (int a = 0; a < 2; ++a) { uu[a] += b.w[k] * du[a] * du[a]; ru[a] += b.w[k] * d[a + 1] * du[a]; }
    }
    M.Q = W * nc / (k0 * k0 * k0) * QE;
    M.ekin_mev = (g1 - 1) * mec2;
    M.de_rel = std::sqrt(gg / W) / g1;
    for (int a = 0; a < 3; ++a) { M.mean[a] = r1[a] / k0; M.sigma[a] = std::sqrt(rr[a] / W) / k0; }
    for (int a = 0; a < 2; ++a) M.eps_n[a] = std::sqrt(std::max(0.0, rr[a + 1] / W * uu[a] / W - ru[a] * ru[a] / (W * W))) / k0;
    return M;
}
