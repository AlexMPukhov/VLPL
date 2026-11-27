// Beams from openPMD particle files (src/beam_import.hpp, docs/specs/structure3d.md).
#include "beam_import.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
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

// ---- a beam from a file, in portions: no rank holds the whole beam (memory rule; before 2026-11-26 every rank read
// the whole file). The file is passed up to three times, CHUNK particles at a time: the total weight (and the rms of
// the six coordinates for resample_smoothing), then the charge centroid and the mean gamma of the particles as they
// will be delivered (a dry run of the same stream), then the delivery. The arithmetic and the order of the random
// numbers are those of the former whole-file import, so the particles are the same to the last bit.
struct BeamFileSource::Impl {
#ifdef RIP_HDF5
    static constexpr size_t CHUNK = size_t(1) << 18;
    File F;
    BeamFileConfig c;
    unsigned long seed;
    std::string g;                                   // the species group
    size_t n = 0;                                    // particles in the file
    bool has_w = false, mw = false, has_py = false, has_uy = false;
    double wpow = 1;
    double W = 0;                                    // total weight of the file
    bool resample = false;
    double sd[6] = {0, 0, 0, 0, 0, 0};               // jitter of the resampling
    double f = 1;                                    // weight factor of `charge`
    double zc = 0, gmean = 1;                        // charge centroid along the file's z [m], mean gamma (delivered particles)
    double k0 = 0, wf = 0;
    // the stream
    std::vector<double> pos[3], mom[3], w;           // the chunk [c0, c0 + clen) of the file, SI and u = p / (m c)
    size_t c0 = 0, clen = 0;
    long k = 0, N = 0;                               // delivered, to deliver
    size_t j = 0;                                    // resampling: the source particle
    double cum = 0, u0 = 0;
    std::mt19937_64 rng;
    std::normal_distribution<double> nd{0.0, 1.0};

    Impl(const BeamFileConfig& cc, unsigned long s) : F(cc.path), c(cc), seed(s) {}

    std::vector<double> part(const std::string& p, size_t off, size_t len, bool required = true) const {
        if (!F.exists(p)) {
            if (required) F.fail("missing record component " + p);
            return std::vector<double>(len, 0.0);
        }
        const double u = F.attr1(p, "unitSI", 1.0);
        std::vector<double> v;
        if (F.group(p)) v.assign(len, F.attr1(p, "value", 0.0));
        else {
            h5::Id d = h5::open_dataset(F.f, p);
            if (F.count(p) != n) F.fail(p + " has " + std::to_string(F.count(p)) + " entries, expected " + std::to_string(n));
            v.resize(len);
            if (len) h5::read<double>(d, v.data(), {hsize_t(off)}, {hsize_t(len)});
        }
        for (double& x : v) x *= u;
        return v;
    }
    std::vector<double> weights(size_t off, size_t len) const {
        return has_w ? part(g + "weighting", off, len) : std::vector<double>(len, 1.0);
    }
    void load(size_t off) {                          // the chunk that starts at particle `off`
        c0 = off;
        clen = std::min(CHUNK, n - off);
        w = weights(c0, clen);
        const char* ax[3] = {"x", "y", "z"};
        for (int a = 0; a < 3; ++a) {
            pos[a] = part(g + "position/" + ax[a], c0, clen, a != 1 || has_py);
            const std::vector<double> o = part(g + "positionOffset/" + ax[a], c0, clen, false);
            for (size_t q = 0; q < clen; ++q) pos[a][q] += o[q];
            mom[a] = part(g + "momentum/" + ax[a], c0, clen, a != 1 || has_uy);
        }
        // macroWeighted = 1: the record holds the sum over the macroparticle, w^weightingPower times the single particle
        for (size_t q = 0; q < clen; ++q) {
            const double s = mw ? std::pow(w[q], -wpow) / (ME * CL) : 1.0 / (ME * CL);
            for (int a = 0; a < 3; ++a) mom[a][q] *= s;
        }
    }
    void need(size_t i) { if (clen == 0 || i < c0 || i >= c0 + clen) load(i - i % CHUNK); }
    void rewind() {
        k = 0; j = 0; cum = 0; clen = 0;
        if (resample) {
            rng.seed(seed * 0x9e3779b97f4a7c15ULL + 12345);
            nd.reset();
            std::uniform_real_distribution<double> ud(0.0, 1.0);
            u0 = ud(rng);
        }
    }
    // the next particle: position [m] and u along the file's x, y, z, weight (number of electrons, after `charge`)
    void one(double v[7]) {
        if (!resample) {
            const size_t i = size_t(k);
            need(i);
            const size_t q = i - c0;
            for (int a = 0; a < 3; ++a) { v[a] = pos[a][q]; v[3 + a] = mom[a][q]; }
            v[6] = c.charge > 0 ? w[q] * f : w[q];
        } else {                                     // systematic, probabilities ~ weight, equal weights (GEM-PIC-RZ)
            const double target = (double(k) + u0) / double(N) * W;
            for (;;) {
                need(j);
                if (!(j + 1 < n && cum + w[j - c0] < target)) break;
                cum += w[j - c0];
                ++j;
            }
            const size_t q = j - c0;
            for (int a = 0; a < 3; ++a) {
                v[a] = pos[a][q] + (c.smoothing > 0 ? sd[a] * nd(rng) : 0.0);
                v[3 + a] = mom[a][q] + (c.smoothing > 0 ? sd[3 + a] * nd(rng) : 0.0);
            }
            const double wn = W / double(N);
            v[6] = c.charge > 0 ? wn * f : wn;
        }
        ++k;
    }
#else
    Impl(const BeamFileConfig&, unsigned long) {}
#endif
};

BeamFileSource::BeamFileSource(const BeamFileConfig& c, double lambda0, unsigned long seed) {
#ifndef RIP_HDF5
    (void)lambda0; (void)seed; (void)p_;
    throw std::runtime_error(c.path + ": beams from files need an HDF5 build (make HDF5=1)");
#else
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    std::unique_ptr<Impl> I(new Impl(c, seed));
    const File& F = I->F;
    if (!F.exists("/") || F.attr_s("/", "openPMD").empty()) F.fail("not an openPMD file (no attribute openPMD)");
    format = "openPMD " + F.attr_s("/", "openPMD");
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
    iteration = it;
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
    species = sp;
    const std::string g = ip + sp + "/";
    I->g = g;
    const size_t n = F.count(g + "position/x");
    I->n = n;
    if (n == 0) F.fail("species '" + sp + "' holds no particles");
    const double q = F.exists(g + "charge") ? I->part(g + "charge", 0, 1)[0] : -QE;
    const double m = F.exists(g + "mass") ? I->part(g + "mass", 0, 1)[0] : ME;
    if (std::fabs(std::fabs(q) / QE - 1) > 1e-3 || std::fabs(m / ME - 1) > 1e-3)
        F.fail("species '" + sp + "' is not electrons or positrons (charge " + std::to_string(q) + " C, mass " + std::to_string(m) + " kg)");
    sign = q < 0 ? -1 : 1;
    I->has_w = F.exists(g + "weighting");
    I->has_py = F.exists(g + "position/y");
    I->has_uy = F.exists(g + "momentum/y");
    I->mw = F.exists(g + "momentum") && F.attr1(g + "momentum", "macroWeighted", 0.0) != 0.0;
    I->wpow = F.exists(g + "momentum") ? F.attr1(g + "momentum", "weightingPower", 1.0) : 1.0;
    // pass 1: the total weight
    double W = 0;
    for (size_t off = 0; off < n; off += Impl::CHUNK)
        for (double x : I->weights(off, std::min(Impl::CHUNK, n - off))) W += x;
    I->W = W;
    I->resample = c.macroparticles > 0 && size_t(c.macroparticles) != n;
    I->N = I->resample ? c.macroparticles : long(n);
    if (I->resample && c.smoothing > 0) {            // the rms of the six coordinates
        double m1[6] = {0, 0, 0, 0, 0, 0}, m2[6] = {0, 0, 0, 0, 0, 0};
        for (size_t off = 0; off < n; off += Impl::CHUNK) {
            I->load(off);
            for (size_t e = 0; e < I->clen; ++e) {
                const double v[6] = {I->pos[0][e], I->pos[1][e], I->pos[2][e], I->mom[0][e], I->mom[1][e], I->mom[2][e]};
                for (int a = 0; a < 6; ++a) { m1[a] += I->w[e] / W * v[a]; m2[a] += I->w[e] / W * v[a] * v[a]; }
            }
        }
        for (int a = 0; a < 6; ++a) I->sd[a] = c.smoothing * std::sqrt(std::max(0.0, m2[a] - m1[a] * m1[a]));
    }
    if (c.charge > 0) I->f = c.charge / (W * QE);    // rescale to the total charge
    // to code units: lengths k0, weight w k0^3 / n_c
    const double k0 = 2 * M_PI / lambda0, w0 = CL * k0, nc = EPS0 * ME * w0 * w0 / (QE * QE);
    I->k0 = k0;
    I->wf = k0 * k0 * k0 / nc;
    // pass 2: the charge centroid and the mean gamma of the particles as delivered
    I->rewind();
    double zc = 0, Wt = 0, gs = 0;
    for (long e = 0; e < I->N; ++e) {
        double v[7];
        I->one(v);
        zc += v[6] * v[2]; Wt += v[6];
        gs += v[6] * std::sqrt(1 + v[3] * v[3] + v[4] * v[4] + v[5] * v[5]);
    }
    I->zc = zc / Wt;
    I->gmean = gs / Wt;
    I->rewind();
    p_ = I.release();
#endif
}

BeamFileSource::~BeamFileSource() { delete p_; }

long BeamFileSource::left() const {
#ifdef RIP_HDF5
    return p_->N - p_->k;
#else
    return 0;
#endif
}
double BeamFileSource::gamma_mean() const {
#ifdef RIP_HDF5
    return p_->gmean;
#else
    return 1;
#endif
}

BeamParticles BeamFileSource::next(long count) {
    BeamParticles B;
    B.sign = sign; B.format = format; B.species = species; B.iteration = iteration;
#ifdef RIP_HDF5
    Impl& I = *p_;
    const BeamFileConfig& c = I.c;
    const size_t N = size_t(std::max(0L, std::min(count, left())));
    B.x.resize(N); B.y.resize(N); B.z.resize(N); B.ux.resize(N); B.uy.resize(N); B.uz.resize(N); B.w.resize(N);
    for (size_t e = 0; e < N; ++e) {
        double v[7];
        I.one(v);
        B.x[e] = c.x_center + (v[2] - I.zc) * I.k0;
        B.y[e] = c.y_center + c.y_offset + v[0] * I.k0;
        B.z[e] = c.z_center + c.z_offset + v[1] * I.k0;
        B.ux[e] = v[5]; B.uy[e] = v[3]; B.uz[e] = v[4];
        B.w[e] = v[6] * I.wf;
    }
#else
    (void)count;
#endif
    return B;
}

// the whole beam at once (tests and small files; the run reads it in portions, Simulation::load_file_species)
BeamParticles import_beam(const BeamFileConfig& c, double lambda0, unsigned long seed) {
    BeamFileSource S(c, lambda0, seed);
    return S.next(S.left());
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

// ---- a bunch without a file (block `bunch <name>`)
struct BunchSource::Rng {
    std::mt19937_64 gen;
    std::normal_distribution<double> nd{0.0, 1.0};
};

BunchSource::BunchSource(const BeamFileConfig& c, double lambda0, unsigned long seed) : rng_(new Rng), c_(c) {
    const BunchShape& g = c.gen;
    if (!(g.charge > 0) || g.n < 1 || !(g.gamma > 1))
        throw std::runtime_error(c.path + ": a bunch needs charge > 0, macroparticles >= 1 and an energy above the rest energy");
    rng_->gen.seed(seed * 0x9e3779b97f4a7c15ULL + 54321);
    const double k0 = 2 * M_PI / lambda0, w0 = CL * k0, nc = EPS0 * ME * w0 * w0 / (QE * QE);
    w_ = g.charge / (QE * double(g.n)) * k0 * k0 * k0 / nc;
    left_ = g.n;
}

BunchSource::~BunchSource() { delete rng_; }

BeamParticles BunchSource::next(long count) {
    const BunchShape& g = c_.gen;
    const size_t n = size_t(std::max(0L, std::min(count, left_)));
    left_ -= long(n);
    BeamParticles B;
    B.sign = g.sign;
    B.format = "generated";
    B.species = g.sign < 0 ? "electron" : "positron";
    B.x.resize(n); B.y.resize(n); B.z.resize(n); B.ux.resize(n); B.uy.resize(n); B.uz.resize(n); B.w.assign(n, w_);
    auto& G = rng_->gen;
    auto& N = rng_->nd;
    const double su[2] = {g.sigma[1] > 0 ? g.eps[0] / g.sigma[1] : 0.0, g.sigma[2] > 0 ? g.eps[1] / g.sigma[2] : 0.0};
    for (size_t k = 0; k < n; ++k) {                 // seven draws per particle, always in this order
        const double d[7] = {N(G), N(G), N(G), N(G), N(G), N(G), N(G)};
        B.x[k] = c_.x_center + g.sigma[0] * d[0];
        B.y[k] = c_.y_center + c_.y_offset + g.sigma[1] * d[1];
        B.z[k] = c_.z_center + c_.z_offset + g.sigma[2] * d[2];
        B.uy[k] = su[0] * d[3];
        B.uz[k] = su[1] * d[4];
        const double gam = g.gamma * (1 + g.de * d[5]);
        const double u2 = gam * gam - 1 - B.uy[k] * B.uy[k] - B.uz[k] * B.uz[k];
        if (!(u2 > 0)) throw std::runtime_error(c_.path + ": the energy is too small for the transverse momenta (emittance / sigma) or the energy spread");
        B.ux[k] = std::sqrt(u2);
    }
    return B;
}

void MomentSums::add(const BeamParticles& b) {
    for (size_t k = 0; k < b.size(); ++k) {
        const double w = b.w[k], g = std::sqrt(1 + b.ux[k] * b.ux[k] + b.uy[k] * b.uy[k] + b.uz[k] * b.uz[k]) - g0_;
        const double d[3] = {b.x[k] - c_[0], b.y[k] - c_[1], b.z[k] - c_[2]}, u[2] = {b.uy[k], b.uz[k]};
        W_ += w; g1_ += w * g; g2_ += w * g * g; ux_ += w * b.ux[k];
        for (int a = 0; a < 3; ++a) { r1_[a] += w * d[a]; r2_[a] += w * d[a] * d[a]; }
        for (int a = 0; a < 2; ++a) { u1_[a] += w * u[a]; u2_[a] += w * u[a] * u[a]; ru_[a] += w * d[a + 1] * u[a]; }
    }
    n_ += long(b.size());
}

FileMoments MomentSums::result(double lambda0) const {
    FileMoments M;
    M.n = n_;
    if (W_ <= 0) return M;
    const double k0 = 2 * M_PI / lambda0, w0 = CL * k0, nc = EPS0 * ME * w0 * w0 / (QE * QE);
    const double gm = g1_ / W_;
    M.Q = W_ * nc / (k0 * k0 * k0) * QE;
    M.ekin_mev = (g0_ + gm - 1) * 0.51099895000;
    M.de_rel = std::sqrt(std::max(0.0, g2_ / W_ - gm * gm)) / (g0_ + gm);
    double var[3];
    for (int a = 0; a < 3; ++a) {
        const double m = r1_[a] / W_;
        var[a] = std::max(0.0, r2_[a] / W_ - m * m);
        M.mean[a] = (c_[a] + m) / k0;
        M.sigma[a] = std::sqrt(var[a]) / k0;
    }
    for (int a = 0; a < 2; ++a) {
        const double um = u1_[a] / W_, uu = std::max(0.0, u2_[a] / W_ - um * um), ru = ru_[a] / W_ - r1_[a + 1] / W_ * um;
        M.eps_n[a] = std::sqrt(std::max(0.0, var[a + 1] * uu - ru * ru)) / k0;
    }
    return M;
}
