#include "beam_diag.hpp"
#include "fsutil.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "portable.hpp"
#include "sim.hpp"

std::vector<BeamDiagConfig> beam_diags_from_deck(const Deck& d, long default_every) {
    std::vector<BeamDiagConfig> v;
    for (const std::string& b : d.block_names()) {
        if (b.rfind("diagnostic ", 0) != 0) continue;
        BeamDiagConfig c;
        c.name = b.substr(11);
        c.species = d.str(b, "species");
        c.every = d.integer(b, "every", default_every);
        c.gamma_min = d.num(b, "gamma_min", Kind::None, 1.0);
        c.bins = int(d.integer(b, "bins", 200));
        c.e_max_mev = d.num(b, "e_max_mev", Kind::None, 1000.0);
        c.particles = d.flag(b, "particles", false);
        if (c.every <= 0) throw std::runtime_error(b + ": every must be > 0");
        if (c.bins < 0 || c.e_max_mev <= 0) throw std::runtime_error(b + ": bins >= 0 and e_max_mev > 0");
        v.push_back(c);
    }
    return v;
}

// Kernel views of a species (raw pointers, captured by value in the kernels).
struct BeamViews { const double *x, *y, *z, *ux, *uy, *uz, *w; double gmin; };
static BeamViews views(const Species& s, double gamma_min) {
    return BeamViews{s.ptr<double>(Species::X), s.ptr<double>(Species::Y), s.ptr<double>(Species::Z),
                     s.ptr<double>(Species::UX), s.ptr<double>(Species::UY), s.ptr<double>(Species::UZ),
                     s.ptr<double>(Species::W), gamma_min};
}
HD inline double gamma_of(const BeamViews& V, long p) {
    return sqrt(1.0 + V.ux[p] * V.ux[p] + V.uy[p] * V.uy[p] + V.uz[p] * V.uz[p]);
}

BeamMoments beam_moments(const Species& s, double gamma_min, Comm& comm) {
    BeamMoments m;
    const long n = long(s.size());
    const BeamViews V = views(s, gamma_min);
    // pass 1: count, weight, weighted sums of gamma, x, y, z, uy, uz
    double a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (n > 0) {
        a[0] = par_sum(n, [=] HD(long p) { return gamma_of(V, p) >= V.gmin ? 1.0 : 0.0; });
        a[1] = par_sum(n, [=] HD(long p) { return gamma_of(V, p) >= V.gmin ? V.w[p] : 0.0; });
        a[2] = par_sum(n, [=] HD(long p) { const double g = gamma_of(V, p); return g >= V.gmin ? V.w[p] * g : 0.0; });
        a[3] = par_sum(n, [=] HD(long p) { return gamma_of(V, p) >= V.gmin ? V.w[p] * V.x[p] : 0.0; });
        a[4] = par_sum(n, [=] HD(long p) { return gamma_of(V, p) >= V.gmin ? V.w[p] * V.y[p] : 0.0; });
        a[5] = par_sum(n, [=] HD(long p) { return gamma_of(V, p) >= V.gmin ? V.w[p] * V.z[p] : 0.0; });
        a[6] = par_sum(n, [=] HD(long p) { return gamma_of(V, p) >= V.gmin ? V.w[p] * V.uy[p] : 0.0; });
        a[7] = par_sum(n, [=] HD(long p) { return gamma_of(V, p) >= V.gmin ? V.w[p] * V.uz[p] : 0.0; });
        m.gmax = par_max_abs(n, [=] HD(long p) { const double g = gamma_of(V, p); return g >= V.gmin ? g : 0.0; });
    }
    comm.allreduce_sum(a, 8);
    m.gmax = comm.allreduce_max(m.gmax);
    m.count = a[0]; m.wsum = a[1];
    if (m.wsum <= 0) return m;
    const double iw = 1.0 / m.wsum;
    const double gm = a[2] * iw, xm = a[3] * iw, ym = a[4] * iw, zm = a[5] * iw, vym = a[6] * iw, vzm = a[7] * iw;
    // pass 2: centred second moments
    double b[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (n > 0) {
        b[0] = par_sum(n, [=] HD(long p) { const double g = gamma_of(V, p); return g >= V.gmin ? V.w[p] * (g - gm) * (g - gm) : 0.0; });
        b[1] = par_sum(n, [=] HD(long p) { const double d = V.x[p] - xm; return gamma_of(V, p) >= V.gmin ? V.w[p] * d * d : 0.0; });
        b[2] = par_sum(n, [=] HD(long p) { const double d = V.y[p] - ym; return gamma_of(V, p) >= V.gmin ? V.w[p] * d * d : 0.0; });
        b[3] = par_sum(n, [=] HD(long p) { const double d = V.z[p] - zm; return gamma_of(V, p) >= V.gmin ? V.w[p] * d * d : 0.0; });
        b[4] = par_sum(n, [=] HD(long p) { const double d = V.uy[p] - vym; return gamma_of(V, p) >= V.gmin ? V.w[p] * d * d : 0.0; });
        b[5] = par_sum(n, [=] HD(long p) { const double d = V.uz[p] - vzm; return gamma_of(V, p) >= V.gmin ? V.w[p] * d * d : 0.0; });
        b[6] = par_sum(n, [=] HD(long p) { return gamma_of(V, p) >= V.gmin ? V.w[p] * (V.y[p] - ym) * (V.uy[p] - vym) : 0.0; });
        b[7] = par_sum(n, [=] HD(long p) { return gamma_of(V, p) >= V.gmin ? V.w[p] * (V.z[p] - zm) * (V.uz[p] - vzm) : 0.0; });
    }
    comm.allreduce_sum(b, 8);
    for (double& q : b) q *= iw;
    m.gmean = gm; m.xmean = xm;
    m.gsig = std::sqrt(b[0]); m.sx = std::sqrt(b[1]); m.sy = std::sqrt(b[2]); m.sz = std::sqrt(b[3]);
    m.eps_y = std::sqrt(std::max(0.0, b[2] * b[4] - b[6] * b[6]));
    m.eps_z = std::sqrt(std::max(0.0, b[3] * b[5] - b[7] * b[7]));
    return m;
}

std::vector<double> beam_spectrum(const Species& s, double gamma_min, int bins, double ek_max, Comm& comm) {
    std::vector<double> h(size_t(bins), 0.0);
    const long n = long(s.size());
    if (bins > 0 && n > 0) {
        Buf<double> d{size_t(bins)};
        double* H = d.data();
        const BeamViews V = views(s, gamma_min);
        const double mass = s.m, scale = bins / ek_max;
        par_for(n, [=] HD(long p) {
            const double g = gamma_of(V, p);
            if (g < V.gmin) return;
            const double e = (g - 1.0) * mass * scale;
            if (e >= 0 && e < bins) atomic_add(&H[int(e)], V.w[p]);
        });
        d.to_host(h.data(), 0, size_t(bins));
    }
    if (bins > 0) comm.allreduce_sum(h.data(), bins);
    return h;
}

std::vector<double> beam_particles(const Species& s, double gamma_min) {
    const long n = long(s.size());
    std::vector<double> out;
    if (n == 0) return out;
    const BeamViews V = views(s, gamma_min);
    Buf<int> list{size_t(n)}, count{1};
    int* L = list.data(); int* C = count.data();
    par_for(n, [=] HD(long p) { if (gamma_of(V, p) >= V.gmin) L[atomic_inc(C)] = int(p); });
    int k = 0;
    count.to_host(&k, 0, 1);
    if (k == 0) return out;
    Buf<double> pack{size_t(k) * 8};
    double* Q = pack.data();
    par_for(k, [=] HD(long q) {
        const long p = L[q];
        double* o = Q + 8 * q;
        o[0] = double(p); o[1] = V.x[p]; o[2] = V.y[p]; o[3] = V.z[p];
        o[4] = V.ux[p]; o[5] = V.uy[p]; o[6] = V.uz[p]; o[7] = V.w[p];
    });
    std::vector<double> h(static_cast<size_t>(k) * 8);
    pack.to_host(h.data(), 0, h.size());
    std::vector<int> order(static_cast<size_t>(k));
    for (int q = 0; q < k; ++q) order[size_t(q)] = q;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return h[8 * size_t(a)] < h[8 * size_t(b)]; });
    out.resize(size_t(k) * 7);
    for (int q = 0; q < k; ++q)
        for (int c = 0; c < 7; ++c) out[7 * size_t(q) + size_t(c)] = h[8 * size_t(order[size_t(q)]) + 1 + size_t(c)];
    return out;
}

namespace {
// fopen that fails loudly: a null FILE* in fprintf / fclose would crash (first CI run on the GPU workstation)
FILE* open_file(const std::string& path, const char* mode) {
    FILE* f = std::fopen(path.c_str(), mode);
    if (!f) throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
    return f;
}
const double e_si = 1.602176634e-19, me_si = 9.1093837015e-31, eps0_si = 8.8541878128e-12, c_si = 299792458.0;
const double mec2_mev = 0.51099895000;

class BeamDiag : public Operator {
public:
    explicit BeamDiag(const BeamDiagConfig& c) : c_(c) {}
    const char* name() const override { return "beam_diag"; }
    Stage stage() const override { return Stage::Diagnostics; }
    void setup(Simulation& s) override {
        for (size_t i = 0; i < s.species.size(); ++i)
            if (s.species[i].name == c_.species) idx_ = int(i);
        if (idx_ < 0) throw std::runtime_error("diagnostic " + c_.name + ": no species " + c_.species);
        const double k0 = 2 * M_PI / s.cfg.lambda0, w0 = c_si * k0;
        const double n_c = eps0_si * me_si * w0 * w0 / (e_si * e_si);
        qpc_ = s.species[idx_].q * e_si * n_c / (k0 * k0 * k0) * 1e12;   // pC per unit weight
        um_ = 1e6 / k0;
        fs_ = 1e15 / w0;
        mev_ = s.species[idx_].m * mec2_mev;                               // MeV per unit of gamma
        if (s.comm.rank == 0) {
            make_dirs(s.cfg.dir);
            // a restart continues the files (lines after the checkpoint's step are dropped, fsutil.hpp)
            bool fresh;
            beam_ = open_log(s.cfg.dir + "/beam_" + c_.name + ".txt", s.restart_step, fresh);
            if (fresh) std::fprintf(beam_, "# %s: species %s, gamma >= %g\n# step t[fs] N_macro Q[pC] E_mean[MeV] dE/E_rms "
                         "E_max[MeV] eps_ny[um] eps_nz[um] x_mean[um] sigma_x[um] sigma_y[um] sigma_z[um]\n",
                         c_.name.c_str(), c_.species.c_str(), c_.gamma_min);
            if (c_.bins > 0) {
                spec_ = open_log(s.cfg.dir + "/spectrum_" + c_.name + ".txt", s.restart_step, fresh);
                if (fresh) {
                    std::fprintf(spec_, "# dQ/dE [pC/MeV], species %s, gamma >= %g; columns: step t[fs], then bins at E[MeV] =",
                                 c_.species.c_str(), c_.gamma_min);
                    for (int b = 0; b < c_.bins; ++b) std::fprintf(spec_, " %.6g", (b + 0.5) * c_.e_max_mev / c_.bins);
                    std::fprintf(spec_, "\n");
                }
            }
        }
        if (c_.particles) {
            make_dirs(s.cfg.dir + "/particles");
            if (s.comm.rank == 0) {
                FILE* m = open_file(s.cfg.dir + "/particles/meta.txt", "w");
                std::fprintf(m, "columns x y z ux uy uz w (float64)\nlength_um %.10g\nweight_pC %.10g\n"
                             "charge %g\nmass %g\nlambda0_m %.6g\n", um_, qpc_, s.species[idx_].q,
                             s.species[idx_].m, s.cfg.lambda0);
                std::fclose(m);
                bool fresh;
                index_ = open_log(s.cfg.dir + "/particles/" + c_.name + ".txt", s.restart_step, fresh);
                if (fresh) std::fprintf(index_, "# step t[fs] count per rank\n");
            }
        }
    }
    void apply(Simulation& s) override {
        const long step = s.istep + 1;                 // the stage runs at the end of step istep -> istep + 1
        if (step % c_.every) return;
        const double t = step * s.dom.dt() * fs_;
        const Species& sp = s.species[size_t(idx_)];
        const BeamMoments m = beam_moments(sp, c_.gamma_min, s.comm);
        if (s.comm.rank == 0) {
            const double emean = m.wsum > 0 ? (m.gmean - 1) * mev_ : 0.0;   // 0 when nothing is selected
            std::fprintf(beam_, "%ld %.6g %.0f %.10g %.10g %.6g %.10g %.6g %.6g %.10g %.6g %.6g %.6g\n", step, t,
                         m.count, m.wsum * std::abs(qpc_), emean, emean > 0 ? m.gsig * mev_ / emean : 0.0,
                         m.count > 0 ? (m.gmax - 1) * mev_ : 0.0, m.eps_y * um_, m.eps_z * um_, m.xmean * um_,
                         m.sx * um_, m.sy * um_, m.sz * um_);
            std::fflush(beam_);
        }
        if (c_.bins > 0) {
            const std::vector<double> h = beam_spectrum(sp, c_.gamma_min, c_.bins, c_.e_max_mev / mec2_mev, s.comm);
            if (s.comm.rank == 0) {
                const double de = c_.e_max_mev / c_.bins;
                std::fprintf(spec_, "%ld %.6g", step, t);
                for (double v : h) std::fprintf(spec_, " %.6g", v * std::abs(qpc_) / de);
                std::fprintf(spec_, "\n");
                std::fflush(spec_);
            }
        }
        if (c_.particles) {
            const std::vector<double> v = beam_particles(sp, c_.gamma_min);
            char fn[64];
            std::snprintf(fn, sizeof fn, "/particles/%s_%06ld_r%d.bin", c_.name.c_str(), step, s.comm.rank);
            FILE* f = open_file(s.cfg.dir + fn, "wb");
            if (!v.empty()) std::fwrite(v.data(), sizeof(double), v.size(), f);
            std::fclose(f);
            std::vector<double> cnt(size_t(s.comm.size), 0.0);
            cnt[size_t(s.comm.rank)] = double(v.size() / 7);
            s.comm.allreduce_sum(cnt.data(), s.comm.size);
            if (s.comm.rank == 0) {
                std::fprintf(index_, "%ld %.6g", step, t);
                for (double q : cnt) std::fprintf(index_, " %.0f", q);
                std::fprintf(index_, "\n");
                std::fflush(index_);
            }
        }
    }
    ~BeamDiag() override {
        if (beam_) std::fclose(beam_);
        if (spec_) std::fclose(spec_);
        if (index_) std::fclose(index_);
    }
private:
    BeamDiagConfig c_;
    int idx_ = -1;
    double qpc_ = 0, um_ = 0, fs_ = 0, mev_ = 0;
    FILE *beam_ = nullptr, *spec_ = nullptr, *index_ = nullptr;
};
}  // namespace

std::unique_ptr<Operator> make_beam_diag_operator(const BeamDiagConfig& c) { return std::make_unique<BeamDiag>(c); }
