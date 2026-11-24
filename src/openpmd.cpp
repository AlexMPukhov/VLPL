#include "openpmd.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include "fsutil.hpp"
#include "h5.hpp"
#include "sim.hpp"
#include "stagger.hpp"

static std::string file_name(const char* prefix, long step) {
    char b[64];
    std::snprintf(b, sizeof b, "%s%06ld.h5", prefix, step);
    return b;
}

std::string latest_checkpoint(const std::string& dir) {
    long best = -1;
    for (const std::string& n : list_dir(dir + "/checkpoints")) {
        if (n.rfind("checkpoint_", 0) != 0 || n.size() < 15 || n.compare(n.size() - 3, 3, ".h5") != 0) continue;
        const std::string num = n.substr(11, n.size() - 14);
        if (num.empty() || num.find_first_not_of("0123456789") != std::string::npos) continue;
        best = std::max(best, std::atol(num.c_str()));
    }
    return best < 0 ? "" : dir + "/checkpoints/" + file_name("checkpoint_", best);
}

#ifndef RIP_HDF5
bool hdf5_available() { return false; }
static void no_hdf5() { throw std::runtime_error("openPMD output and checkpoints need a build with HDF5 (make HDF5=1)"); }
void write_openpmd(Simulation&) { no_hdf5(); }
std::string write_checkpoint(Simulation&) { no_hdf5(); return ""; }
void read_checkpoint(Simulation&, const std::string&) { no_hdf5(); }
bool read_op_state(const Simulation&, const std::string&, const std::vector<std::string>&, OpState&) { return false; }
#else
bool hdf5_available() { return true; }

namespace {
const double e_si = 1.602176634e-19, me_si = 9.1093837015e-31, eps0_si = 8.8541878128e-12, c_si = 299792458.0;
using Dim = std::vector<double>;                  // unitDimension: L M T I theta N J
const Dim dim_E{1, 1, -3, -1, 0, 0, 0}, dim_B{0, 1, -2, -1, 0, 0, 0}, dim_J{-2, 0, 0, 1, 0, 0, 0},
    dim_rho{-3, 0, 1, 1, 0, 0, 0}, dim_L{1, 0, 0, 0, 0, 0, 0}, dim_p{1, 1, -1, 0, 0, 0, 0},
    dim_q{0, 0, 1, 1, 0, 0, 0}, dim_m{0, 1, 0, 0, 0, 0, 0}, dim_0{0, 0, 0, 0, 0, 0, 0};

// SI value of the code units (README: lengths 1/k0, times 1/omega0, fields m c omega0 / e, densities n_c)
struct Units {
    double L, T, E, B, rho, J, w;
    explicit Units(double lambda0) {
        const double k0 = 2 * M_PI / lambda0, w0 = c_si * k0, nc = eps0_si * me_si * w0 * w0 / (e_si * e_si);
        L = 1 / k0; T = 1 / w0; E = me_si * c_si * w0 / e_si; B = me_si * w0 / e_si;
        rho = e_si * nc; J = e_si * nc * c_si; w = nc / (k0 * k0 * k0);    // real particles per unit weight
    }
};

// Position within the cell in openPMD order (z, y, x) of a staggering location.
std::vector<double> pos(const Loc& l) { return {l.o[2], l.o[1], l.o[0]}; }

using Sink = std::function<void(const long long* hdr, size_t byte_off, const unsigned char* data, size_t n)>;
const size_t CHUNK = size_t(1) << 28;             // bytes per message (MPI counts are int)

// Every rank passes a header (the same length everywhere) and a payload; rank 0 calls the sink for the
// ranks in order. whole: the sink gets each rank's payload in one piece, else in chunks of < CHUNK.
void to_root(Comm& c, const std::vector<long long>& hdr, const void* pay, size_t bytes, bool whole, const Sink& sink) {
    const unsigned char* p = static_cast<const unsigned char*>(pay);
    if (c.rank == 0) sink(hdr.data(), 0, p, bytes);
    std::vector<long long> h(hdr.size());
    std::vector<unsigned char> buf;
    for (int r = 1; r < c.size; ++r) {
        if (c.rank == r) {
            unsigned long long n = bytes;
            c.sendrecv(hdr.data(), hdr.size() * 8, 0, nullptr, 0, -1, 300);
            c.sendrecv(&n, 8, 0, nullptr, 0, -1, 301);
            for (size_t o = 0; o < bytes; o += CHUNK) c.sendrecv(p + o, std::min(CHUNK, bytes - o), 0, nullptr, 0, -1, 302);
        } else if (c.rank == 0) {
            unsigned long long n = 0;
            c.sendrecv(nullptr, 0, -1, h.data(), h.size() * 8, r, 300);
            c.sendrecv(nullptr, 0, -1, &n, 8, r, 301);
            buf.resize(whole ? size_t(n) : std::min<size_t>(size_t(n), CHUNK));
            for (size_t o = 0; o < n; o += CHUNK) {
                const size_t m = std::min(CHUNK, size_t(n) - o);
                c.sendrecv(nullptr, 0, -1, whole ? buf.data() + o : buf.data(), m, r, 302);
                if (!whole) sink(h.data(), o, buf.data(), m);
            }
            if (whole) sink(h.data(), 0, buf.data(), size_t(n));
        }
    }
}

template <class F> void with_type(Attr t, F f) {
    switch (t) {
    case Attr::F64: f(double()); break;
    case Attr::F32: f(float()); break;
    case Attr::I64: f((long long)0); break;
    case Attr::I32: f(int()); break;
    }
}
Attr attr_of(const std::string& tn) {
    if (tn == "f64") return Attr::F64;
    if (tn == "f32") return Attr::F32;
    if (tn == "i64") return Attr::I64;
    if (tn == "i32") return Attr::I32;
    throw std::runtime_error("checkpoint: particle record of unsupported type " + tn);
}

class Writer {
public:
    Writer(Simulation& s, bool chk) : s_(s), chk_(chk), single_(!chk && s.cfg.openpmd.single), u_(s.cfg.lambda0) {}
    void open(const std::string& path);
    void fields();
    void particles(const Species& sp);
    void state();
private:
    h5::Id component(hid_t parent, const std::string& name, const Field& f, int xlo);
    void raw_block(const std::string& name, const Buf<double>& b, int n2);   // O7 RZ atoms: an owned array [n2][j][i]
    void mesh_attrs(hid_t rec, const Dim& dim, double toff, int xlo);
    void vector_mesh(const std::string& name, const Field* f[3], const Loc loc[3], double unit, const Dim& dim, double toff, int xlo = 0);
    void scalar_mesh(const std::string& name, const Field& f, const Loc& loc, double unit, const Dim& dim, double toff, int xlo = 0);
    // RZ (thetaMode, docs/specs/rz_modes.md): components r, t, z; pos = (r, z) within the cell
    void rz_fields();
    void rz_attrs(hid_t rec, const Dim& dim, double toff);
    // axis: 0 as stored; 1 the m >= 1 values on the axis node zero (E_z, J_z, rho); 2 the m = 1 value on the axis node the regular one,
    // i s G_1(dr/2) of the partner half-node component G (E_phi from E_r, s = 1; B_r from B_phi, s = -1; J_phi from J_r, s = 1),
    // m >= 2 zero (output only, not the checkpoints)
    void rz_mesh(hid_t parent, const std::string& name, const Field& f, const double pos[2], double unit, int axis = 0,
                 const Field* partner = nullptr, double s = 1);
    void record_attrs(hid_t rec, const Dim& dim, double toff, double wpow, unsigned macro);
    void constant(hid_t parent, const std::string& name, double value, double unit, unsigned long long n);
    template <class T> void values(hid_t parent, const std::string& name, const Species& sp, int slot, double scale,
                                   const std::vector<long long>& off, unsigned long long total);
    Simulation& s_;
    bool chk_, single_;
    Units u_;
    h5::Id file_, it_, meshes_, parts_;           // rank 0
};

void Writer::open(const std::string& path) {
    if (s_.comm.rank != 0) return;
    file_ = h5::create_file(path);
    h5::attr(file_, "openPMD", "1.1.0");
    h5::attr(file_, "openPMDextension", unsigned(chk_ ? 0 : 1));          // 1: ED-PIC
    h5::attr(file_, "basePath", "/data/%T/");
    h5::attr(file_, "meshesPath", "meshes/");
    h5::attr(file_, "particlesPath", "particles/");
    h5::attr(file_, "iterationEncoding", "fileBased");
    h5::attr(file_, "iterationFormat", chk_ ? "checkpoint_%06T.h5" : "data_%06T.h5");
    h5::attr(file_, "software", "VLPL");
    h5::attr(file_, "softwareVersion", "1");
    const char* user = std::getenv("USER");
    h5::attr(file_, "author", user && *user ? user : "unknown");
    char date[64];
    const std::time_t now = std::time(nullptr);
    std::strftime(date, sizeof date, "%Y-%m-%d %H:%M:%S %z", std::localtime(&now));
    h5::attr(file_, "date", std::string(date));
    it_ = h5::group(file_, "data/" + std::to_string(s_.istep));
    h5::attr(it_, "time", s_.t);
    h5::attr(it_, "dt", s_.dom.dt());
    h5::attr(it_, "timeUnitSI", u_.T);
    meshes_ = h5::group(it_, "meshes");
    parts_ = h5::group(it_, "particles");
    if (chk_) return;
    const Config& c = s_.cfg;
    if (s_.dom.solver == Solver::Yee) h5::attr(meshes_, "fieldSolver", "Yee");
    else if (s_.dom.solver == Solver::IRip) {
        h5::attr(meshes_, "fieldSolver", "other");
        h5::attr(meshes_, "fieldSolverParameters", "iRIP, box closure (implicit RIP-CC with time-averaged Ex, Bx; c dt = dx)");
    } else {
        h5::attr(meshes_, "fieldSolver", "other");
        h5::attr(meshes_, "fieldSolverParameters", "RIP-CC (X-dispersionless, c dt = dx; Pukhov, JCP 418, 109622 (2020))");
    }
    std::vector<std::string> fb, pb;
    if (c.rz) {                                   // (r, z): the conducting wall absorbs particles
        fb = {"reflecting", "reflecting"};
        pb = {"absorbing", "absorbing"};
        for (int side = 0; side < 2; ++side) {
            fb.push_back(c.periodic[0] ? "periodic" : "open");
            pb.push_back(c.periodic[0] ? "periodic" : "absorbing");
        }
    }
    for (int a = 0; a < 3 && !c.rz; ++a)
        for (int side = 0; side < 2; ++side) {
            const bool per = c.periodic[a];
            fb.push_back(per ? "periodic" : (a == 0 ? "open" : "absorbing"));
            pb.push_back(per ? "periodic" : "absorbing");
        }
    h5::attr(meshes_, "fieldBoundary", fb);
    h5::attr(meshes_, "particleBoundary", pb);
    h5::attr(meshes_, "currentSmoothing", "none");
    h5::attr(meshes_, "chargeCorrection", "none");
}

// The owned block of f (with xlo cells below cell 0 along x at the lower x edge) to the global dataset name of rank 0.
// Ranks without the array (f.n(0) == 0: a PML memory away from the layers) send nothing; the file
// keeps HDF5's fill value 0 there. The cells below cell 0 of the other ranks are their left neighbour's last cells
// (for the CPML memories of the predictor and corrector an unused guard): they are not written.
h5::Id Writer::component(hid_t parent, const std::string& name, const Field& f, int xlo) {
    const Domain& d = s_.dom;
    const bool has = f.n(0) > 0;
    const int xl = d.off[0] == 0 ? xlo : 0;
    const long n0 = d.n[0] + xl, n1 = d.n[1], n2 = d.n[2];
    std::vector<double> blk;
    if (has) {
        const std::vector<double> hv = f.host();
        blk.resize(size_t(n0 * n1 * n2));
        for (int k = 0; k < n2; ++k)
            for (int j = 0; j < n1; ++j)
                for (int i = -xl; i < d.n[0]; ++i) blk[size_t((k * n1 + j) * n0 + i + xl)] = hv[f.idx(i, j, k)];
    }
    h5::Id ds;
    if (s_.comm.rank == 0) {
        const std::vector<hsize_t> dims{hsize_t(d.N[2]), hsize_t(d.N[1]), hsize_t(d.N[0] + xlo)};
        ds = single_ ? h5::dataset<float>(parent, name, dims) : h5::dataset<double>(parent, name, dims);
    }
    const std::vector<long long> hdr{has, d.off[2], d.off[1], d.off[0] + xlo - xl, n2, n1, n0};
    const hid_t id = ds.get();
    to_root(s_.comm, hdr, blk.data(), blk.size() * 8, true, [id](const long long* h, size_t, const unsigned char* p, size_t) {
        if (!h[0]) return;
        h5::write<double>(id, reinterpret_cast<const double*>(p), {hsize_t(h[1]), hsize_t(h[2]), hsize_t(h[3])},
                          {hsize_t(h[4]), hsize_t(h[5]), hsize_t(h[6])});
    });
    return ds;
}

// O7 RZ atoms (medium_rz_bound.cpp): the owned array [n2][j][i] of every rank into one dataset (n2, N_r, N_z)
void Writer::raw_block(const std::string& name, const Buf<double>& b, int n2) {
    const Domain& d = s_.dom;
    const std::vector<double> blk = b.host();
    h5::Id ds;
    if (s_.comm.rank == 0) ds = h5::dataset<double>(meshes_.get(), name, {hsize_t(n2), hsize_t(d.N[1]), hsize_t(d.N[0])});
    const std::vector<long long> hdr{1, 0, d.off[1], d.off[0], n2, d.n[1], d.n[0]};
    const hid_t id = ds.get();
    to_root(s_.comm, hdr, blk.data(), blk.size() * 8, true, [id](const long long* h, size_t, const unsigned char* p, size_t) {
        h5::write<double>(id, reinterpret_cast<const double*>(p), {hsize_t(h[1]), hsize_t(h[2]), hsize_t(h[3])},
                          {hsize_t(h[4]), hsize_t(h[5]), hsize_t(h[6])});
    });
}

void Writer::mesh_attrs(hid_t rec, const Dim& dim, double toff, int xlo) {
    const Domain& d = s_.dom;
    h5::attr(rec, "geometry", "cartesian");
    h5::attr(rec, "dataOrder", "C");
    h5::attr(rec, "axisLabels", std::vector<std::string>{"z", "y", "x"});
    h5::attr(rec, "gridSpacing", std::vector<double>{d.h[2], d.h[1], d.h[0]});
    h5::attr(rec, "gridGlobalOffset", std::vector<double>{0.0, 0.0, double(d.wx - xlo) * d.h[0]});   // absolute x
    h5::attr(rec, "gridUnitSI", u_.L);
    h5::attr(rec, "unitDimension", dim);
    h5::attr(rec, "timeOffset", toff);
    if (!chk_) h5::attr(rec, "fieldSmoothing", "none");
}

void Writer::vector_mesh(const std::string& name, const Field* f[3], const Loc loc[3], double unit, const Dim& dim,
                         double toff, int xlo) {
    h5::Id rec;
    if (s_.comm.rank == 0) { rec = h5::group(meshes_, name); mesh_attrs(rec, dim, toff, xlo); }
    const char* comp[3] = {"x", "y", "z"};
    for (int a = 0; a < 3; ++a) {
        h5::Id ds = component(rec.get(), comp[a], *f[a], xlo);
        if (s_.comm.rank == 0) { h5::attr(ds, "unitSI", unit); h5::attr(ds, "position", pos(loc[a])); }
    }
}

void Writer::scalar_mesh(const std::string& name, const Field& f, const Loc& loc, double unit, const Dim& dim,
                         double toff, int xlo) {
    h5::Id ds = component(meshes_.get(), name, f, xlo);
    if (s_.comm.rank != 0) return;
    mesh_attrs(ds, dim, toff, xlo);
    h5::attr(ds, "unitSI", unit);
    h5::attr(ds, "position", pos(loc));
}

// The PML memories in a checkpoint: name, array (12, pml.hpp)
std::vector<std::pair<std::string, Field*>> pml_arrays(Pml* p) {
    if (!p) return {};
    return {{"p_bz_y", &p->p_bz_y}, {"p_ez_y", &p->p_ez_y}, {"p_by_z", &p->p_by_z}, {"p_ey_z", &p->p_ey_z},
            {"t_bx_y", &p->t_bx_y}, {"t_ex_y", &p->t_ex_y}, {"t_bx_z", &p->t_bx_z}, {"t_ex_z", &p->t_ex_z},
            {"c_bz_y", &p->c_bz_y}, {"c_ez_y", &p->c_ez_y}, {"c_by_z", &p->c_by_z}, {"c_ey_z", &p->c_ey_z}};
}

// O7 media in a checkpoint (medium.hpp): the state of the linear media (J, P per component) and of the atoms (model = bound: the
// density matrix, their polarization bP and the last Et, the predictor of the outer iteration); raw arrays (RZ: every slot)
std::vector<std::pair<std::string, Field*>> medium_arrays(Medium* m) {
    std::vector<std::pair<std::string, Field*>> v;
    if (!m) return v;
    for (int c = 0; c < 3; ++c) { v.push_back({"J" + std::to_string(c), &m->J[c]}); v.push_back({"P" + std::to_string(c), &m->P[c]}); }
    // the coefficients too: in a moving window they belong to the shifted profile (setup builds them at the start position)
    for (int c = 0; c < 3; ++c) {
        v.push_back({"wp2" + std::to_string(c), &m->wp2[c]}); v.push_back({"w02" + std::to_string(c), &m->w02[c]});
        v.push_back({"nu" + std::to_string(c), &m->nu[c]});
    }
    if (m->bound) v.push_back({"bn", &m->bn});
    if (m->bound && m->rz) {                          // RZ: per mode the lagged atom current and the polarization (the atoms: raw_block)
        v.push_back({"bLr", &m->bLr}); v.push_back({"bLp", &m->bLp}); v.push_back({"bPr", &m->bPr}); v.push_back({"bPp", &m->bPp});
    } else if (m->bound) {
        for (int q = 0; q < BS_N; ++q) v.push_back({"rho" + std::to_string(q), &m->bs[q]});
        v.push_back({"bP", &m->bP}); v.push_back({"bE", &m->bE});
        if (m->bc.ion) v.push_back({"bacc", &m->bacc});
    }
    return v;
}

void Writer::fields() {
    if (s_.cfg.rz) { rz_fields(); return; }
    Fields& F = *s_.fields;
    const double dt = s_.dom.dt();
    const Field* E[3] = {&F.ex, &F.ey, &F.ez};
    const Field* B[3] = {&F.bx, &F.by, &F.bz};
    const Field* J[3] = {&F.jx, &F.jy, &F.jz};
    // the positions of the run's solver (stagger.hpp)
    const bool yee = s_.dom.solver == Solver::Yee;
    const Loc* LE = yee ? Yee::E : RipCC::E;
    const Loc* LB = yee ? Yee::B : RipCC::B;
    const Loc Lrho = yee ? Yee::rho : RipCC::rho;
    const Loc LJ[3] = {yee ? current_loc<Yee>(0) : current_loc<RipCC>(0), yee ? current_loc<Yee>(1) : current_loc<RipCC>(1),
                       yee ? current_loc<Yee>(2) : current_loc<RipCC>(2)};
    if (chk_) {
        vector_mesh("E", E, LE, u_.E, dim_E, 0);
        vector_mesh("B", B, LB, u_.B, dim_B, 0);
        // the initial Gauss residual (diagnostics), moved with the window
        scalar_mesh("rippic_gauss0", s_.gauss0(), Lrho, u_.rho, dim_rho, 0);
        // the CPML memories at the derivative locations; the transport ones also at i = -1 (rip_cc_step_pml)
        const Loc o{{0, 0, 0}};
        for (auto& pa : pml_arrays(s_.pml())) {
            const bool any = s_.comm.allreduce_max(pa.second->n(0) > 0 ? 1.0 : 0.0) > 0;
            if (any) scalar_mesh("rippic_pml_" + pa.first, *pa.second, o, 1.0, dim_0, 0, 1);
        }
        for (auto& ma : medium_arrays(s_.medium.get())) component(meshes_.get(), "vlpl_medium_" + ma.first, *ma.second, 0);
        // the cold fluid of a side tube in a background plasma: j^{n-1/2} at the points of J (side.hpp)
        const Field* Jf[3] = {&F.jfx, &F.jfy, &F.jfz};
        const char* jn[3] = {"rippic_fluid_jx", "rippic_fluid_jy", "rippic_fluid_jz"};
        if (F.jfx.n(0) > 0)
            for (int a = 0; a < 3; ++a) scalar_mesh(jn[a], *Jf[a], LJ[a], 1.0, dim_0, -0.5 * dt);
        return;
    }
    for (const std::string& n : s_.cfg.openpmd.fields) {
        if (n == "E") vector_mesh("E", E, LE, u_.E, dim_E, 0);
        else if (n == "B") vector_mesh("B", B, LB, u_.B, dim_B, 0);
        else if (n == "J") vector_mesh("J", J, LJ, u_.J, dim_J, -0.5 * dt);   // j^{n-1/2} of the last step
        else if (n == "rho") { s_.compute_rho(); scalar_mesh("rho", F.rho, Lrho, u_.rho, dim_rho, 0); }
        else if (n.rfind("rho_", 0) == 0) {
            size_t i = 0;
            while (i < s_.species.size() && s_.species[i].name != n.substr(4)) ++i;
            if (i == s_.species.size()) throw std::runtime_error("output.openpmd_fields: no species " + n.substr(4));
            Field r(s_.dom);
            deposit_rho(s_.species[i], r, s_.dom, s_.cfg.order);
            s_.comm.add_guards(s_.dom, {&r});
            scalar_mesh(n, r, Lrho, u_.rho, dim_rho, 0);
        } else throw std::runtime_error("output.openpmd_fields: unknown field " + n + " (E B J rho rho_<species>)");
    }
}

// thetaMode (openPMD 1.1): F(r, theta, z) = F_0 + sum_m [C_m cos(m theta) + S_m sin(m theta)], dataset
// (2M - 1, Nr, Nz) with C_m, S_m at 2m - 1, 2m. Ours F = F_0 + 2 Re sum F_m e^{i m theta}: C_m = 2 Re F_m,
// S_m = -2 Im F_m (the slots hold Re, Im of F_m).
void Writer::rz_attrs(hid_t rec, const Dim& dim, double toff) {
    const Domain& d = s_.dom;
    h5::attr(rec, "geometry", "thetaMode");
    h5::attr(rec, "geometryParameters", "m=" + std::to_string(s_.cfg.modes) + ";imag=+");
    h5::attr(rec, "dataOrder", "C");
    h5::attr(rec, "axisLabels", std::vector<std::string>{"r", "z"});
    h5::attr(rec, "gridSpacing", std::vector<double>{d.h[1], d.h[0]});
    h5::attr(rec, "gridGlobalOffset", std::vector<double>{0.0, double(d.wx) * d.h[0]});   // absolute z
    // a stretched radial grid (docs/specs/rz_modes.md step 6c): openPMD has no non-uniform mesh, so
    // gridSpacing holds the spacing next to the axis and the node radii r_0 .. r_nr (in gridUnitSI) are given
    // here; the half-node components sit at the midpoints (position 0.5)
    const RadialGrid& g = s_.rz->grid();
    if (!g.uniform) h5::attr(rec, "rippic_r_nodes", std::vector<double>(g.r.begin(), g.r.begin() + g.nr + 1));
    h5::attr(rec, "gridUnitSI", u_.L);
    h5::attr(rec, "unitDimension", dim);
    h5::attr(rec, "timeOffset", toff);
    h5::attr(rec, "fieldSmoothing", "none");
}

void Writer::rz_mesh(hid_t parent, const std::string& name, const Field& f, const double pos[2], double unit, int axis,
                     const Field* partner, double s) {
    const Domain& d = s_.dom;
    Field c(d);
    std::vector<double> h = f.host();
    if (axis > 0 && d.n[2] > 1) {                  // the physical axis values of the modes m >= 1 (output only)
        std::vector<double> g;
        if (axis == 2 && partner) g = partner->host();
        for (int i = 0; i < d.n[0]; ++i)
            for (int k = 1; k < d.n[2]; ++k) {
                double v = 0;
                if (axis == 2 && partner && k <= 2)      // i s G_1(dr/2): Re -> -s Im G, Im -> s Re G
                    v = k == 1 ? -s * g[partner->idx(i, 0, 2)] : s * g[partner->idx(i, 0, 1)];
                h[f.idx(i, 0, k)] = v;
            }
    }
    for (int k = 1; k < d.n[2]; ++k)
        for (int j = 0; j < d.n[1]; ++j)
            for (int i = 0; i < d.n[0]; ++i) h[f.idx(i, j, k)] *= k % 2 ? 2.0 : -2.0;
    c.upload(h);
    h5::Id ds = component(parent, name, c, 0);
    if (s_.comm.rank != 0) return;
    h5::attr(ds, "unitSI", unit);
    h5::attr(ds, "position", std::vector<double>{pos[0], pos[1]});
}

// The staggering of src/rz.hpp in (r, z): Ez, Ephi, Br, Jz at (r_j, z_i); Er, Bphi, Bz at r_{j+1/2};
// rho, Jphi at z_{i+1/2}; Jr at (r_{j+1/2}, z_{i+1/2}).
void Writer::rz_fields() {
    FieldsRZ& F = *s_.rz;
    const double dt = s_.dom.dt();
    // axis: for the output (not the checkpoints) the physical axis values of m >= 1: component t (E_phi, B_phi, J_phi) and r (B_r) on the
    // axis node from the partner, z (E_z, J_z) zero; ax[3] per component: 0 as stored, 1 zero, 2 regular from the partner with sign s
    auto vec = [&](const char* name, const Field* f[3], const double pos[3][2], double unit, const Dim& dim, double toff,
                   const int* ax = nullptr, const Field* part = nullptr, double s = 1) {
        h5::Id rec;
        if (s_.comm.rank == 0) { rec = h5::group(meshes_, name); rz_attrs(rec, dim, toff); }
        const char* comp[3] = {"r", "t", "z"};
        for (int a = 0; a < 3; ++a) rz_mesh(rec.get(), comp[a], *f[a], pos[a], unit, ax ? ax[a] : 0, part, s);
    };
    auto scal = [&](const std::string& name, const Field& f) {
        const double pr[2] = {0.0, 0.5};
        rz_mesh(meshes_.get(), name, f, pr, u_.rho, 1);   // m >= 1 zero on the axis
        if (s_.comm.rank == 0) rz_attrs(h5::open_dataset(meshes_, name), dim_rho, 0);
    };
    const Field* E[3] = {&F.er, &F.ep, &F.ez};
    const Field* B[3] = {&F.br, &F.bp, &F.bz};
    const double pE[3][2] = {{0.5, 0}, {0, 0}, {0, 0}}, pB[3][2] = {{0, 0}, {0.5, 0}, {0.5, 0}};
    if (chk_) {
        // E, B as in the output (float64; the factors +-2 are exact), the initial Gauss residual per slot
        vec("E", E, pE, u_.E, dim_E, 0);
        vec("B", B, pB, u_.B, dim_B, 0);
        component(meshes_.get(), "rippic_gauss0", s_.gauss0(), 0);
        for (auto& ma : medium_arrays(s_.medium.get())) component(meshes_.get(), "vlpl_medium_" + ma.first, *ma.second, 0);
        if (s_.medium && s_.medium->bound)            // the atoms: the density matrix, the last Et, the ionization accumulator
            for (int q = 0; q < RA_N; ++q)
                if (q < BS_N || q == RA_E || q == RA_ACC)
                    raw_block("vlpl_medium_atoms_" + std::to_string(q), s_.medium->ra[size_t(q)], s_.medium->nphi);
        return;
    }
    for (const std::string& n : s_.cfg.openpmd.fields) {
        if (n == "E") {
            const Field** f = E;
            const int ax[3] = {0, 2, 1};                  // E_phi1(0) = i E_r1(dr/2), E_z of m >= 1 zero on the axis
            vec("E", f, pE, u_.E, dim_E, 0, ax, &F.er, 1.0);
        } else if (n == "B") {
            const int ax[3] = {2, 0, 0};                  // B_r1(0) = -i B_phi1(dr/2); B_phi, B_z: half nodes
            vec("B", B, pB, u_.B, dim_B, 0, ax, &F.bp, -1.0);
        } else if (n == "J") {
            const Field* f[3] = {&F.jr, &F.jp, &F.jz};
            const double p[3][2] = {{0.5, 0.5}, {0, 0.5}, {0, 0}};
            const int ax[3] = {0, 2, 1};                  // J_phi1(0) = i J_r1(dr/2), J_z of m >= 1 zero on the axis
            vec("J", f, p, u_.J, dim_J, -0.5 * dt, ax, &F.jr, 1.0);   // j^{n-1/2} of the last step
        } else if (n == "rho") {
            Field r(s_.dom);
            s_.rz_rho_all(r);
            scal("rho", r);
        } else if (n.rfind("rho_", 0) == 0) {
            size_t i = 0;
            while (i < s_.species.size() && s_.species[i].name != n.substr(4)) ++i;
            if (i == s_.species.size()) throw std::runtime_error("output.openpmd_fields: no species " + n.substr(4));
            Field r(s_.dom);
            rz_rho(s_.species[i], F, r, s_.cfg.order);
            s_.comm.add_guards(s_.dom, {&r});
            rz_smooth_rho(F, s_.comm, r);
            rz_filter_rho(F, s_.comm, r, s_.cfg.filter_passes);
            scal(n, r);
        } else throw std::runtime_error("output.openpmd_fields: unknown field " + n + " (E B J rho rho_<species>)");
    }
}

void Writer::record_attrs(hid_t rec, const Dim& dim, double toff, double wpow, unsigned macro) {
    h5::attr(rec, "unitDimension", dim);
    h5::attr(rec, "timeOffset", toff);
    if (chk_) return;
    h5::attr(rec, "weightingPower", wpow);
    h5::attr(rec, "macroWeighted", macro);
}

void Writer::constant(hid_t parent, const std::string& name, double value, double unit, unsigned long long n) {
    h5::Id g = h5::group(parent, name);
    h5::attr(g, "value", value);
    h5::attr(g, "shape", std::vector<unsigned long long>{n});
    h5::attr(g, "unitSI", unit);
}

// One attribute of all particles: dataset name (float32 in single-precision output) at the offsets
// of the ranks; scale != 1 converts the values (weighting of the output).
template <class T>
void Writer::values(hid_t parent, const std::string& name, const Species& sp, int slot, double scale,
                    const std::vector<long long>& off, unsigned long long total) {
    std::vector<T> v = sp.get_as<T>(slot);
    if (scale != 1)
        for (T& x : v) x = T(x * scale);
    h5::Id ds;
    if (s_.comm.rank == 0) {
        const std::vector<hsize_t> dims{hsize_t(total)};
        ds = single_ && std::is_same<T, double>::value ? h5::dataset<float>(parent, name, dims) : h5::dataset<T>(parent, name, dims);
    }
    const hid_t id = ds.get();
    const std::vector<long long> hdr{s_.comm.rank};
    to_root(s_.comm, hdr, v.data(), v.size() * sizeof(T), false, [id, &off](const long long* h, size_t bo, const unsigned char* p, size_t n) {
        h5::write<T>(id, reinterpret_cast<const T*>(p), {hsize_t(off[size_t(h[0])]) + bo / sizeof(T)}, {n / sizeof(T)});
    });
    if (s_.comm.rank == 0) h5::attr(ds, "unitSI", 1.0);
}

void Writer::particles(const Species& sp) {
    Comm& c = s_.comm;
    const Domain& d = s_.dom;
    const int R = c.size;
    // counts and boxes of the ranks (particle patches)
    std::vector<double> cnt(size_t(R) * 7, 0.0);
    cnt[size_t(c.rank) * 7] = double(sp.size());
    // RZ: particle axes (z, x, y) in X, Y, Z; the openPMD components x, y, z are Y, Z, X; the box of a
    // rank is its z slab of the cylinder
    const bool rz = s_.cfg.rz;
    const double Rw = rz ? s_.rz->grid().R() : 0.0;
    auto slot_of = [rz](int a) { return rz ? (a + 1) % 3 : a; };
    for (int a = 0; a < 3; ++a) {
        const bool zax = rz && a == 2;
        cnt[size_t(c.rank) * 7 + 1 + a] = rz ? (zax ? d.lo(0) : -Rw) : d.lo(a);
        cnt[size_t(c.rank) * 7 + 4 + a] = rz ? (zax ? d.hi(0) - d.lo(0) : 2 * Rw) : d.hi(a) - d.lo(a);
    }
    c.allreduce_sum(cnt.data(), int(cnt.size()));
    std::vector<long long> off(size_t(R) + 1, 0);
    for (int r = 0; r < R; ++r) off[size_t(r) + 1] = off[size_t(r)] + (long long)(cnt[size_t(r) * 7]);
    const unsigned long long total = (unsigned long long)off[size_t(R)];
    h5::Id g, pg, mg, wg;
    const char* comp[3] = {"x", "y", "z"};
    const double dt = d.dt();
    if (c.rank == 0) {
        g = h5::group(parts_, sp.name);
        if (!chk_) {
            h5::attr(g, "particleShape", double(s_.cfg.order));
            h5::attr(g, "currentDeposition", "Esirkepov");
            h5::attr(g, "particlePush", s_.cfg.pusher == Pusher::Boris ? "Boris" : "HigueraCary");
            h5::attr(g, "particleInterpolation", "energyConserving");
            h5::attr(g, "particleSmoothing", "none");
        }
        pg = h5::group(g, "position"); record_attrs(pg, dim_L, 0, 0, 0);
        mg = h5::group(g, "momentum"); record_attrs(mg, dim_p, -0.5 * dt, 1, 0);      // u^{n-1/2} (particles.hpp)
        h5::Id og = h5::group(g, "positionOffset");
        record_attrs(og, dim_L, 0, 0, 0);
        for (int a = 0; a < 3; ++a) constant(og, comp[a], 0.0, u_.L, total);
        constant(g, "charge", sp.q, e_si, total);
        constant(g, "mass", sp.m, me_si, total);
        for (const char* n : {"charge", "mass"}) {
            h5::Id r = h5::open_group(g, n);
            record_attrs(r, std::string(n) == "charge" ? dim_q : dim_m, 0, 1, 0);
        }
        // patches: one per rank (its box and its particles), used by the restart
        h5::Id pp = h5::group(g, "particlePatches");
        std::vector<unsigned long long> np(static_cast<size_t>(R)), no(static_cast<size_t>(R));
        for (int r = 0; r < R; ++r) { np[size_t(r)] = (unsigned long long)off[size_t(r) + 1] - off[size_t(r)]; no[size_t(r)] = off[size_t(r)]; }
        auto counts = [&](const char* name, const std::vector<unsigned long long>& v) {
            h5::Id ds = h5::dataset<unsigned long long>(pp, name, {hsize_t(R)});
            h5::write(ds.get(), v.data(), {0}, {hsize_t(R)});
            h5::attr(ds, "unitSI", 1.0);
        };
        counts("numParticles", np);
        counts("numParticlesOffset", no);
        for (int e = 0; e < 2; ++e) {
            h5::Id rg = h5::group(pp, e ? "extent" : "offset");
            h5::attr(rg, "unitDimension", dim_L);
            for (int a = 0; a < 3; ++a) {
                std::vector<double> v(static_cast<size_t>(R));
                for (int r = 0; r < R; ++r) v[size_t(r)] = cnt[size_t(r) * 7 + 1 + 3 * e + a];
                h5::Id ds = h5::dataset<double>(rg, comp[a], {hsize_t(R)});
                h5::write(ds.get(), v.data(), {0}, {hsize_t(R)});
                h5::attr(ds, "unitSI", u_.L);
            }
        }
    }
    const double mom_unit = sp.m * me_si * c_si;
    for (int a = 0; a < 3; ++a) {
        values<double>(pg.get(), comp[a], sp, Species::X + slot_of(a), 1, off, total);
        if (c.rank == 0) h5::attr(h5::open_dataset(pg, comp[a]), "unitSI", u_.L);
        values<double>(mg.get(), comp[a], sp, Species::UX + slot_of(a), 1, off, total);
        if (c.rank == 0) h5::attr(h5::open_dataset(mg, comp[a]), "unitSI", mom_unit);
    }
    // weighting: real particles per macro-particle (ED-PIC: unitSI 1); the checkpoint keeps the raw weight
    values<double>(g.get(), "weighting", sp, Species::W, chk_ ? 1 : u_.w, off, total);
    if (c.rank == 0) {
        h5::Id w = h5::open_dataset(g, "weighting");
        record_attrs(w, dim_0, 0, 1, 1);
        h5::attr(w, "unitSI", chk_ ? u_.w : 1.0);
    }
    // further registered attributes, one scalar record each in their own type
    for (int sl = Species::NC; sl < sp.n_attr(); ++sl) {
        const AttrInfo& ai = sp.info(sl);
        with_type(ai.type, [&](auto z) { values<decltype(z)>(g.get(), ai.name, sp, sl, 1, off, total); });
        if (c.rank == 0) record_attrs(h5::open_dataset(g, ai.name), dim_0, 0, 0, 0);
    }
}

// Checkpoint only: the state outside fields and particles (group rippic of the iteration).
void Writer::state() {
    Comm& c = s_.comm;
    const Domain& d = s_.dom;
    const int R = c.size;
    std::vector<double> box(size_t(R) * 6, 0.0);
    for (int a = 0; a < 3; ++a) { box[size_t(c.rank) * 6 + a] = d.off[a]; box[size_t(c.rank) * 6 + 3 + a] = d.n[a]; }
    c.allreduce_sum(box.data(), int(box.size()));
    double emitted = s_.medium ? s_.medium->emitted : 0.0;    // O7 ionization: the weight emitted as pairs, summed over the ranks
    c.allreduce_sum(&emitted, 1);
    h5::Id g;
    if (c.rank == 0) {
        g = h5::group(it_, "rippic");
        h5::attr(g, "istep", (long long)s_.istep);
        h5::attr(g, "t", s_.t);
        h5::attr(g, "wx", (long long)d.wx);
        h5::attr(g, "N", std::vector<int>{d.N[0], d.N[1], d.N[2]});
        h5::attr(g, "h", std::vector<double>{d.h[0], d.h[1], d.h[2]});
        h5::attr(g, "shape", s_.cfg.order);
        h5::attr(g, "geometry", s_.cfg.rz ? "rz" : "cartesian");
        if (s_.cfg.rz) {                               // the radial grid (a restart must use the same)
            const RadialGrid& rg = s_.rz->grid();
            h5::attr(g, "r_nodes", std::vector<double>(rg.r.begin(), rg.r.begin() + rg.nr + 1));
        }
        h5::attr(g, "ranks", R);
        if (s_.medium) {                               // O7: the dissipated energy and the solver statistics (medium.txt continues)
            const MediumStats& ms = s_.medium->stats;
            h5::attr(g, "medium_dissipated", s_.medium->dissipated);
            h5::attr(g, "medium_ionized", s_.medium->ionized);
            h5::attr(g, "medium_emitted", emitted);
            h5::attr(g, "medium_stats", std::vector<long long>{ms.solves, ms.iterations, ms.max_iterations, ms.unconverged, ms.newton,
                                                              ms.newton_steps, ms.newton_max, ms.newton_unconverged});
        }
        std::vector<std::string> names;
        std::vector<unsigned char> loaded;
        for (size_t i = 0; i < s_.species.size(); ++i) { names.push_back(s_.species[i].name); loaded.push_back((unsigned char)s_.loaded[i]); }
        if (!names.empty()) { h5::attr(g, "species", names); h5::attr(g, "loaded", loaded); }
        std::vector<long long> b(box.begin(), box.end());
        h5::Id ds = h5::dataset<long long>(g, "rank_box", {hsize_t(R), 6});
        h5::write(ds.get(), b.data(), {0, 0}, {hsize_t(R), 6});
    }
    // the tile starts of the last sort (the tiled and box deposits use them until the next change)
    for (const Species& sp : s_.species) {
        std::vector<double> n(size_t(R), 0.0);
        n[size_t(c.rank)] = double(sp.tile_start.size());
        c.allreduce_sum(n.data(), R);
        std::vector<long long> off(size_t(R) + 1, 0);
        for (int r = 0; r < R; ++r) off[size_t(r) + 1] = off[size_t(r)] + (long long)n[size_t(r)];
        h5::Id ds;
        if (c.rank == 0) {
            ds = h5::dataset<int>(g, "tile_start/" + sp.name, {hsize_t(std::max<long long>(off[size_t(R)], 1))});
            h5::Id o = h5::dataset<long long>(g, "tile_start_offset/" + sp.name, {hsize_t(R) + 1});
            h5::write(o.get(), off.data(), {0}, {hsize_t(R) + 1});
        }
        const hid_t id = ds.get();
        to_root(c, {c.rank}, sp.tile_start.data(), sp.tile_start.size() * 4, false,
                [id, &off](const long long* h, size_t bo, const unsigned char* p, size_t m) {
                    h5::write<int>(id, reinterpret_cast<const int*>(p), {hsize_t(off[size_t(h[0])]) + bo / 4}, {m / 4});
                });
    }
    // the state of the operators (Operator::save, decision 116; e.g. the partial lab snapshots of a boosted run): one group
    // op_<name> per operator, one float64 dataset per non-empty array (an empty array is left out: read back empty)
    for (const auto& os : s_.pipeline.save(s_)) {
        if (c.rank != 0) continue;
        h5::Id og = h5::group(g, "op_" + os.first);
        for (const auto& a : os.second.arrays) {
            if (a.second.empty()) continue;
            h5::Id ds = h5::dataset<double>(og, a.first, {hsize_t(a.second.size())});
            h5::write<double>(ds.get(), a.second.data(), {0}, {hsize_t(a.second.size())});
        }
    }
}

// Writes the file under a temporary name and renames it when complete (rank 0).
std::string write_file(Simulation& s, bool chk) {
    const std::string dir = s.cfg.dir + (chk ? "/checkpoints" : "/openpmd");
    const std::string path = dir + "/" + file_name(chk ? "checkpoint_" : "data_", s.istep), tmp = path + ".tmp";
    if (s.comm.rank == 0) make_dirs(dir);
    {
        Writer w(s, chk);
        w.open(tmp);
        w.fields();
        const OpenPmdConfig& o = s.cfg.openpmd;
        for (size_t i = 0; i < s.species.size(); ++i) {
            const bool sel = chk || std::find(o.species.begin(), o.species.end(), "all") != o.species.end() ||
                             std::find(o.species.begin(), o.species.end(), s.species[i].name) != o.species.end();
            if (sel) w.particles(s.species[i]);
        }
        if (chk) w.state();
    }                                                  // closes the file
    if (s.comm.rank == 0 && std::rename(tmp.c_str(), path.c_str()) != 0)
        throw std::runtime_error("cannot rename " + tmp + " to " + path);
    s.comm.barrier();
    return path;
}

// ---------------------------------------------------------------- restart
void read_component(hid_t ds, Field& f, const Domain& d, int xlo) {
    if (f.n(0) == 0) return;
    const std::vector<hsize_t> dims = h5::dims(ds);
    if (dims.size() != 3 || dims[0] != hsize_t(d.N[2]) || dims[1] != hsize_t(d.N[1]) || dims[2] != hsize_t(d.N[0] + xlo))
        throw std::runtime_error("checkpoint: a mesh of another grid size");
    const long n0 = d.n[0] + xlo, n1 = d.n[1], n2 = d.n[2];
    std::vector<double> blk(static_cast<size_t>(n0 * n1 * n2));
    h5::read<double>(ds, blk.data(), {hsize_t(d.off[2]), hsize_t(d.off[1]), hsize_t(d.off[0])},
                     {hsize_t(n2), hsize_t(n1), hsize_t(n0)});
    std::vector<double> hv = f.host();
    for (int k = 0; k < n2; ++k)
        for (int j = 0; j < n1; ++j)
            for (int i = -xlo; i < d.n[0]; ++i) hv[f.idx(i, j, k)] = blk[size_t((k * n1 + j) * n0 + i + xlo)];
    f.upload(hv);
}

template <class T> std::vector<T> read_1d(hid_t g, const std::string& name, unsigned long long off, unsigned long long n) {
    std::vector<T> v(static_cast<size_t>(n));
    h5::Id ds = h5::open_dataset(g, name);
    h5::read<T>(ds, v.data(), {hsize_t(off)}, {hsize_t(n)});
    return v;
}

// Datasets of one species: (record path, slot) of every attribute of sp, after registering the extra ones.
// RZ: the components x, y, z are the slots Y, Z, X (Writer::particles).
std::vector<std::pair<std::string, int>> species_records(hid_t g, Species& sp, bool rz) {
    std::vector<std::pair<std::string, int>> v;
    const char* comp[3] = {"x", "y", "z"};
    for (int a = 0; a < 3; ++a) v.push_back({std::string("position/") + comp[a], Species::X + (rz ? (a + 1) % 3 : a)});
    for (int a = 0; a < 3; ++a) v.push_back({std::string("momentum/") + comp[a], Species::UX + (rz ? (a + 1) % 3 : a)});
    v.push_back({"weighting", Species::W});
    for (const std::string& n : h5::children(g)) {
        if (n == "position" || n == "momentum" || n == "weighting" || n == "positionOffset" || n == "charge" ||
            n == "mass" || n == "particlePatches")
            continue;
        h5::Id ds = h5::open_dataset(g, n);
        v.push_back({n, sp.require(n, attr_of(h5::type_name(ds)))});
    }
    return v;
}
void read_fields(Simulation& s, hid_t me) {
    const Domain& d = s.dom;
    Fields& F = *s.fields;
    const char* comp[3] = {"x", "y", "z"};
    Field* E[3] = {&F.ex, &F.ey, &F.ez};
    Field* B[3] = {&F.bx, &F.by, &F.bz};
    for (int a = 0; a < 3; ++a) {
        read_component(h5::open_dataset(me, std::string("E/") + comp[a]), *E[a], d, 0);
        read_component(h5::open_dataset(me, std::string("B/") + comp[a]), *B[a], d, 0);
    }
    s.comm.exchange(d, {&F.ex, &F.ey, &F.ez, &F.bx, &F.by, &F.bz});
    read_component(h5::open_dataset(me, "rippic_gauss0"), s.gauss0(), d, 0);
    for (auto& pa : pml_arrays(s.pml())) {
        const std::string n = "rippic_pml_" + pa.first;
        if (h5::exists(me, n)) read_component(h5::open_dataset(me, n), *pa.second, d, 1);
    }
    if (F.jfx.n(0) > 0) {                              // the cold fluid of a side tube (side.hpp)
        Field* Jf[3] = {&F.jfx, &F.jfy, &F.jfz};
        const char* jn[3] = {"rippic_fluid_jx", "rippic_fluid_jy", "rippic_fluid_jz"};
        for (int a = 0; a < 3; ++a) {
            if (!h5::exists(me, jn[a])) throw std::runtime_error("checkpoint: no fluid current of side.background_density");
            read_component(h5::open_dataset(me, jn[a]), *Jf[a], d, 0);
        }
        s.comm.exchange(d, {&F.jfx, &F.jfy, &F.jfz});
    }
}

// RZ: the thetaMode coefficients back to the slots (C_m = 2 Re F_m, S_m = -2 Im F_m: exact)
void read_rz_fields(Simulation& s, hid_t me) {
    const Domain& d = s.dom;
    FieldsRZ& F = *s.rz;
    const char* comp[3] = {"r", "t", "z"};
    Field* E[3] = {&F.er, &F.ep, &F.ez};
    Field* B[3] = {&F.br, &F.bp, &F.bz};
    auto read = [&](const std::string& name, Field& f) {
        read_component(h5::open_dataset(me, name), f, d, 0);
        std::vector<double> h = f.host();
        for (int k = 1; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) h[f.idx(i, j, k)] *= k % 2 ? 0.5 : -0.5;
        f.upload(h);
    };
    for (int a = 0; a < 3; ++a) {
        read(std::string("E/") + comp[a], *E[a]);
        read(std::string("B/") + comp[a], *B[a]);
    }
    s.comm.exchange(d, {&F.ez, &F.er, &F.ep, &F.bz, &F.br, &F.bp});
    read_component(h5::open_dataset(me, "rippic_gauss0"), s.gauss0(), d, 0);
}
}  // namespace

void write_openpmd(Simulation& s) { write_file(s, false); }

std::string write_checkpoint(Simulation& s) {
    const std::string path = write_file(s, true);
    const int keep = s.cfg.checkpoint.keep;
    if (s.comm.rank == 0 && keep > 0) {                // delete all but the newest `keep`
        std::vector<std::string> v;
        for (const std::string& n : list_dir(s.cfg.dir + "/checkpoints"))
            if (n.rfind("checkpoint_", 0) == 0 && n.size() > 3 && n.compare(n.size() - 3, 3, ".h5") == 0) v.push_back(n);
        std::sort(v.begin(), v.end());                 // zero-padded step numbers
        for (size_t i = 0; i + size_t(keep) < v.size(); ++i) std::remove((s.cfg.dir + "/checkpoints/" + v[i]).c_str());
    }
    return path;
}

void read_checkpoint(Simulation& s, const std::string& path) {
    Comm& c = s.comm;
    Domain& d = s.dom;
    h5::Id f = h5::open_file(path);
    h5::Id data = h5::open_group(f, "data");
    const std::vector<std::string> its = h5::children(data);
    if (its.size() != 1) throw std::runtime_error("checkpoint " + path + ": expected one iteration");
    h5::Id it = h5::open_group(data, its[0]);
    if (!h5::exists(it, "rippic")) throw std::runtime_error(path + " is no VLPL checkpoint (an output file?)");
    h5::Id st = h5::open_group(it, "rippic");
    const std::vector<int> N = h5::attr_vec<int>(st, "N");
    const std::vector<double> hh = h5::attr_vec<double>(st, "h");
    for (int a = 0; a < 3; ++a)
        if (N.at(size_t(a)) != d.N[a] || hh.at(size_t(a)) != d.h[a])
            throw std::runtime_error("checkpoint " + path + ": another grid (grid.n*, grid.d*, lambda0 must not change)");
    if (h5::attr_num<int>(st, "shape") != s.cfg.order) throw std::runtime_error("checkpoint: another numerics.shape");
    const std::string geo = h5::has_attr(st, "geometry") ? h5::attr_string(st, "geometry") : "cartesian";
    if (geo != (s.cfg.rz ? "rz" : "cartesian")) throw std::runtime_error("checkpoint: another grid.geometry (" + geo + ")");
    if (s.cfg.rz && h5::has_attr(st, "r_nodes")) {
        const std::vector<double> rn = h5::attr_vec<double>(st, "r_nodes");
        const RadialGrid& rg = s.rz->grid();
        if (rn != std::vector<double>(rg.r.begin(), rg.r.begin() + rg.nr + 1))
            throw std::runtime_error("checkpoint: another radial grid (grid.r_uniform, stretch, dr_max must not change)");
    }
    std::vector<std::string> names;
    std::vector<unsigned char> loaded;
    if (h5::has_attr(st, "species")) {
        names = h5::attr_strings(st, "species");
        loaded = h5::attr_vec<unsigned char>(st, "loaded");
    }
    if (names.size() != s.species.size())
        throw std::runtime_error("checkpoint: another number of species");
    for (size_t i = 0; i < names.size(); ++i)
        if (names[i] != s.species[i].name) throw std::runtime_error("checkpoint: species " + names[i] + " in place of " + s.species[i].name);
    s.istep = long(h5::attr_num<long long>(st, "istep"));
    s.t = h5::attr_num<double>(st, "t");
    d.wx = long(h5::attr_num<long long>(st, "wx"));
    for (size_t i = 0; i < loaded.size(); ++i) s.loaded[i] = char(loaded[i]);

    // fields: the owned block, guards from the exchange (as at the end of a step)
    h5::Id me = h5::open_group(it, "meshes");
    if (s.cfg.rz) read_rz_fields(s, me);
    else read_fields(s, me);
    if (s.medium) {                                   // O7: the media state (medium_arrays), the dissipated energy, the statistics
        Medium& M = *s.medium;
        std::vector<Field*> ex;
        for (auto& ma : medium_arrays(&M)) {
            const std::string n = "vlpl_medium_" + ma.first;
            if (!h5::exists(me, n)) throw std::runtime_error("checkpoint: no " + n + " (the media of the deck differ from the run's)");
            read_component(h5::open_dataset(me, n), *ma.second, d, 0);
            ex.push_back(ma.second);
        }
        if (M.rz && M.bound)                              // the RZ atoms (Writer::raw_block)
            for (int q = 0; q < RA_N; ++q) {
                if (!(q < BS_N || q == RA_E || q == RA_ACC)) continue;
                const std::string n = "vlpl_medium_atoms_" + std::to_string(q);
                if (!h5::exists(me, n)) throw std::runtime_error("checkpoint: no " + n);
                std::vector<double> v(size_t(M.nphi) * d.n[1] * d.n[0]);
                h5::Id ds = h5::open_dataset(me, n);
                if (h5::dims(ds)[0] != hsize_t(M.nphi)) throw std::runtime_error("checkpoint: another n_phi of the atoms");
                h5::read<double>(ds, v.data(), {0, hsize_t(d.off[1]), hsize_t(d.off[0])}, {hsize_t(M.nphi), hsize_t(d.n[1]), hsize_t(d.n[0])});
                M.ra[size_t(q)].from_host(v.data(), 0, v.size());
            }
        s.comm.exchange(d, ex);
        if (!h5::has_attr(st, "medium_dissipated")) throw std::runtime_error("checkpoint: written without media");
        M.dissipated = h5::attr_num<double>(st, "medium_dissipated");
        if (h5::has_attr(st, "medium_ionized")) M.ionized = h5::attr_num<double>(st, "medium_ionized");
        if (h5::has_attr(st, "medium_emitted")) M.emitted = s.comm.rank == 0 ? h5::attr_num<double>(st, "medium_emitted") : 0.0;
        const std::vector<long long> v = h5::attr_vec<long long>(st, "medium_stats");
        MediumStats& ms = M.stats;
        if (v.size() == 8) {
            ms.solves = v[0]; ms.iterations = v[1]; ms.max_iterations = v[2]; ms.unconverged = v[3];
            ms.newton = v[4]; ms.newton_steps = v[5]; ms.newton_max = v[6]; ms.newton_unconverged = v[7];
        }
    }

    // particles: this rank's patch if the decomposition is the same, else every particle in its box
    const int R = h5::attr_num<int>(st, "ranks");
    bool same = R == c.size;
    if (same) {
        h5::Id ds = h5::open_dataset(st, "rank_box");
        std::vector<long long> b(size_t(R) * 6);
        h5::read<long long>(ds, b.data(), {0, 0}, {hsize_t(R), 6});
        for (int a = 0; a < 3; ++a)
            same = same && b[size_t(c.rank) * 6 + a] == d.off[a] && b[size_t(c.rank) * 6 + 3 + a] == d.n[a];
    }
    same = c.allreduce_max(same ? 0.0 : 1.0) == 0;
    h5::Id pa = h5::open_group(it, "particles");
    for (Species& sp : s.species) {
        if (!h5::exists(pa, sp.name)) continue;                         // not loaded yet (t_start)
        h5::Id g = h5::open_group(pa, sp.name);
        const auto recs = species_records(g, sp, s.cfg.rz);
        h5::Id pp = h5::open_group(g, "particlePatches");
        if (same) {
            const auto np = read_1d<unsigned long long>(pp, "numParticles", hsize_t(c.rank), 1)[0];
            const auto no = read_1d<unsigned long long>(pp, "numParticlesOffset", hsize_t(c.rank), 1)[0];
            sp.resize(size_t(np));
            for (auto& r : recs)
                with_type(sp.info(r.second).type, [&](auto z) {
                    using T = decltype(z);
                    sp.set_as<T>(r.second, read_1d<T>(g, r.first, no, np));
                });
            h5::Id o = h5::open_dataset(st, "tile_start_offset/" + sp.name);
            std::vector<long long> to(size_t(R) + 1);
            h5::read<long long>(o, to.data(), {0}, {hsize_t(R) + 1});
            const long long a0 = to[size_t(c.rank)], a1 = to[size_t(c.rank) + 1];
            sp.tile_start = a1 > a0 ? read_1d<int>(st, "tile_start/" + sp.name, (unsigned long long)a0, (unsigned long long)(a1 - a0))
                                    : std::vector<int>();
            continue;
        }
        // another decomposition: scan all particles in chunks, keep those in [keep_lo, hi) (as migrate)
        const unsigned long long total = h5::dims(h5::open_dataset(g, "position/x")).at(0);
        const unsigned long long CH = 1 << 20;
        std::vector<std::vector<unsigned char>> col(recs.size());
        size_t kept = 0;
        for (unsigned long long c0 = 0; c0 < total; c0 += CH) {
            const unsigned long long m = std::min(CH, total - c0);
            // the particle axes X, Y, Z (RZ: only X = z is split, recs 2)
            const int na = s.cfg.rz ? 1 : 3;
            std::vector<double> x[3];
            for (int a = 0; a < na; ++a) x[a] = read_1d<double>(g, recs[size_t(s.cfg.rz ? 2 : a)].first, c0, m);
            std::vector<size_t> sel;
            for (size_t q = 0; q < m; ++q) {
                bool in = true;
                for (int a = 0; a < na; ++a) in = in && x[a][q] >= d.keep_lo(a) && x[a][q] < d.keep_hi(a);
                if (in) sel.push_back(q);
            }
            for (size_t r = 0; r < recs.size(); ++r)
                with_type(sp.info(recs[r].second).type, [&](auto z) {
                    using T = decltype(z);
                    const std::vector<T> v = read_1d<T>(g, recs[r].first, c0, m);
                    for (size_t q : sel) {
                        const unsigned char* p = reinterpret_cast<const unsigned char*>(&v[q]);
                        col[r].insert(col[r].end(), p, p + sizeof(T));
                    }
                });
            kept += sel.size();
        }
        sp.resize(kept);
        for (size_t r = 0; r < recs.size(); ++r)
            with_type(sp.info(recs[r].second).type, [&](auto z) {
                using T = decltype(z);
                std::vector<T> v(kept);
                if (kept) std::memcpy(v.data(), col[r].data(), kept * sizeof(T));
                sp.set_as<T>(recs[r].second, v);
            });
    }
    s.restart_step = s.istep;
}

bool read_op_state(const Simulation& s, const std::string& op, const std::vector<std::string>& keys, OpState& st) {
    st.arrays.clear();
    if (s.restart_path.empty()) return false;
    h5::Id f = h5::open_file(s.restart_path);
    h5::Id data = h5::open_group(f, "data");
    const std::vector<std::string> its = h5::children(data);
    if (its.size() != 1) throw std::runtime_error("checkpoint " + s.restart_path + ": expected one iteration");
    h5::Id it = h5::open_group(data, its[0]);
    h5::Id rp = h5::open_group(it, "rippic");
    if (!h5::exists(rp, "op_" + op)) return false;
    h5::Id og = h5::open_group(rp, "op_" + op);
    for (const std::string& k : keys) {
        std::vector<double> v;
        if (h5::exists(og, k)) {
            h5::Id ds = h5::open_dataset(og, k);
            const std::vector<hsize_t> n = h5::dims(ds);
            if (n.size() != 1) throw std::runtime_error("checkpoint: op_" + op + "/" + k + " is not one-dimensional");
            v.resize(size_t(n[0]));
            if (!v.empty()) h5::read<double>(ds, v.data(), {0}, {n[0]});
        }
        st.add(k, std::move(v));
    }
    return true;
}
#endif
