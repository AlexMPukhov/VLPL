// Local grid of one MPI rank and 3D field arrays with guard cells (device memory, see portable.hpp).
//
// Staggering of RIP-CC (all fields at integer time levels, all at integer x; Yee: stagger.hpp):
//   Ex, Jx        (i,     j,     k    )
//   Ey, Bz        (i,     j+1/2, k    )      transverse pairs co-located: F_y = Ey - vx Bz exact
//   Ez, By        (i,     j,     k+1/2)
//   Bx            (i,     j+1/2, k+1/2)
//   rho           (i+1/2, j,     k    )      Gauss law: dx Ex + mu_x dy Ey + mu_x dz Ez = rho
//   Jy            (i+1/2, j+1/2, k    )      stored directly where the transport source lives
//   Jz            (i+1/2, j,     k+1/2)
// Array index (i,j,k) of every component refers to the lower-left corner of the stagger above.
// The same positions as code: the descriptor RipCC in stagger.hpp (gather and deposition use only that).
#pragma once
#include <cstddef>
#include <vector>
#include "portable.hpp"

// Field solver of a run (docs/specs/yee.md): RIP-CC (c dt = h_x, the default) or Yee (the reference,
// dt from the Courant limit). The staggering descriptors are RipCC and Yee in stagger.hpp.
enum class Solver { RipCC, Yee, IRip };     // IRip: implicit RIP, box closure (docs/specs/implicit_rip.md)

struct Domain {
    int N[3] = {1, 1, 1};          // global cells
    double h[3] = {1, 1, 1};       // cell sizes (code units); dt = h[0] (c tau = h_x)
    bool periodic[3] = {true, true, true};
    int n[3] = {1, 1, 1};          // local (owned) cells
    int off[3] = {0, 0, 0};        // global index of local cell 0
    int ng = 3;                    // guard cells on each side
    // grid.geometry = cartesian2d (docs/specs/cart2d.md): one z cell, no z guards, and every field array has the z stride 0,
    // so f(i, j, k +- 1) is f(i, j, k): d/dz = 0 exactly in every Cart3D field kernel. N[2] = n[2] = 1, z periodic, h[2] = dz
    // the thickness of the slab that totals refer to.
    bool flat = false;
    // RZ (rz_domain): axis 2 holds the mode slots, never addressed outside [0, 2M - 1): its guards are neither exchanged nor
    // zeroed (they stay 0; the exchange of 6 fields zeroed 62 M doubles per call on a 2420 x 711 grid, 80 % of the step)
    bool slots = false;
    bool filt_comp = false;        // numerics.filter_compensate: one compensating step along x (z in RZ) after the passes
    bool filt_xonly = false;       // numerics.filter_axes = x: the binomial passes along x (z in RZ) only
    // numerics.filter_flat = L (binomial, decision 109): after the N passes the maximally flat (Herrmann) factor
    // P(S) = sum_{j<L} binom(N-1+j, j) S^j along x (z in RZ), S = 1 - T, applied as its real factors: kind 1 = (1 - a S),
    // one step with cf = a; kind 2 = (1 - a S + b S^2) of a complex root pair, two plain steps and a combination
    // (set_flat_filter, src/filter.cpp). L = 2 is the compensator.
    int filt_nflat = 0;
    int filt_fk[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    double filt_fa[8] = {0, 0, 0, 0, 0, 0, 0, 0}, filt_fb[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    bool o5 = false;               // numerics.closure = o5 (iRIP, decisions 72-76): the exact plane waves of side.cpp follow it
    int gw(int a) const { return a == 2 && flat ? 0 : ng; }      // guard cells along axis a
    int ext(int a) const { return n[a] + 2 * gw(a); }           // stored cells along axis a
    long wx = 0;                   // cells the moving window has shifted along +x (docs/specs/moving_window.md)
    double xfreeze = 1e300;        // particles at x >= xfreeze are not pushed (the window's entry columns)
    Solver solver = Solver::RipCC;
    double tau = 0;                // time step; 0: h_x (RIP-CC, c dt = h_x)
    long cell0(int a) const { return off[a] + (a == 0 ? wx : 0); }   // absolute index of local cell 0
    double lo(int a) const { return cell0(a) * h[a]; }        // local owned extent [lo, hi)
    double hi(int a) const { return (cell0(a) + n[a]) * h[a]; }
    // Lower bound of the particles a rank keeps: keep_below cells below lo at an open lower global edge, so a
    // leaving particle deposits the current that takes its charge out of cell 0 (docs/specs/open_boundary.md);
    // upper bound: keep_above cells above hi at an open upper global edge. Shapes 1, 2: 1 and 0; shape 3 (support of
    // 2 cells): 2 and 1 (docs/specs/solid_target.md).
    int keep_below = 1, keep_above = 0;
    double keep_lo(int a) const { return !periodic[a] && off[a] == 0 ? lo(a) - keep_below * h[a] : lo(a); }
    double keep_hi(int a) const { return !periodic[a] && off[a] + n[a] == N[a] ? hi(a) + keep_above * h[a] : hi(a); }
    double L(int a) const { return N[a] * h[a]; }
    double dt() const { return tau > 0 ? tau : h[0]; }
    double cell_volume() const { return h[0] * h[1] * h[2]; }
    long owned() const { return long(n[0]) * n[1] * n[2]; }
};

// Trivially copyable view of a field, usable inside kernels.
struct FV {
    double* p = nullptr;
    int sx = 0, sy = 0, ng = 0;    // sy = 0 on a flat (2D) domain: every k is the one (x, y) plane (Domain::flat)
    HD double& operator()(int i, int j, int k) const {
        return p[(long(k + ng) * sy + (j + ng)) * long(sx) + (i + ng)];
    }
};

// The array starts at element o_ of its buffer, which has one (x, y) plane of slack behind the array.
// x is the fastest index, so f(i) = f(i + 1) for the whole array is the array seen one element further
// on: shift_x moves o_ and copies only when the slack is used up (once per sx * sy shifts; the flat
// copy of every field was 1.2 ms per shift in bench/lwfa_bubble on the RTX 5090, 2026-10-28).
class Field {
public:
    Field() = default;
    explicit Field(const Domain& d) { resize(d); }
    void resize(const Domain& d) {
        n_[0] = d.n[0]; n_[1] = d.n[1]; n_[2] = d.n[2]; ng_ = d.ng; gz_ = d.gw(2); flat_ = d.flat;
        sx_ = n_[0] + 2 * ng_; sy_ = n_[1] + 2 * ng_; sz_ = n_[2] + 2 * gz_;
        ne_ = size_t(sx_) * sy_ * sz_; o_ = 0;
        v_.resize(ne_ + size_t(sx_) * sy_);
    }
    FV view() const {
        FV v; v.p = const_cast<double*>(v_.data()) + o_; v.sx = sx_; v.sy = flat_ ? 0 : sy_; v.ng = ng_;
        return v;
    }
    size_t idx(int i, int j, int k) const { return (size_t(flat_ ? 0 : k + gz_) * sy_ + (j + ng_)) * sx_ + (i + ng_); }
    void fill(double x) {
        if (x == 0.0) { v_.zero(); return; }
        double* p = v_.data();
        par_for(long(v_.size()), [=] HD(long q) { p[q] = x; });
    }
    void swap(Field& o) { v_.swap(o.v_); std::swap(o_, o.o_); }
    size_t size() const { return ne_; }             // elements including the guards
    // f(i) = f(i + 1) over the whole array. The last element (the top corner of the right x guard) is
    // not defined; the x guards need an exchange afterwards. tmp: a field of the same layout (scratch).
    void shift_x(Field& tmp) {
        if (o_ + 1 + ne_ <= v_.size()) { ++o_; return; }
        if (tmp.v_.size() != v_.size()) throw std::runtime_error("Field::shift_x: scratch of another layout");
        const double* a = v_.data() + o_;
        double* t = tmp.v_.data();
        par_for(long(ne_) - 1, [=] HD(long p) { t[p] = a[p + 1]; });
        tmp.o_ = 0;
        swap(tmp);
    }
    // The layout of o (scratch of the flat filter); storage kept when the size matches.
    void like(const Field& o) {
        for (int a = 0; a < 3; ++a) n_[a] = o.n_[a];
        ng_ = o.ng_; gz_ = o.gz_; sx_ = o.sx_; sy_ = o.sy_; sz_ = o.sz_; flat_ = o.flat_; ne_ = o.ne_; o_ = 0;
        if (v_.size() != o.v_.size()) v_.resize(o.v_.size());
    }
    // this = o over the whole array (guards included)
    void assign(const Field& o) {
        like(o);
        const double* a = o.v_.data() + o.o_;
        double* t = v_.data();
        par_for(long(ne_), [=] HD(long q) { t[q] = a[q]; });
    }
    // this = a x + b y + c this over the whole array (x, y of this layout)
    void lincomb(double a, const Field& x, double b, const Field& y, double c) {
        double* t = v_.data() + o_;
        const double* px = x.v_.data() + x.o_;
        const double* py = y.v_.data() + y.o_;
        par_for(long(ne_), [=] HD(long q) { t[q] = a * px[q] + b * py[q] + c * t[q]; });
    }
    // host mirror (whole array including guards), for tests and setup; index with idx()
    std::vector<double> host() const { std::vector<double> h(ne_); v_.to_host(h.data(), o_, ne_); return h; }
    void upload(const std::vector<double>& h) { v_.from_host(h.data(), o_, h.size()); }
    int n(int a) const { return n_[a]; }
    int ng() const { return ng_; }
    bool flat() const { return flat_; }
private:
    int n_[3] = {0, 0, 0}, ng_ = 0, gz_ = 0, sx_ = 0, sy_ = 0, sz_ = 0;
    bool flat_ = false;
    size_t ne_ = 0, o_ = 0;                         // elements of the array, its first element in v_
    Buf<double> v_;
};

// Kernel over the owned cells: f(i, j, k), f must be an HD lambda.
template <class F> inline void for_owned(const Domain& d, F f) {
    const int nx = d.n[0], ny = d.n[1];
    par_for(d.owned(), [=] HD(long q) {
        const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
        f(i, j, k);
    });
}

// The same over the owned cells and w guard layers on each side of every guarded axis (w <= the guard width)
template <class F> inline void for_owned_halo(const Domain& d, int w, F f) {
    int lo[3], ext[3];
    for (int a = 0; a < 3; ++a) {
        const int g = d.gw(a) > 0 ? w : 0;
        lo[a] = -g; ext[a] = d.n[a] + 2 * g;
    }
    const int l0 = lo[0], l1 = lo[1], l2 = lo[2], e0 = ext[0], e1 = ext[1];
    par_for(long(ext[0]) * ext[1] * ext[2], [=] HD(long q) {
        const int i = l0 + int(q % e0), j = l1 + int((q / e0) % e1), k = l2 + int(q / (long(e0) * e1));
        f(i, j, k);
    });
}
