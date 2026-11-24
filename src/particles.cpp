#include "particles.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include "geometry.hpp"
#include "pushers.hpp"
#include "external.hpp"
#include "tiles.hpp"

// ---------------------------------------------------------------- pushers (src/pushers.hpp)
HD void push_momentum(Pusher p, double qmdt, const double E[3], const double B[3], double u[3]) {
    if (p == Pusher::Boris) push_momentum_t<Pusher::Boris>(qmdt, E, B, u);
    else if (p == Pusher::HigueraCary) push_momentum_t<Pusher::HigueraCary>(qmdt, E, B, u);
}

// ---------------------------------------------------------------- particle views passed to the kernels
// zs: the charge state per particle (field ionization, ionization.hpp: charge q zs[p]), null for a species of one charge
struct PartViews { double *x, *y, *z, *ux, *uy, *uz, *w; const int* zs; };

static PartViews views(Species& s) {
    return PartViews{s.ptr<double>(Species::X), s.ptr<double>(Species::Y), s.ptr<double>(Species::Z),
                     s.ptr<double>(Species::UX), s.ptr<double>(Species::UY), s.ptr<double>(Species::UZ),
                     s.ptr<double>(Species::W), s.zslot >= 0 ? s.ptr<int>(s.zslot) : nullptr};
}

// ---------------------------------------------------------------- push kernels
struct PushParams {
    double dt, qmdt, q, ih[3], lo[3];
    Pusher pusher;
    double xf;                     // Domain::xfreeze
    QuadView quad;                 // external quadrupoles (external.hpp; n = 0: none)
};

static PushParams params(const Species& sp, const Domain& d, Pusher pusher) {
    const double dt = d.dt();
    return PushParams{dt, sp.q * dt / sp.m, sp.q, {1.0 / d.h[0], 1.0 / d.h[1], 1.0 / d.h[2]},
                      {d.lo(0), d.lo(1), d.lo(2)}, pusher, d.xfreeze, quadrupoles()};
}

// ---------------------------------------------------------------- external quadrupoles (external.hpp)
namespace {
struct QuadTable { Buf<Quad> dev; int n = 0; };
QuadTable& quad_table() {                          // process lifetime (never freed)
    static QuadTable* t = new QuadTable;
    return *t;
}
}  // namespace
void set_quadrupoles(const std::vector<Quad>& q) {
    QuadTable& t = quad_table();
    t.n = int(q.size());
    if (t.dev.size() < q.size()) t.dev.resize(q.size());
    if (!q.empty()) t.dev.from_host(q.data(), 0, q.size());
}
QuadView quadrupoles() { QuadTable& t = quad_table(); return QuadView{t.dev.data(), t.n}; }

// One particle: gather from F, push, move, and (Dep) the Esirkepov deposit through add (atomic, plain or
// fixed point) into the currents of FD whose index 0 is the local cell off[a] (FD = F, off = 0 for the
// global arrays).
using Atomic = cart3d::AddTo<true>;
using Plain = cart3d::AddTo<false>;
// Gather, push and move of one particle; r0, r1 are the local positions before and after (false: not
// moved, an entry column of the moving window). No writes to shared cells, so it vectorises over particles.
// r1 is taken back from the stored global position: fl(r1 + lo) - lo != r1 when lo != 0 (MPI ranks, the
// moving window), and the current must end where the next step and the charge deposit start, or the Gauss
// law drifts by ulp(x) / h_x per particle and step (2e-12 on 8 ranks at 4 n_c, decision 58).
template <class G, class St, int O>
HD inline bool move_one(const typename G::FieldViews& F, const PartViews& P, long p, const PushParams& c, double r0[3],
                        double r1[3]) {
    if (P.x[p] >= c.xf) return false;              // entry columns of the moving window: not yet pushed
    r0[0] = P.x[p] - c.lo[0]; r0[1] = P.y[p] - c.lo[1]; r0[2] = P.z[p] - c.lo[2];
    double E[3], B[3], u[3] = {P.ux[p], P.uy[p], P.uz[p]};
    if (c.pusher != Pusher::None) {                // None: pushed before by push_store (decision 15)
        G::template gather<St, O>(F, r0, c.ih, E, B);
        // external quadrupoles (external.hpp), averaged over the x crossed during the kick interval
        if (c.quad.n) c.quad.add(P.x[p], P.y[p], P.z[p], u[0] * c.dt / sqrt(1.0 + u[0] * u[0] + u[1] * u[1] + u[2] * u[2]), B);
        push_momentum(c.pusher, P.zs ? c.qmdt * P.zs[p] : c.qmdt, E, B, u);
    }
    const double ig = c.dt / sqrt(1.0 + u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    r1[0] = r0[0] + u[0] * ig; r1[1] = r0[1] + u[1] * ig; r1[2] = r0[2] + u[2] * ig;
    P.ux[p] = u[0]; P.uy[p] = u[1]; P.uz[p] = u[2];
    P.x[p] = r1[0] + c.lo[0]; P.y[p] = r1[1] + c.lo[1]; P.z[p] = r1[2] + c.lo[2];
    r1[0] = P.x[p] - c.lo[0]; r1[1] = P.y[p] - c.lo[1]; r1[2] = P.z[p] - c.lo[2];
    return true;
}
constexpr int CHUNK = 64;
template <class G, class St, int O, class Add>
HD inline void deposit_one(const typename G::FieldViews& FD, const int off[3], const PartViews& P, long p,
                           const PushParams& c, const double r0[3], const double r1[3], const Add& add) {
    const double qw = (P.zs ? c.q * P.zs[p] : c.q) * P.w[p] / c.dt;
    const double coef[3] = {qw * c.ih[1] * c.ih[2], qw * c.ih[0] * c.ih[2], qw * c.ih[0] * c.ih[1]};
    G::template deposit_shifted<St, O>(FD, r0, r1, c.ih, coef, off, add);
}
template <class G, class St, int O, bool Dep, class Add>
HD inline void push_one(const typename G::FieldViews& F, const typename G::FieldViews& FD, const int off[3],
                        const PartViews& P, long p, const PushParams& c, const Add& add) {
    double r0[3], r1[3];
    if (move_one<G, St, O>(F, P, p, c, r0, r1) && Dep) deposit_one<G, St, O>(FD, off, P, p, c, r0, r1, add);
}

template <class G, class St, int O, bool Dep>
void push_impl(Species& sp, Fields& f, const Domain& d, Pusher pusher) {
    const PushParams c = params(sp, d, pusher);
    const typename G::FieldViews F = G::views(f);
    const PartViews P = views(sp);
    par_for(long(sp.size()), [=] HD(long p) {
        const int z[3] = {0, 0, 0};
        push_one<G, St, O, Dep>(F, F, z, P, p, c, Atomic{});
    });
}

// Device scratch of the tiled and box deposits, kept between calls and only grown: allocating it per
// call cost ~1 ms per step on the GPU (fifth GPU check). Never freed (process lifetime).
struct DepositScratch { Buf<int> ts, tl, list, count; };
static DepositScratch& deposit_scratch() {
    static DepositScratch* s = new DepositScratch;
    return *s;
}
static void ensure(Buf<int>& b, size_t n) { if (b.size() < n) b.resize(n); }

// Coloured tiles (docs/specs/tile_deposit.md): one thread per tile of a colour, plain additions; the
// particles that left their tile since the last sort are listed (atomic counter, strays only) and
// deposited atomically afterwards.
template <class G, class St, int O>
long push_tiled_impl(Species& sp, Fields& f, const Domain& d, Pusher pusher, const TileGrid& g) {
    const long n = long(sp.size());
    if (n == 0) return 0;
    const PushParams c = params(sp, d, pusher);
    const typename G::FieldViews F = G::views(f);
    const PartViews P = views(sp);
    std::vector<int> order, cstart;
    tile_colours(g, order, cstart);
    DepositScratch& W = deposit_scratch();
    ensure(W.ts, sp.tile_start.size()); ensure(W.tl, order.size()); ensure(W.list, size_t(n)); ensure(W.count, 1);
    W.ts.from_host(sp.tile_start.data(), 0, sp.tile_start.size());
    W.tl.from_host(order.data(), 0, order.size());
    W.count.zero();
    const int* TS = W.ts.data(); const int* TL = W.tl.data(); int* L = W.list.data(); int* C = W.count.data();
    const int n0 = d.n[0], n1 = d.n[1], n2 = d.n[2];
    for (int col = 0; col < 8; ++col) {
        const int c0 = cstart[col];
        par_for(long(cstart[col + 1] - c0), [=] HD(long q) {
            const int t = TL[c0 + q];
            const int tx = t % g.nt[0], ty = (t / g.nt[0]) % g.nt[1], tz = t / (g.nt[0] * g.nt[1]);
            const int a0 = tx * g.T[0], a1 = ty * g.T[1], a2 = tz * g.T[2];
            const int b0 = a0 + g.T[0] < n0 ? a0 + g.T[0] : n0, b1 = a1 + g.T[1] < n1 ? a1 + g.T[1] : n1,
                      b2 = a2 + g.T[2] < n2 ? a2 + g.T[2] : n2;
            // chunks of particles: first gather, push and move of the whole chunk, then the deposits (two
            // passes overlap better than one particle at a time: 1.6x, decision 29); the arithmetic of push_one
            const int ta[3] = {a0, a1, a2}, tb[3] = {b0, b1, b2};
            for (int p0 = TS[t]; p0 < TS[t + 1]; p0 += CHUNK) {
                const int np = TS[t + 1] - p0 < CHUNK ? TS[t + 1] - p0 : CHUNK;
                double R0[3][CHUNK], R1[3][CHUNK];
                unsigned char st[CHUNK];
                for (int q = 0; q < np; ++q) {         // 0: moved, 1: not moved, 2: stray (not in its tile)
                    const long p = p0 + q;
                    int i = int(floor((P.x[p] - c.lo[0]) * c.ih[0])), j = int(floor((P.y[p] - c.lo[1]) * c.ih[1])),
                        k = int(floor((P.z[p] - c.lo[2]) * c.ih[2]));
                    // clamped as by the sort (tile_keys_kernel): a particle in the cell below an open lower
                    // edge (leaving, Domain::keep_lo) belongs to the first tile, and no tile lies below it,
                    // so its plain deposit races with none (else it would be a stray deposited atomically
                    // in a thread-dependent order, one column per step in a moving window)
                    // (shape 3 also keeps particles one cell above an open upper edge, Domain::keep_hi: the last tile)
                    i = i < 0 ? 0 : (i >= n0 ? n0 - 1 : i); j = j < 0 ? 0 : (j >= n1 ? n1 - 1 : j);
                    k = k < 0 ? 0 : (k >= n2 ? n2 - 1 : k);
                    const bool stray = i < ta[0] || i >= tb[0] || j < ta[1] || j >= tb[1] || k < ta[2] || k >= tb[2];
                    double r0[3] = {0, 0, 0}, r1[3] = {0, 0, 0};   // (not set for strays and unmoved particles)
                    st[q] = stray ? 2 : (move_one<G, St, O>(F, P, p, c, r0, r1) ? 0 : 1);
                    for (int a = 0; a < 3; ++a) { R0[a][q] = r0[a]; R1[a][q] = r1[a]; }
                }
                const int z[3] = {0, 0, 0};
                for (int q = 0; q < np; ++q) {
                    if (st[q] == 2) { L[atomic_inc(C)] = p0 + q; continue; }
                    if (st[q] == 1) continue;
                    const double r0[3] = {R0[0][q], R0[1][q], R0[2][q]}, r1[3] = {R1[0][q], R1[1][q], R1[2][q]};
                    deposit_one<G, St, O>(F, z, P, p0 + q, c, r0, r1, Plain{});
                }
            }
        });
    }
    int ns = 0;
    W.count.to_host(&ns, 0, 1);
    if (ns > 0)
        par_for(ns, [=] HD(long q) {
            const int z[3] = {0, 0, 0};
            push_one<G, St, O, true>(F, F, z, P, L[q], c, Atomic{});
        });
    return ns;
}

// Box deposit (docs/specs/tile_deposit.md, GPU version): one block per tile accumulates the current of its
// particles in a scratch box of (T + 5) cells per axis and 3 components, which covers the reach
// [a - 2, b + 2] of a tile [a, b) (shape 3: [a - 3, b + 2] along x, T + 6 cells); on the GPU the box is in shared memory. Then one atomic add per
// non-zero box cell into the global currents. Strays (not in their tile) are listed and deposited
// globally with atomics afterwards, as in the tiled path: the box kernel holds one copy of the deposit
// (less to spill under a register cap, decision 27).
// The box holds fixed-point int64 (decision 26), added as two native 32-bit atomics (block_add):
// shared-memory FP64 and int64 atomics are CAS loops on the GPU. A particle adds at most |q w| / dV to
// a box cell (|v_a| < 1 in the Esirkepov sums), so a tile of N particles sums to less than
// B = N max|q w| / dV. The scale 2^(62 - e) with B < 2^e keeps the sums below 2^62 plus half a unit
// per addend (< 2^63) and rounds each addend to within B 2^-62 absolute: round-off relative to the heaviest content of the tile, but a cell of the same
// tile that is 1e-9 of it keeps only ~1e-6 relative accuracy (test 15). Integer sums are associative:
// the box is the same in any thread order.
struct FixedAdd {
    static constexpr bool skip_zero = true;
    double scale;
    HD void operator()(double* p, double v) const {
        const long long q = llrint(v * scale);
        if (q != 0) block_add(reinterpret_cast<long long*>(p), q);
    }
};
HD inline double box_scale(int count, double qwmax) {      // power of two: the conversion back is exact
    const double b = count * qwmax;
    if (!(b > 0)) return 1.0;
    int e;
    frexp(b, &e);
    return ldexp(1.0, 62 - e);
}
// Occupancy (decision 27): uncapped, the shape-1 box kernel takes 240 registers (ptxas, sm_120), one
// block of 256 per SM; the global push takes 80 (three blocks). RIP_BOX_MINB1 = 2 caps it at 128
// without spills (two blocks), 3 at 80 with 56 bytes of spills; shape 2 (254 registers, like the global
// push) is capped by RIP_BOX_MINB2. Build with -DRIP_BOX_MINB1=1 / 3 to compare.
#ifndef RIP_BOX_MINB1
#define RIP_BOX_MINB1 2
#endif
#ifndef RIP_BOX_MINB2
#define RIP_BOX_MINB2 1
#endif
template <class G, class St, int O>
long push_box_impl(Species& sp, Fields& f, const Domain& d, Pusher pusher, const TileGrid& g) {
    const long n = long(sp.size());
    if (n == 0) return 0;
    const PushParams c = params(sp, d, pusher);
    const typename G::FieldViews F = G::views(f);
    const PartViews P = views(sp);
    DepositScratch& W = deposit_scratch();
    ensure(W.ts, sp.tile_start.size()); ensure(W.list, size_t(n)); ensure(W.count, 1);
    W.ts.from_host(sp.tile_start.data(), 0, sp.tile_start.size());
    W.count.zero();
    const int* TS = W.ts.data(); int* L = W.list.data(); int* C = W.count.data();
    constexpr int X3 = O == 3;                    // shape 3 reaches one cell lower along x (charge grid at i + 1/2)
    const int B0 = g.T[0] + 5 + X3, B1 = g.T[1] + 5, B2 = G::grid_dims == 2 ? 1 : g.T[2] + 5;   // Cart2D: one plane
    const long nb3 = long(B0) * B1 * B2;
    const int n0 = d.n[0], n1 = d.n[1], n2 = d.n[2], ng = d.ng;
    const double* PW = P.w;
    const double qwmax = std::abs(sp.q) * (sp.zslot >= 0 ? sp.charge_max : 1.0) * par_max_abs(n, [=] HD(long p) { return PW[p]; }) * c.ih[0] * c.ih[1] * c.ih[2];
    par_blocks<O == 1 ? RIP_BOX_MINB1 : RIP_BOX_MINB2>(g.count(), int(3 * nb3),
        [=] HD(long t, int tid, int nt, double* s) {
            const int tx = int(t % g.nt[0]), ty = int((t / g.nt[0]) % g.nt[1]), tz = int(t / (long(g.nt[0]) * g.nt[1]));
            const int a0 = tx * g.T[0], a1 = ty * g.T[1], a2 = tz * g.T[2];
            const int b0 = a0 + g.T[0] < n0 ? a0 + g.T[0] : n0, b1 = a1 + g.T[1] < n1 ? a1 + g.T[1] : n1,
                      b2 = a2 + g.T[2] < n2 ? a2 + g.T[2] : n2;
            typename G::FieldViews FB = F;
            FB.jx = FV{s, B0, B1, 0}; FB.jy = FV{s + nb3, B0, B1, 0}; FB.jz = FV{s + 2 * nb3, B0, B1, 0};
            const int off[3] = {a0 - 2 - X3, a1 - 2, a2 - 2};
            const FixedAdd add{box_scale(TS[t + 1] - TS[t], qwmax)};
            for (int p = TS[t] + tid; p < TS[t + 1]; p += nt) {
                const int i = int(floor((P.x[p] - c.lo[0]) * c.ih[0])), j = int(floor((P.y[p] - c.lo[1]) * c.ih[1])),
                          k = int(floor((P.z[p] - c.lo[2]) * c.ih[2]));
                if (i < a0 || i >= b0 || j < a1 || j >= b1 || k < a2 || k >= b2) { L[atomic_inc(C)] = p; continue; }
                push_one<G, St, O, true>(F, FB, off, P, p, c, add);
            }
        },
        [=] HD(long t, int tid, int nt, double* s) {
            const int tx = int(t % g.nt[0]), ty = int((t / g.nt[0]) % g.nt[1]), tz = int(t / (long(g.nt[0]) * g.nt[1]));
            const int o0 = tx * g.T[0] - 2 - X3, o1 = ty * g.T[1] - 2, o2 = tz * g.T[2] - 2;
            const long long* S = reinterpret_cast<const long long*>(s);
            const double inv = 1.0 / box_scale(TS[t + 1] - TS[t], qwmax);
            for (long q = tid; q < 3 * nb3; q += nt) {
                if (S[q] == 0) continue;
                const double v = double(S[q]) * inv;
                const long r = q % nb3;
                const int gi = o0 + int(r % B0), gj = o1 + int((r / B0) % B1), gk = o2 + int(r / (long(B0) * B1));
                if (gi >= n0 + ng || gj >= n1 + ng || gk >= n2 + ng) continue;   // beyond a partial last tile: never written
                const FV& J = q < nb3 ? F.jx : (q < 2 * nb3 ? F.jy : F.jz);
                atomic_add(&J(gi, gj, gk), v);
            }
        });
    int ns = 0;
    W.count.to_host(&ns, 0, 1);
    if (ns > 0)
        par_for(ns, [=] HD(long q) {
            const int z[3] = {0, 0, 0};
            push_one<G, St, O, true>(F, F, z, P, L[q], c, Atomic{});
        });
    return ns;
}

void push_and_deposit(Species& s, Fields& f, const Domain& d, Pusher pusher, int order) {
    with_stagger(d, order, [&](auto geo, auto st, auto o) { push_impl<decltype(geo), decltype(st), decltype(o)::value, true>(s, f, d, pusher); });
}

// Gather + push + deposit of a part of the particles, for the overlap of the guard exchanges with the push
// (PushDeposit::apply_overlap in sim.cpp): part 1 the boundary particles, whose cell is within 2 g of a face of the owned
// box (g = guard width along that axis), part 2 the interior ones. An interior particle reads and writes only cells in
// [g, n - g) along every guarded axis, so the guard exchanges of the currents (which write the guards and the edge
// layers [0, g), [n - g, n)) can run while the interior part is pushed. The same kernel as push_and_deposit per particle.
// The classification is made once, by part 1 from the positions before the push (in[p] = 1: interior, pushed by part 2):
// a boundary particle that part 1 moves into the interior must not be pushed again (the first version did, decision 92).
template <class G, class St, int O>
void push_part_impl(Species& sp, Fields& f, const Domain& d, Pusher pusher, int part, unsigned char* in) {
    const PushParams c = params(sp, d, pusher);
    const typename G::FieldViews F = G::views(f);
    const PartViews P = views(sp);
    int lo[3], hi[3];
    for (int a = 0; a < 3; ++a) {
        const int g = d.gw(a);
        lo[a] = g > 0 ? 2 * g : -(1 << 30);
        hi[a] = g > 0 ? d.n[a] - 2 * g : (1 << 30);
    }
    const int l0 = lo[0], l1 = lo[1], l2 = lo[2], h0 = hi[0], h1 = hi[1], h2 = hi[2];
    if (part == 1)
        par_for(long(sp.size()), [=] HD(long p) {
            const double cx = floor((P.x[p] - c.lo[0]) * c.ih[0]), cy = floor((P.y[p] - c.lo[1]) * c.ih[1]),
                         cz = floor((P.z[p] - c.lo[2]) * c.ih[2]);
            const bool inner = cx >= l0 && cx < h0 && cy >= l1 && cy < h1 && cz >= l2 && cz < h2;
            in[p] = inner ? 1 : 0;
            if (inner) return;
            const int z[3] = {0, 0, 0};
            push_one<G, St, O, true>(F, F, z, P, p, c, Atomic{});
        });
    else
        par_for(long(sp.size()), [=] HD(long p) {
            if (!in[p]) return;
            const int z[3] = {0, 0, 0};
            push_one<G, St, O, true>(F, F, z, P, p, c, Atomic{});
        });
}

void push_and_deposit_part(Species& s, Fields& f, const Domain& d, Pusher pusher, int order, int part, unsigned char* in) {
    with_stagger(d, order, [&](auto geo, auto st, auto o) {
        push_part_impl<decltype(geo), decltype(st), decltype(o)::value>(s, f, d, pusher, part, in);
    });
}

long push_and_deposit_box(Species& s, Fields& f, const Domain& d, Pusher pusher, int order, const TileGrid& g) {
    if (s.tile_start.size() != size_t(g.count()) + 1) {        // not sorted since the last change
        push_and_deposit(s, f, d, pusher, order);
        return long(s.size());
    }
    return with_stagger(d, order, [&](auto geo, auto st, auto o) { return push_box_impl<decltype(geo), decltype(st), decltype(o)::value>(s, f, d, pusher, g); });
}

long push_and_deposit_tiled(Species& s, Fields& f, const Domain& d, Pusher pusher, int order, const TileGrid& g) {
    if (s.tile_start.size() != size_t(g.count()) + 1) {        // not sorted since the last change
        push_and_deposit(s, f, d, pusher, order);
        return long(s.size());
    }
    return with_stagger(d, order, [&](auto geo, auto st, auto o) { return push_tiled_impl<decltype(geo), decltype(st), decltype(o)::value>(s, f, d, pusher, g); });
}

void push_only(Species& s, const Fields& f, const Domain& d, Pusher pusher, int order) {
    auto& ff = const_cast<Fields&>(f);              // not written when deposit == false
    with_stagger(d, order, [&](auto geo, auto st, auto o) { push_impl<decltype(geo), decltype(st), decltype(o)::value, false>(s, ff, d, pusher); });
}

template <class G, class St, int O> void push_store_impl(Species& sp, Fields& f, const Domain& d, Pusher pusher, double* EB) {
    const PushParams c = params(sp, d, pusher);
    const typename G::FieldViews F = G::views(f);
    const PartViews P = views(sp);
    par_for(long(sp.size()), [=] HD(long p) {
        double* o = EB + 9 * p;
        if (P.x[p] >= c.xf) { for (int k = 0; k < 9; ++k) o[k] = 0; return; }   // entry columns: not pushed
        const double r0[3] = {P.x[p] - c.lo[0], P.y[p] - c.lo[1], P.z[p] - c.lo[2]};
        double E[3], B[3], u[3] = {P.ux[p], P.uy[p], P.uz[p]};
        G::template gather<St, O>(F, r0, c.ih, E, B);
        if (c.quad.n) c.quad.add(P.x[p], P.y[p], P.z[p], u[0] * c.dt / sqrt(1.0 + u[0] * u[0] + u[1] * u[1] + u[2] * u[2]), B);
        for (int k = 0; k < 3; ++k) { o[k] = E[k]; o[3 + k] = B[k]; o[6 + k] = u[k]; }
        push_momentum(c.pusher, P.zs ? c.qmdt * P.zs[p] : c.qmdt, E, B, u);
        P.ux[p] = u[0]; P.uy[p] = u[1]; P.uz[p] = u[2];
    });
}
void push_store(Species& s, Fields& f, const Domain& d, Pusher pusher, int order, double* EB) {
    with_stagger(d, order, [&](auto geo, auto st, auto o) { push_store_impl<decltype(geo), decltype(st), decltype(o)::value>(s, f, d, pusher, EB); });
}

template <class G, class St, int O> void rho_impl(const Species& sp, Field& rho, const Domain& d, double x0, double x1, bool range) {
    const double ih0 = 1.0 / d.h[0], ih1 = 1.0 / d.h[1], ih2 = 1.0 / d.h[2], iv = 1.0 / d.cell_volume();
    const double lo0 = d.lo(0), lo1 = d.lo(1), lo2 = d.lo(2), q = sp.q;
    const PartViews P = views(const_cast<Species&>(sp));
    const FV R = rho.view();
    par_for(long(sp.size()), [=] HD(long p) {
        if (range && (P.x[p] < x0 || P.x[p] >= x1)) return;
        const double ih[3] = {ih0, ih1, ih2};
        const double r[3] = {P.x[p] - lo0, P.y[p] - lo1, P.z[p] - lo2};
        G::template deposit_rho<St, O>(R, r, ih, (P.zs ? q * P.zs[p] : q) * P.w[p] * iv);
    });
}

void deposit_rho(const Species& s, Field& rho, const Domain& d, int order) {
    with_stagger(d, order, [&](auto geo, auto st, auto o) { rho_impl<decltype(geo), decltype(st), decltype(o)::value>(s, rho, d, 0, 0, false); });
}

void deposit_rho(const Species& s, Field& rho, const Domain& d, int order, double x0, double x1) {
    with_stagger(d, order, [&](auto geo, auto st, auto o) { rho_impl<decltype(geo), decltype(st), decltype(o)::value>(s, rho, d, x0, x1, true); });
}
