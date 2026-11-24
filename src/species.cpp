#include "species.hpp"
#include "prof.hpp"
#include <algorithm>
#include <cstring>
#include <list>
#include <stdexcept>
#if RIP_GPU
#include <cub/cub.cuh>
#else
#include <omp.h>
#endif

// ---------------------------------------------------------------- kernels (free functions: no this capture)
void attr_fill_kernel(unsigned char* p, Attr t, double v, long from, long to) {
    const long n = to - from;
    if (t == Attr::F64) { double* q = reinterpret_cast<double*>(p) + from; par_for(n, [=] HD(long i) { q[i] = v; }); }
    else if (t == Attr::F32) { float* q = reinterpret_cast<float*>(p) + from; const float f = float(v); par_for(n, [=] HD(long i) { q[i] = f; }); }
    else if (t == Attr::I64) { long long* q = reinterpret_cast<long long*>(p) + from; const long long f = (long long)v; par_for(n, [=] HD(long i) { q[i] = f; }); }
    else { int* q = reinterpret_cast<int*>(p) + from; const int f = int(v); par_for(n, [=] HD(long i) { q[i] = f; }); }
}

// Device scratch (append_rows), kept for the whole run (never freed, like rip_reduce_partials). Slot `which` per type.
template <class T> static Buf<T>& scratch(int which, size_t n) {
    static Buf<T>* b[3] = {};
    if (!b[which]) b[which] = new Buf<T>();
    if (b[which]->size() < std::max<size_t>(n, 1)) b[which]->resize(std::max<size_t>(n, 1) + n / 4);
    return *b[which];
}

// dst[q] = src[P[q]]
void attr_gather_kernel(unsigned char* dst, const unsigned char* src, int es, const int* P, long n) {
    if (es == 8) {
        unsigned long long* d = reinterpret_cast<unsigned long long*>(dst);
        const unsigned long long* s = reinterpret_cast<const unsigned long long*>(src);
        par_for(n, [=] HD(long q) { d[q] = s[P[q]]; });
    } else {
        unsigned int* d = reinterpret_cast<unsigned int*>(dst);
        const unsigned int* s = reinterpret_cast<const unsigned int*>(src);
        par_for(n, [=] HD(long q) { d[q] = s[P[q]]; });
    }
}

// ---------------------------------------------------------------- registry
Species::Species() {
    const char* core[NC] = {"x", "y", "z", "ux", "uy", "uz", "w"};
    for (int c = 0; c < NC; ++c) require(core[c], Attr::F64, 0.0);
}

int Species::slot(const std::string& nm) const {
    for (int s = 0; s < n_attr(); ++s)
        if (info_[s].name == nm) return s;
    return -1;
}

int Species::require(const std::string& nm, Attr type, double init) {
    const int s = slot(nm);
    if (s >= 0) {
        if (info_[s].type != type) throw std::runtime_error("species " + name + ": attribute " + nm + " registered with another type");
        return s;
    }
    if (n_attr() >= MAX_ATTR) throw std::runtime_error("species " + name + ": more than MAX_ATTR attributes");
    info_.push_back(AttrInfo{nm, type, init});
    a_.emplace_back(cap_ * size_t(attr_size(type)));
    fill_init(n_attr() - 1, 0, n_);
    return n_attr() - 1;
}

int Species::row_bytes() const {
    int r = 0;
    for (auto& i : info_) r += attr_size(i.type);
    return r;
}

void Species::bad_access(int s) const {
    throw std::runtime_error("species " + name + ": attribute slot " + std::to_string(s) + " accessed with a wrong type");
}

void Species::fill_init(int s, size_t from, size_t to) {
    if (to > from) attr_fill_kernel(a_[s].data(), info_[s].type, info_[s].init, long(from), long(to));
}

// ---------------------------------------------------------------- storage
void Species::reserve(size_t cap) {
    if (cap <= cap_) return;
    const size_t c = std::max(cap, size_t(1.25 * cap_) + 1024);
    for (int s = 0; s < n_attr(); ++s) a_[s].grow_keep(c * size_t(attr_size(info_[s].type)));
    cap_ = c;
}

void Species::append(const std::vector<double> h[NC]) {
    const size_t m = h[0].size();
    reserve(n_ + m);
    for (int c = 0; c < NC; ++c) a_[c].from_host(reinterpret_cast<const unsigned char*>(h[c].data()), n_ * 8, m * 8);
    for (int s = NC; s < n_attr(); ++s) fill_init(s, n_, n_ + m);
    n_ += m;
    tile_start.clear();
}

size_t Species::extend(size_t m) {
    const size_t first = n_;
    reserve(n_ + m);
    for (int s = NC; s < n_attr(); ++s) fill_init(s, n_, n_ + m);
    n_ += m;
    tile_start.clear();
    return first;
}

AttrTable Species::table() {
    AttrTable t;
    t.n = n_attr(); t.R = row_bytes();
    for (int s = 0; s < t.n; ++s) { t.p[s] = a_[s].data(); t.es[s] = attr_size(info_[s].type); }
    return t;
}

// Row q of B (row_bytes, attributes in slot order) -> element n0 + q of every attribute. 4-byte words: a row with an I32
// or F32 attribute puts the following F64 at a 4-byte offset.
void attr_unpack_rows(AttrTable T, const unsigned char* B, long n0, long m) {
    par_for(m, [=] HD(long q) {
        const unsigned* s = reinterpret_cast<const unsigned*>(B + q * T.R);
        for (int a = 0; a < T.n; ++a) {
            unsigned* d = reinterpret_cast<unsigned*>(T.p[a] + (n0 + q) * T.es[a]);
            for (int w = 0; w < (T.es[a] >> 2); ++w) d[w] = *s++;
        }
    });
}

void Species::append_rows(const std::vector<unsigned char>& rows) { append_rows(rows.data(), rows.size()); }

void Species::append_rows(const unsigned char* rows, size_t bytes) {
    const size_t R = size_t(row_bytes()), m = bytes / R;
    if (m == 0) return;
    reserve(n_ + m);
    Buf<unsigned char>& db = scratch<unsigned char>(1, bytes);   // one upload, then one kernel for all attributes
    db.from_host(rows, 0, bytes);
    attr_unpack_rows(table(), db.data(), long(n_), long(m));
    n_ += m;
    tile_start.clear();
}

void Species::permute(const std::vector<int>& perm) {
    if (perm.size() != n_) throw std::runtime_error("Species::permute: wrong length");
    if (n_ == 0) return;
    if (perm_.size() < n_) perm_.resize(n_);
    perm_.from_host(perm.data(), 0, n_);
    permute_device(perm_.data());
}

// Each attribute is gathered into one scratch buffer and copied back, so a sort allocates nothing
// (the fifth / sixth GPU checks: one allocation per attribute and sort was ~1 ms per step).
void Species::permute_device(const int* perm) {
    if (n_ == 0) return;
    if (tmp_.size() < n_ * 8) tmp_.resize(cap_ * 8);
    for (int s = 0; s < n_attr(); ++s) {
        const int es = attr_size(info_[s].type);
        attr_gather_kernel(tmp_.data(), a_[s].data(), es, perm, long(n_));
        a_[s].copy_from(tmp_, 0, 0, n_ * size_t(es));
    }
}

template <class T> std::vector<T> Species::get_as(int s) const {
    check<T>(s);
    std::vector<T> v(n_);
    a_[s].to_host(reinterpret_cast<unsigned char*>(v.data()), 0, n_ * sizeof(T));
    return v;
}

template <class T> void Species::set_as(int s, const std::vector<T>& v) {
    check<T>(s);
    if (v.size() != n_) throw std::runtime_error("Species::set: wrong length");
    a_[s].from_host(reinterpret_cast<const unsigned char*>(v.data()), 0, n_ * sizeof(T));
}

template std::vector<double> Species::get_as<double>(int) const;
template std::vector<float> Species::get_as<float>(int) const;
template std::vector<long long> Species::get_as<long long>(int) const;
template std::vector<int> Species::get_as<int>(int) const;
template void Species::set_as<double>(int, const std::vector<double>&);
template void Species::set_as<float>(int, const std::vector<float>&);
template void Species::set_as<long long>(int, const std::vector<long long>&);
template void Species::set_as<int>(int, const std::vector<int>&);

double Species::kinetic_energy() const {
    const double* ux = ptr<double>(UX); const double* uy = ptr<double>(UY); const double* uz = ptr<double>(UZ);
    const double* w = ptr<double>(W);
    if (photon) return par_sum(long(n_), [=] HD(long p) { return w[p] * sqrt(ux[p] * ux[p] + uy[p] * uy[p] + uz[p] * uz[p]); });
    return m * par_sum(long(n_), [=] HD(long p) {
        const double u2 = ux[p] * ux[p] + uy[p] * uy[p] + uz[p] * uz[p];
        return w[p] * u2 / (sqrt(1.0 + u2) + 1.0);       // gamma - 1 without cancellation
    });
}

// ---------------------------------------------------------------- migration
// The leaving particles (a thin surface layer, |dx| < 1 cell per step) are flagged on the device, an exclusive scan
// of the flags gives each its row, one kernel packs the rows (all registered attributes) and one kernel closes the
// holes from the tail; the rows of all species go to the host in one copy, are exchanged with MPI along x, y, z (all
// species in one message per direction), and the arrivals are appended on the device. The core position x_a is the
// F64 at byte offset 8 a of a row. The particle order is that of the former version (the leavers in ascending order,
// the holes below n - m filled with the staying particles from the tail, the last one first), so the results are
// bit-identical to it.
void migrate_wrap_periodic(double* x, double* y, double* z, double L0, double L1, double L2, long n) {
    par_for(n, [=] HD(long p) {
        if (x[p] < 0) x[p] += L0; else if (x[p] >= L0) x[p] -= L0;
        if (y[p] < 0) y[p] += L1; else if (y[p] >= L1) y[p] -= L1;
        if (z[p] < 0) z[p] += L2; else if (z[p] >= L2) z[p] -= L2;
    });
}

// 2D (Domain::flat): z wraps over the one cell of the slab on the device, so no particle leaves along z
void migrate_wrap_z(double* z, double L2, long n) {
    par_for(n, [=] HD(long p) {
        if (z[p] < 0) z[p] += L2; else if (z[p] >= L2) z[p] -= L2;
    });
}

// F[p] = 1 for a particle outside the kept box (RZ: also at r >= rwall), else 0
void migrate_flag_outside(const double* x, const double* y, const double* z, const Domain& d, double rwall, int* F, long n) {
    const double l0 = d.keep_lo(0), l1 = d.keep_lo(1), l2 = d.keep_lo(2), h0 = d.keep_hi(0), h1 = d.keep_hi(1), h2 = d.keep_hi(2);
    const double rw2 = rwall * rwall;
    if (rwall > 0)
        par_for(n, [=] HD(long p) { F[p] = (x[p] < l0 || x[p] >= h0 || y[p] * y[p] + z[p] * z[p] >= rw2) ? 1 : 0; });
    else
        par_for(n, [=] HD(long p) {
            F[p] = (x[p] < l0 || x[p] >= h0 || y[p] < l1 || y[p] >= h1 || z[p] < l2 || z[p] >= h2) ? 1 : 0;
        });
}

// The leaver p goes to row P[p] of B, and L[P[p]] = p (the leavers in ascending order)
void migrate_pack(AttrTable T, const int* F, const int* P, int* L, unsigned char* B, long n) {
    par_for(n, [=] HD(long p) {
        if (!F[p]) return;
        const long q = P[p];
        L[q] = int(p);
        unsigned* d = reinterpret_cast<unsigned*>(B + q * T.R);
        for (int a = 0; a < T.n; ++a) {
            const unsigned* s = reinterpret_cast<const unsigned*>(T.p[a] + p * T.es[a]);
            for (int w = 0; w < (T.es[a] >> 2); ++w) *d++ = s[w];
        }
    });
}

// The holes below nn = n - m are the first h leavers L[0 .. h); the staying particles t in [nn, n) fill them, the last
// one the first hole: t gets the hole L[r] with r = (staying particles after t) = (n - 1 - t) - (m - P[t]).
void migrate_fill(AttrTable T, const int* F, const int* P, const int* L, long n, long m) {
    const long nn = n - m;
    par_for(m, [=] HD(long u) {
        const long t = nn + u;
        if (F[t]) return;
        const long dst = L[(n - 1 - t) - (m - P[t])];
        for (int a = 0; a < T.n; ++a) {
            if (T.es[a] == 8) {
                unsigned long long* v = reinterpret_cast<unsigned long long*>(T.p[a]);
                v[dst] = v[t];
            } else {
                unsigned* v = reinterpret_cast<unsigned*>(T.p[a]);
                v[dst] = v[t];
            }
        }
    });
}

void migrate_total(const int* F, const int* P, int* tot, long n) {
    par_for(1, [=] HD(long) { tot[0] = P[n - 1] + F[n - 1]; });
}

namespace {
struct MigScratch {
    Buf<int> F, P, L, tot;
    Buf<unsigned char> rows, tmp;
    HostBuf<unsigned char> host;                 // the rows of all species (pinned in the GPU build)
    HostBuf<unsigned char> up;                   // the arrivals of one species, for the upload
    std::vector<long> part;                      // CPU scan: one partial sum per thread
};
MigScratch& mig_scratch() {                      // kept for the whole run (never freed)
    static MigScratch* s = new MigScratch;
    return *s;
}
template <class T> void grow(Buf<T>& b, size_t n) {
    if (b.size() < std::max<size_t>(n, 1)) b.resize(std::max<size_t>(n, 1) + n / 4);
}

// P = exclusive sum of the flags F over n particles; returns the total
long scan_flags(MigScratch& S, long n) {
#if RIP_GPU
    size_t bytes = 0;
    gpu_check(cub::DeviceScan::ExclusiveSum(nullptr, bytes, S.F.data(), S.P.data(), int(n)), "cub scan (size)");
    grow(S.tmp, bytes);
    gpu_check(cub::DeviceScan::ExclusiveSum(S.tmp.data(), bytes, S.F.data(), S.P.data(), int(n)), "cub scan");
    grow(S.tot, 1);
    migrate_total(S.F.data(), S.P.data(), S.tot.data(), n);
    int m = 0;
    S.tot.to_host(&m, 0, 1);
    return m;
#else
    const int* F = S.F.data();
    int* P = S.P.data();
    std::vector<long>& part = S.part;
    long total = 0;
#pragma omp parallel
    {
        const int nt = omp_get_num_threads(), t = omp_get_thread_num();
        const long a = n * t / nt, b = n * (t + 1) / nt;
        long c = 0;
        for (long q = a; q < b; ++q) c += F[q];
#pragma omp single
        part.assign(size_t(nt) + 1, 0);
        part[size_t(t) + 1] = c;
#pragma omp barrier
#pragma omp single
        {
            for (int k = 0; k < nt; ++k) part[size_t(k) + 1] += part[size_t(k)];
            total = part[size_t(nt)];
        }
        long o = part[size_t(t)];
        for (long q = a; q < b; ++q) { P[q] = int(o); o += F[q]; }
    }
    return total;
#endif
}
}  // namespace

// w m (gamma - 1) of a row (the core attributes X, Y, Z, UX, UY, UZ, W are its first F64)
static double row_kinetic(const unsigned char* v, double m, bool photon) {   // photon: the energy |u| (m_e c^2)
    double u[3], w;
    std::memcpy(u, v + 8 * Species::UX, 24); std::memcpy(&w, v + 8 * Species::W, 8);
    const double u2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    return photon ? w * std::sqrt(u2) : m * w * u2 / (std::sqrt(1.0 + u2) + 1.0);
}

void migrate(Species& s, const Domain& d, Comm& comm, double rwall) {
    std::vector<Species*> v{&s};
    migrate(v, d, comm, rwall);
}

void migrate(const std::vector<Species*>& sp, const Domain& d, Comm& comm, double rwall) {
    const int S = int(sp.size());
    if (rwall <= 0 && comm.size == 1 && d.periodic[0] && d.periodic[1] && d.periodic[2]) {
        // single rank, all periodic: wrap in place on the device, nothing leaves
        for (Species* s : sp)
            migrate_wrap_periodic(s->ptr<double>(Species::X), s->ptr<double>(Species::Y), s->ptr<double>(Species::Z), d.L(0),
                                  d.L(1), d.L(2), long(s->size()));
        return;
    }
    MigScratch& M = mig_scratch();
    // 1. per species on the device: flag, scan, pack the leaving rows, close the holes; then the rows of all species
    // to the host in one copy
    std::vector<size_t> off(size_t(S) + 1, 0);     // byte offsets of the species' rows
    {
        PROF_SCOPE("migrate.device");
        for (int k = 0; k < S; ++k) {
            Species& s = *sp[size_t(k)];
            const long n = long(s.size());
            if (d.flat) migrate_wrap_z(s.ptr<double>(Species::Z), d.L(2), n);
            long m = 0;
            if (n > 0) {
                grow(M.F, size_t(n)); grow(M.P, size_t(n));
                migrate_flag_outside(s.ptr<double>(Species::X), s.ptr<double>(Species::Y), s.ptr<double>(Species::Z), d, rwall,
                                     M.F.data(), n);
                m = scan_flags(M, n);
            }
            off[size_t(k) + 1] = off[size_t(k)] + size_t(m) * size_t(s.row_bytes());
            if (m == 0) continue;
            grow(M.L, size_t(m));
            if (M.rows.size() < off[size_t(k) + 1]) M.rows.grow_keep(off[size_t(k) + 1] + off[size_t(k) + 1] / 4);
            const AttrTable T = s.table();
            migrate_pack(T, M.F.data(), M.P.data(), M.L.data(), M.rows.data() + off[size_t(k)], n);
            migrate_fill(T, M.F.data(), M.P.data(), M.L.data(), n, m);
            s.resize(size_t(n - m));
        }
        M.host.ensure(std::max<size_t>(off[size_t(S)], 1));
        M.rows.to_host(M.host.data(), 0, off[size_t(S)]);
    }
    // 2. route along x, y, z (host); particles that arrive may move on along the next axes. One message per direction
    // and axis for all species: S byte counts, then the rows of species 0, 1, ... The rows that stay are kept as runs
    // (segments) of the buffers they are in, so only the rows that move are copied.
    PROF_SCOPE("migrate.route");
    struct Seg { const unsigned char* p; size_t n; };
    std::vector<std::vector<Seg>> cur(static_cast<size_t>(S));
    std::list<std::vector<unsigned char>> store;                     // the received messages, alive to the end
    for (int k = 0; k < S; ++k)
        if (off[size_t(k) + 1] > off[size_t(k)]) cur[size_t(k)].push_back(Seg{M.host.data() + off[size_t(k)], off[size_t(k) + 1] - off[size_t(k)]});
    // keep(k, v, R): row v of species k stays; consecutive rows of one buffer become one segment
    auto keep_row = [](std::vector<Seg>& out, const unsigned char* v, size_t R) {
        if (!out.empty() && out.back().p + out.back().n == v) out.back().n += R;
        else out.push_back(Seg{v, R});
    };
    if (rwall > 0)                                                   // RZ: the wall absorbs, z is the only box axis
        for (int k = 0; k < S; ++k) {
            Species& s = *sp[size_t(k)];
            const size_t R = size_t(s.row_bytes());
            std::vector<Seg> keep;
            for (const Seg& g : cur[size_t(k)])
                for (size_t q = 0; q < g.n; q += R) {
                    const unsigned char* v = g.p + q;
                    double y, z;
                    std::memcpy(&y, v + 8, 8); std::memcpy(&z, v + 16, 8);
                    if (y * y + z * z < rwall * rwall) keep_row(keep, v, R);
                    else s.lost_energy += row_kinetic(v, s.m, s.photon);
                }
            cur[size_t(k)].swap(keep);
        }
    for (int a = 0; a < (rwall > 0 ? 1 : 3); ++a) {
        const double lo = d.keep_lo(a), hi = d.keep_hi(a), L = d.L(a);
        std::vector<std::vector<Seg>> keep(static_cast<size_t>(S));
        std::vector<unsigned char> msg[2];
        for (int dir = 0; dir < 2; ++dir) msg[dir].assign(size_t(S) * sizeof(unsigned long long), 0);
        for (int k = 0; k < S; ++k) {
            Species& s = *sp[size_t(k)];
            const size_t R = size_t(s.row_bytes());
            const size_t start[2] = {msg[0].size(), msg[1].size()};
            for (const Seg& g : cur[size_t(k)])
                for (size_t q = 0; q < g.n; q += R) {
                    const unsigned char* v = g.p + q;
                    double xa;
                    std::memcpy(&xa, v + 8 * a, 8);
                    const int dir = xa < lo ? 0 : (xa >= hi ? 1 : -1);
                    if (dir < 0) { keep_row(keep[size_t(k)], v, R); continue; }
                    if (comm.nbr[a][dir] < 0) { s.lost_energy += row_kinetic(v, s.m, s.photon); continue; }   // open boundary: particle leaves
                    const size_t at = msg[dir].size();
                    msg[dir].insert(msg[dir].end(), v, v + R);
                    if (d.periodic[a]) {                             // wrap at the global boundary (in the copy sent)
                        if (xa < 0) xa += L;
                        else if (xa >= L) xa -= L;
                        std::memcpy(msg[dir].data() + at + 8 * a, &xa, 8);
                    }
                }
            for (int dir = 0; dir < 2; ++dir) {
                const unsigned long long b = msg[dir].size() - start[dir];
                std::memcpy(msg[dir].data() + size_t(k) * sizeof b, &b, sizeof b);
            }
        }
        std::vector<unsigned char> in[2];                            // dir 0: to the left, both at once
        { PROF_SCOPE("migrate.mpi"); comm.exchange_bytes(a, msg, in, 200 + 4 * a); }
        for (int dir = 0; dir < 2; ++dir) {
            if (in[dir].size() < size_t(S) * sizeof(unsigned long long)) continue;
            store.push_back(std::move(in[dir]));
            const unsigned char* b0 = store.back().data();
            size_t p = size_t(S) * sizeof(unsigned long long);
            for (int k = 0; k < S; ++k) {
                unsigned long long b;
                std::memcpy(&b, b0 + size_t(k) * sizeof b, sizeof b);
                if (b > 0) keep[size_t(k)].push_back(Seg{b0 + p, size_t(b)});
                p += size_t(b);
            }
        }
        cur.swap(keep);
    }
    // 3. append the arrivals on the device: the segments of each species into the pinned buffer, one upload
    PROF_SCOPE("migrate.append");
    for (int k = 0; k < S; ++k) {
        size_t bytes = 0;
        for (const Seg& g : cur[size_t(k)]) bytes += g.n;
        if (bytes == 0) continue;
        unsigned char* h = M.up.ensure(bytes);
        size_t o = 0;
        for (const Seg& g : cur[size_t(k)]) { std::memcpy(h + o, g.p, g.n); o += g.n; }
        sp[size_t(k)]->append_rows(h, bytes);
    }
}
