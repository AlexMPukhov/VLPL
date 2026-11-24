#include "env.hpp"
#include "comm.hpp"
#include "prof.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

static double since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void Comm::init(int* argc, char*** argv) {
#ifdef RIP_MPI
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &local_rank);
    MPI_Comm_free(&node);
#else
    (void)argc; (void)argv;
#endif
#if RIP_GPU
    int ndev = 0;
    gpu_check(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
    if (ndev < 1) throw std::runtime_error("GPU build, but no CUDA device found");
    device = local_rank % ndev;
    gpu_check(cudaSetDevice(device), "cudaSetDevice");
#ifdef RIP_MPI
    const char* ca = vlpl_getenv("VLPL_CUDA_AWARE");
    cuda_aware = ca && std::atoi(ca) != 0;
#endif
#endif
}

void Comm::stage(size_t cnt) {
    if (hcap_ >= cnt) return;
#if RIP_GPU
    if (hsb_) { cudaFreeHost(hsb_); cudaFreeHost(hrb_); }
    gpu_check(cudaMallocHost(&hsb_, 2 * cnt * sizeof(double)), "cudaMallocHost");
    gpu_check(cudaMallocHost(&hrb_, 2 * cnt * sizeof(double)), "cudaMallocHost");
#else
    delete[] hsb_; delete[] hrb_;
    hsb_ = new double[2 * cnt]; hrb_ = new double[2 * cnt];
#endif
    hcap_ = cnt;
}

void Comm::finalize() {
#if RIP_GPU
    if (hsb_) { cudaFreeHost(hsb_); cudaFreeHost(hrb_); }
#else
    delete[] hsb_; delete[] hrb_;
#endif
    hsb_ = hrb_ = nullptr; hcap_ = 0;
#ifdef RIP_MPI
    if (cart_ != MPI_COMM_NULL) MPI_Comm_free(&cart_);
    MPI_Finalize();
#endif
}

void Comm::decompose(Domain& d, const int want[3]) {
#ifdef RIP_MPI
    int dm[3] = {want[0], want[1], want[2]}, per[3];
    for (int a = 0; a < 3; ++a) per[a] = d.periodic[a] ? 1 : 0;
    MPI_Dims_create(size, 3, dm);
    MPI_Cart_create(MPI_COMM_WORLD, 3, dm, per, 0, &cart_);
    MPI_Comm_rank(cart_, &rank);
    MPI_Cart_coords(cart_, rank, 3, coords);
    for (int a = 0; a < 3; ++a) {
        dims[a] = dm[a];
        int l, r;
        MPI_Cart_shift(cart_, a, 1, &l, &r);
        nbr[a][0] = (l == MPI_PROC_NULL) ? -1 : l;
        nbr[a][1] = (r == MPI_PROC_NULL) ? -1 : r;
    }
#else
    for (int a = 0; a < 3; ++a) {
        if (want[a] > 1) throw std::runtime_error("decomposition requested, but built without MPI");
        dims[a] = 1; coords[a] = 0;
        nbr[a][0] = nbr[a][1] = d.periodic[a] ? 0 : -1;
    }
#endif
    for (int a = 0; a < 3; ++a) {                      // as even as possible, remainder to the first ranks
        const int base = d.N[a] / dims[a], rem = d.N[a] % dims[a], c = coords[a];
        d.n[a] = base + (c < rem ? 1 : 0);
        d.off[a] = c * base + (c < rem ? c : rem);
        // guards are copied from owned cells only along periodic or split axes (else they are zeroed)
        if (d.n[a] < d.gw(a) && (d.periodic[a] || dims[a] > 1))
            throw std::runtime_error("axis " + std::to_string(a) + ": " + std::to_string(d.n[a]) +
                                     " local cells, at least " + std::to_string(d.ng) + " (guard width) needed");
    }
}

void Comm::sendrecv(const void* sbuf, size_t sbytes, int dest, void* rbuf, size_t rbytes, int src, int tag) {
#ifdef RIP_MPI
    MPI_Sendrecv(sbuf, int(sbytes), MPI_BYTE, dest < 0 ? MPI_PROC_NULL : dest, tag, rbuf, int(rbytes), MPI_BYTE,
                 src < 0 ? MPI_PROC_NULL : src, tag, cart_ != MPI_COMM_NULL ? cart_ : MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
#else
    (void)tag;
    const size_t n = std::min(sbytes, rbytes);
    if (dest >= 0 && src >= 0 && n > 0) std::memcpy(rbuf, sbuf, n);   // empty buffers may be null
#endif
}

void Comm::allreduce_sum(double* v, int n) {
#ifdef RIP_MPI
    MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    (void)v; (void)n;
#endif
}

double Comm::allreduce_max(double v) {
#ifdef RIP_MPI
    MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    return v;
}

void Comm::allgather(const double* send, int n, double* recv) {
#ifdef RIP_MPI
    MPI_Allgather(send, n, MPI_DOUBLE, recv, n, MPI_DOUBLE, MPI_COMM_WORLD);
#else
    for (int q = 0; q < n; ++q) recv[q] = send[q];
#endif
}

void Comm::bcast(std::string& s) {
#ifdef RIP_MPI
    unsigned long n = s.size();
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    s.resize(n);
    if (n) MPI_Bcast(&s[0], int(n), MPI_CHAR, 0, MPI_COMM_WORLD);
#else
    (void)s;
#endif
}

void Comm::barrier() {
#ifdef RIP_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void Comm::exchange_bytes(int a, const std::vector<unsigned char> send[2], std::vector<unsigned char> recv[2], int tag) {
    const auto t0 = std::chrono::steady_clock::now();
    const int dest[2] = {nbr[a][0], nbr[a][1]}, src[2] = {nbr[a][1], nbr[a][0]};
#ifdef RIP_MPI
    // One round: every neighbour gets one message (possibly empty); the receiver learns its size from a matched
    // probe (MPI_Mprobe, MPI-3), so no separate message with the sizes (that was a second latency per call).
    MPI_Comm c = cart_ != MPI_COMM_NULL ? cart_ : MPI_COMM_WORLD;
    MPI_Request req[2];
    int k = 0;
    for (int dir = 0; dir < 2; ++dir)
        if (dest[dir] >= 0)
            MPI_Isend(send[dir].data(), int(send[dir].size()), MPI_BYTE, dest[dir], tag + 2 + dir, c, &req[k++]);
    for (int dir = 0; dir < 2; ++dir) {
        recv[dir].clear();
        if (src[dir] < 0) continue;
        MPI_Message msg;
        MPI_Status st;
        MPI_Mprobe(src[dir], tag + 2 + dir, c, &msg, &st);
        int nb = 0;
        MPI_Get_count(&st, MPI_BYTE, &nb);
        recv[dir].resize(size_t(nb));
        MPI_Mrecv(recv[dir].data(), nb, MPI_BYTE, &msg, MPI_STATUS_IGNORE);
    }
    MPI_Waitall(k, req, MPI_STATUSES_IGNORE);
#else
    (void)tag;
    for (int dir = 0; dir < 2; ++dir) {                // the neighbours are this rank or none
        recv[dir].clear();
        if (dest[dir] >= 0 && src[dir] >= 0) recv[dir] = send[dir];
    }
#endif
    t_bytes += since(t0);
}

void Comm::report(FILE* f) {
    const double h = allreduce_max(t_halo), w = allreduce_max(t_halo_mpi), b = allreduce_max(t_bytes);
    if (rank != 0 || !f) return;
    std::fprintf(f, "  exchanges (max over ranks): guard cells %.2f s in %ld calls (MPI wait %.2f s), particle messages %.2f s; %s\n",
                 h, n_halo, w, b, size == 1 ? "one rank" : !RIP_GPU ? "CPU ranks"
                                  : cuda_aware ? "CUDA-aware MPI" : "through pinned host memory");
}

// Pack / unpack a slab along `axis` with indices [a0, a1) and the full guarded range in the other two
// axes, for all fields one after the other. mode 0: buf = f, 1: f = buf, 2: f += buf, 3: f = 0.
void slab_op(const Domain& d, std::vector<Field*>& f, int axis, int a0, int a1, double* buf, int mode) {
    int lo[3], ext[3];
    for (int b = 0; b < 3; ++b) { lo[b] = -d.gw(b); ext[b] = d.ext(b); }
    lo[axis] = a0; ext[axis] = a1 - a0;
    const long cnt = long(ext[0]) * ext[1] * ext[2];
    for (size_t q = 0; q < f.size(); ++q) {
        const FV v = f[q]->view();
        double* b = buf ? buf + q * cnt : nullptr;
        const int l0 = lo[0], l1 = lo[1], l2 = lo[2], e0 = ext[0], e1 = ext[1];
        par_for(cnt, [=] HD(long p) {
            const int i = l0 + int(p % e0), j = l1 + int((p / e0) % e1), k = l2 + int(p / (long(e0) * e1));
            double& x = v(i, j, k);
            if (mode == 0) b[p] = x;
            else if (mode == 1) x = b[p];
            else if (mode == 2) x += b[p];
            else x = 0.0;
        });
    }
}

// The same for modes 0-2 with all fields in one kernel on the GPU (one launch instead of one per field);
// the buffer layout is that of slab_op. On the CPU the loops of slab_op are cheaper (no division per cell).
constexpr int SLAB_MAXF = 8;
struct SlabViews { FV v[SLAB_MAXF]; };
void slab_op_all(const Domain& d, std::vector<Field*>& f, int axis, int a0, int a1, double* buf, int mode) {
    if (!RIP_GPU || f.size() > size_t(SLAB_MAXF)) { slab_op(d, f, axis, a0, a1, buf, mode); return; }
    int lo[3], ext[3];
    for (int b = 0; b < 3; ++b) { lo[b] = -d.gw(b); ext[b] = d.ext(b); }
    lo[axis] = a0; ext[axis] = a1 - a0;
    const long cnt = long(ext[0]) * ext[1] * ext[2];
    SlabViews V;
    for (size_t q = 0; q < f.size(); ++q) V.v[q] = f[q]->view();
    const int l0 = lo[0], l1 = lo[1], l2 = lo[2], e0 = ext[0], e1 = ext[1];
    par_for(cnt * long(f.size()), [=] HD(long t) {
        const int q = int(t / cnt);
        const long p = t - q * cnt;
        const int i = l0 + int(p % e0), j = l1 + int((p / e0) % e1), k = l2 + int(p / (long(e0) * e1));
        double& x = V.v[q](i, j, k);
        if (mode == 0) buf[t] = x;
        else if (mode == 1) x = buf[t];
        else x += buf[t];
    });
}

// Axis a periodic with this rank as both neighbours (no message): the guards straight from the owned edges, all fields
// and both directions in one kernel (was a pack and two unpacks through the buffers). add: the owned edges += the guards
// of the other side, then the guards = 0, as the general path does it; for n >= 2 g only (else the two edge slabs
// overlap and the general path keeps the order of the adds). Bit-identical to the general path.
struct SelfOp {
    int n, g, l0, l1, l2, e0, e1, ax;
    bool add;
    HD void operator()(const FV& v, long p) const {
        int r[3] = {l0 + int(p % e0), l1 + int((p / e0) % e1), l2 + int(p / (long(e0) * e1))};
        const int t = r[ax];
        auto at = [&](int c) -> double& { r[ax] = c; return v(r[0], r[1], r[2]); };
        if (!add) {
            at(-g + t) = at(n - g + t);
            at(n + t) = at(t);
        } else {
            at(t) += at(n + t);
            at(n - g + t) += at(-g + t);
            at(n + t) = 0.0;
            at(-g + t) = 0.0;
        }
    }
};
void slab_self(const Domain& d, std::vector<Field*>& f, int axis, bool add) {
    const int n = d.n[axis], g = d.gw(axis);
    int lo[3], ext[3];
    for (int b = 0; b < 3; ++b) { lo[b] = -d.gw(b); ext[b] = d.ext(b); }
    lo[axis] = 0; ext[axis] = g;
    const long cnt = long(ext[0]) * ext[1] * ext[2];
    const SelfOp op{n, g, lo[0], lo[1], lo[2], ext[0], ext[1], axis, add};
    if (RIP_GPU && f.size() <= size_t(SLAB_MAXF)) {
        SlabViews V;
        for (size_t q = 0; q < f.size(); ++q) V.v[q] = f[q]->view();
        par_for(cnt * long(f.size()), [=] HD(long u) { op(V.v[int(u / cnt)], u % cnt); });
    } else
        for (size_t q = 0; q < f.size(); ++q) {
            const FV v = f[q]->view();
            par_for(cnt, [=] HD(long p) { op(v, p); });
        }
}

#if RIP_GPU
namespace {
cudaEvent_t halo_event(int which) {               // two events, made once: entering and leaving Comm::halo
    static cudaEvent_t e[2] = {nullptr, nullptr};
    if (!e[which]) gpu_check(cudaEventCreateWithFlags(&e[which], cudaEventDisableTiming), "cudaEventCreate");
    return e[which];
}
}  // namespace
#endif

// Both directions of axis a at once. dir 0: send towards the right neighbour, receive from the left;
// dir 1: the reverse. The slabs sent are never the slabs received, so packing both before unpacking
// either gives the result of the two directions one after the other (the unpacks keep that order).
// GPU: everything of the call (pack, copies, unpack) runs in a high-priority non-blocking stream that starts after the
// work queued so far in the caller's stream and that the caller's stream waits for afterwards: the same order as
// before, but a kernel in another non-blocking stream (the interior push, push_deposit_overlap) runs on, and the
// unpack is not waited for on the host.
void Comm::halo(const Domain& d, std::vector<Field*>& f, int a, bool add) {
    const int n = d.n[a], g = d.gw(a);
    if (g == 0) return;                               // z of a flat (2D) domain: no guards
    const auto t0 = std::chrono::steady_clock::now();
#if RIP_GPU
    const cudaStream_t caller = rip_stream(), cs = rip_side_stream(1);
    gpu_check(cudaEventRecord(halo_event(0), caller), "cudaEventRecord");
    gpu_check(cudaStreamWaitEvent(cs, halo_event(0), 0), "cudaStreamWaitEvent");
    StreamGuard sg(cs);
#endif
    int dest[2], src[2];
    for (int dir = 0; dir < 2; ++dir) { dest[dir] = nbr[a][dir == 0 ? 1 : 0]; src[dir] = nbr[a][dir == 0 ? 0 : 1]; }
    if (dest[0] == rank && src[0] == rank && dest[1] == rank && src[1] == rank && (!add || n >= 2 * g)) {
        PROF_SCOPE("halo.self");
        slab_self(d, f, a, add);
    } else {
    size_t cnt = 1;
    for (int b = 0; b < 3; ++b) cnt *= size_t(b == a ? g : d.ext(b));
    cnt *= f.size();
    if (dsb_.size() < 2 * cnt) { dsb_.resize(2 * cnt); drb_.resize(2 * cnt); }
    int s0[2], r0[2];                                // first index of the sent / received slab
    bool remote[2];
    double* rbuf[2];                                 // the received slab of each direction (device)
    { PROF_SCOPE("halo.pack");
    for (int dir = 0; dir < 2; ++dir) {
        if (!add) { s0[dir] = dir == 0 ? n - g : 0;  r0[dir] = dir == 0 ? -g : n; }
        else      { s0[dir] = dir == 0 ? n : -g;     r0[dir] = dir == 0 ? 0 : n - g; }
        if (dest[dir] >= 0) slab_op_all(d, f, a, s0[dir], s0[dir] + g, dsb_.data() + dir * cnt, 0);
        // periodic with itself: unpack the packed slab directly (it is not overwritten before the unpack)
        const bool self = dest[dir] == rank && src[dir] == rank;
        rbuf[dir] = (self ? dsb_.data() : drb_.data()) + dir * cnt;
        remote[dir] = !self && (dest[dir] >= 0 || src[dir] >= 0);
    }
    }
    if (remote[0] || remote[1]) {
        PROF_SCOPE("halo.mpi");
#ifdef RIP_MPI
        MPI_Comm c = cart_ != MPI_COMM_NULL ? cart_ : MPI_COMM_WORLD;
        const bool dev = RIP_GPU && cuda_aware;
        double* sb = dsb_.data();
        double* rb = drb_.data();
        if (!dev) {
            stage(cnt); sb = hsb_; rb = hrb_;
            for (int dir = 0; dir < 2; ++dir)
                if (remote[dir] && dest[dir] >= 0) {
#if RIP_GPU
                    dsb_.to_host_async(sb + dir * cnt, dir * cnt, cnt, cs);
#else
                    dsb_.to_host(sb + dir * cnt, dir * cnt, cnt);
#endif
                }
        }
#if RIP_GPU
        gpu_check(cudaStreamSynchronize(cs), "halo stream");   // the packed slabs (and their copies) are complete
#endif
        const auto tm = std::chrono::steady_clock::now();
        MPI_Request req[4];
        int k = 0;
        for (int dir = 0; dir < 2; ++dir)
            if (remote[dir] && src[dir] >= 0)
                MPI_Irecv(rb + dir * cnt, int(cnt), MPI_DOUBLE, src[dir], 100 + 2 * a + dir, c, &req[k++]);
        for (int dir = 0; dir < 2; ++dir)
            if (remote[dir] && dest[dir] >= 0)
                MPI_Isend(sb + dir * cnt, int(cnt), MPI_DOUBLE, dest[dir], 100 + 2 * a + dir, c, &req[k++]);
        MPI_Waitall(k, req, MPI_STATUSES_IGNORE);
        t_halo_mpi += since(tm);
        if (!dev)
            for (int dir = 0; dir < 2; ++dir)
                if (remote[dir] && src[dir] >= 0) {
#if RIP_GPU
                    drb_.from_host_async(rb + dir * cnt, dir * cnt, cnt, cs);
#else
                    drb_.from_host(rb + dir * cnt, dir * cnt, cnt);
#endif
                }
#else
        for (int dir = 0; dir < 2; ++dir)              // without MPI the neighbours are this rank or none
            if (remote[dir] && dest[dir] >= 0 && src[dir] >= 0) {
#if RIP_GPU
                drb_.copy_from_async(dsb_, dir * cnt, dir * cnt, cnt, cs);
#else
                drb_.copy_from(dsb_, dir * cnt, dir * cnt, cnt);
#endif
            }
#endif
    }
    { PROF_SCOPE("halo.unpack");
    for (int dir = 0; dir < 2; ++dir) {
        if (src[dir] >= 0) slab_op_all(d, f, a, r0[dir], r0[dir] + g, rbuf[dir], add ? 2 : 1);
        else if (!add) slab_op(d, f, a, r0[dir], r0[dir] + g, nullptr, 3);
    }
    if (add) {
        slab_op(d, f, a, -g, 0, nullptr, 3);
        slab_op(d, f, a, n, n + g, nullptr, 3);
    }
    }
    }
#if RIP_GPU
    gpu_check(cudaEventRecord(halo_event(1), cs), "cudaEventRecord");
    gpu_check(cudaStreamWaitEvent(caller, halo_event(1), 0), "cudaStreamWaitEvent");
#endif
    t_halo += since(t0);
    ++n_halo;
}

void Comm::exchange(const Domain& d, std::vector<Field*> f) {
    for (int a = 0; a < (d.slots ? 2 : 3); ++a) halo(d, f, a, false);
}

void Comm::exchange_x(const Domain& d, std::vector<Field*> f) { halo(d, f, 0, false); }

void Comm::add_guards(const Domain& d, std::vector<Field*> f) {
    for (int a = 0; a < (d.slots ? 2 : 3); ++a) halo(d, f, a, true);
}
