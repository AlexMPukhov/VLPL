// Portability layer (after GEM-PIC-3D): every kernel is written once as a lambda that runs as an
// OpenMP loop (CPU build) or as a CUDA kernel (nvcc, --extended-lambda). Device memory is Buf<T>;
// in the CPU build it is ordinary host memory.
#pragma once
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define HD __host__ __device__
#define RIP_GPU 1
#else
#define HD
#define RIP_GPU 0
#endif

inline void gpu_check(int code, const char* what) {
#if RIP_GPU
    if (code != 0) throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(cudaError_t(code)));
#else
    (void)code; (void)what;
#endif
}

// ---------------------------------------------------------------- kernels
#if RIP_GPU
// The stream par_for launches into: the legacy default stream (0) unless a StreamGuard is active. Everything else
// (copies, reductions, CUB) runs in the legacy stream, which does not wait for the non-blocking streams of the guards:
// this is how the guard exchanges run on the device while the interior push runs (Comm::halo, push_deposit_overlap).
inline cudaStream_t& rip_stream() { static cudaStream_t s = 0; return s; }
class StreamGuard {
public:
    explicit StreamGuard(cudaStream_t s) : old_(rip_stream()) { rip_stream() = s; }
    ~StreamGuard() { rip_stream() = old_; }
    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;
private:
    cudaStream_t old_;
};
// A non-blocking stream of the given priority: 0 = the lowest, 1 = the highest the device offers (made once each)
inline cudaStream_t rip_side_stream(int high) {
    static cudaStream_t s[2] = {nullptr, nullptr};
    if (!s[high]) {
        int lo = 0, hi = 0;
        gpu_check(cudaDeviceGetStreamPriorityRange(&lo, &hi), "stream priorities");
        gpu_check(cudaStreamCreateWithPriority(&s[high], cudaStreamNonBlocking, high ? hi : lo), "cudaStreamCreate");
    }
    return s[high];
}
// Stream s starts its next work after everything queued so far in the stream prior (0: the legacy default stream)
inline void rip_stream_after(cudaStream_t s, cudaStream_t prior) {
    static cudaEvent_t e = nullptr;
    if (!e) gpu_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "cudaEventCreate");
    gpu_check(cudaEventRecord(e, prior), "cudaEventRecord");
    gpu_check(cudaStreamWaitEvent(s, e, 0), "cudaStreamWaitEvent");
}
template <class F> __global__ void rip_kernel(long n, F f) {
    const long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) f(i);
}
template <class F> inline void par_for(long n, F f) {
    if (n <= 0) return;
    const int t = 256;
    rip_kernel<<<unsigned((n + t - 1) / t), t, 0, rip_stream()>>>(n, f);
    gpu_check(cudaGetLastError(), "kernel launch");
}
inline void device_sync() { gpu_check(cudaDeviceSynchronize(), "synchronize"); }
#else
class StreamGuard {                                  // CPU build: kernels run in order on the host
public:
    explicit StreamGuard(int) {}
    ~StreamGuard() {}
};
inline int rip_side_stream(int) { return 0; }
inline void rip_stream_after(int, int) {}
template <class F> inline void par_for(long n, F f) {
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) f(i);
}
inline void device_sync() {}
#endif

HD inline void atomic_add(double* p, double v) {
#if defined(__CUDA_ARCH__)
    atomicAdd(p, v);
#elif defined(_OPENMP)
#pragma omp atomic
    *p += v;
#else
    *p += v;
#endif
}

HD inline int atomic_inc(int* p) {                 // returns the old value
#if defined(__CUDA_ARCH__)
    return atomicAdd(p, 1);
#else
    int old;
#if defined(_OPENMP)
#pragma omp atomic capture
#endif
    old = (*p)++;
    return old;
#endif
}

// max of non-negative doubles: their bit patterns are ordered like unsigned integers
HD inline void atomic_max_nonneg(double* p, double v) {
#if defined(__CUDA_ARCH__)
    atomicMax(reinterpret_cast<unsigned long long*>(p), static_cast<unsigned long long>(__double_as_longlong(v)));
#else
#if defined(_OPENMP)
#pragma omp critical(rip_atomic_max)
#endif
    { if (v > *p) *p = v; }
#endif
}

// ---------------------------------------------------------------- block-parallel loop with scratch
// par_blocks(nb, ns, body, flush): for every block b in [0, nb) a scratch array s of ns doubles is
// zeroed, body(b, t, nt, s) runs for the threads t in [0, nt) of the block, then (after a barrier)
// flush(b, t, nt, s). CUDA: one thread block per b, s in shared memory, threads concurrent (use
// block_add for s). CPU: one OpenMP iteration per b with nt = 1 and a thread-local s.
// body and flush must be HD lambdas like those of par_for.
HD inline void block_add(double* p, double v) {     // add into the block scratch
#if defined(__CUDA_ARCH__)
    atomicAdd(p, v);
#else
    *p += v;
#endif
}
// Integer (fixed-point) add into the block scratch as two 32-bit adds: a 64-bit atomicAdd in shared
// memory is a CAS loop (ATOMS.CAST.SPIN.64, fifth GPU check), the 32-bit one is native (ATOMS.ADD).
// The carry of the low word comes from its old value, so the sum of the carries is the carry of the
// sum in any order, and the result is the exact 64-bit sum (mod 2^64). Little endian: p[0] is the low word.
HD inline unsigned block_add_u32(unsigned* p, unsigned v) {     // returns the old value
#if defined(__CUDA_ARCH__)
    return atomicAdd(p, v);
#else
    const unsigned old = *p;
    *p = old + v;
    return old;
#endif
}
// CPU: one thread per block (par_blocks), so a plain 64-bit add; through memcpy, because the scratch is
// allocated as double and read back as long long: the two unsigned halves broke strict aliasing (g++ 15 -O3
// moved the read before the writes, unit test block_add).
HD inline void block_add(long long* p, long long v) {
#if defined(__CUDA_ARCH__)
    unsigned* w = reinterpret_cast<unsigned*>(p);
    const unsigned long long u = static_cast<unsigned long long>(v);
    const unsigned lo = unsigned(u), old = block_add_u32(w, lo);
    const unsigned hi = unsigned(u >> 32) + (unsigned(old + lo) < old ? 1u : 0u);
    if (hi != 0) block_add_u32(w + 1, hi);
#else
    unsigned long long s;
    std::memcpy(&s, p, sizeof s);
    s += static_cast<unsigned long long>(v);
    std::memcpy(p, &s, sizeof s);
#endif
}
#if RIP_GPU
// MinB: blocks of 256 threads per SM the register allocation must allow (__launch_bounds__), e.g. 3
// for <= 85 registers; 1 leaves the compiler free.
template <int MinB, class B, class F> __global__ void __launch_bounds__(256, MinB) rip_block_kernel(int ns, B body, F flush) {
    extern __shared__ double rip_shared[];
    for (int i = threadIdx.x; i < ns; i += blockDim.x) rip_shared[i] = 0.0;
    __syncthreads();
    body(long(blockIdx.x), int(threadIdx.x), int(blockDim.x), rip_shared);
    __syncthreads();
    flush(long(blockIdx.x), int(threadIdx.x), int(blockDim.x), rip_shared);
}
template <int MinB = 1, class B, class F> inline void par_blocks(long nb, int ns, B body, F flush) {
    if (nb <= 0) return;
    const int t = 256;
    const size_t bytes = size_t(ns) * sizeof(double);
    if (bytes > 48 * 1024)
        gpu_check(cudaFuncSetAttribute(rip_block_kernel<MinB, B, F>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)),
                  "shared memory size of par_blocks");
    rip_block_kernel<MinB><<<unsigned(nb), t, bytes>>>(ns, body, flush);
    gpu_check(cudaGetLastError(), "par_blocks launch");
}
#else
template <int MinB = 1, class B, class F> inline void par_blocks(long nb, int ns, B body, F flush) {
#pragma omp parallel
    {
        std::vector<double> s(size_t(ns > 0 ? ns : 1));
#pragma omp for schedule(dynamic)
        for (long b = 0; b < nb; ++b) {
            std::fill(s.begin(), s.end(), 0.0);
            body(b, 0, 1, s.data());
            flush(b, 0, 1, s.data());
        }
    }
}
#endif

// ---------------------------------------------------------------- device buffer
template <class T> class Buf {
public:
    Buf() = default;
    explicit Buf(size_t n) { resize(n); }
    ~Buf() { release(); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    Buf(Buf&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    Buf& operator=(Buf&& o) noexcept { if (this != &o) { release(); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; } return *this; }
    void resize(size_t n) {                        // contents are zeroed
        release(); n_ = n;
        if (n_ == 0) return;
#if RIP_GPU
        gpu_check(cudaMalloc(&p_, n_ * sizeof(T)), "cudaMalloc");
#else
        p_ = static_cast<T*>(::operator new(n_ * sizeof(T)));
#endif
        zero();
    }
    void grow_keep(size_t n) {                     // capacity at least n, contents kept
        if (n <= n_) return;
        Buf<T> b(n);
        b.copy_from(*this, 0, 0, n_);
        *this = std::move(b);
    }
    void zero() {
        if (!n_) return;
#if RIP_GPU
        gpu_check(cudaMemset(p_, 0, n_ * sizeof(T)), "cudaMemset");
#else
        std::memset(static_cast<void*>(p_), 0, n_ * sizeof(T));
#endif
    }
    void copy_from(const Buf<T>& o, size_t dst_off, size_t src_off, size_t cnt) {   // device to device
        if (!cnt) return;
#if RIP_GPU
        gpu_check(cudaMemcpy(p_ + dst_off, o.p_ + src_off, cnt * sizeof(T), cudaMemcpyDeviceToDevice), "cudaMemcpy D2D");
#else
        std::memmove(static_cast<void*>(p_ + dst_off), o.p_ + src_off, cnt * sizeof(T));
#endif
    }
    T* data() { return p_; }
    const T* data() const { return p_; }
    size_t size() const { return n_; }
    void swap(Buf<T>& o) { std::swap(p_, o.p_); std::swap(n_, o.n_); }
    void to_host(T* h, size_t off, size_t cnt) const {
        if (!cnt) return;
#if RIP_GPU
        gpu_check(cudaMemcpy(h, p_ + off, cnt * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
#else
        std::memcpy(static_cast<void*>(h), p_ + off, cnt * sizeof(T));
#endif
    }
    void from_host(const T* h, size_t off, size_t cnt) {
        if (!cnt) return;
#if RIP_GPU
        gpu_check(cudaMemcpy(p_ + off, h, cnt * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
#else
        std::memcpy(static_cast<void*>(p_ + off), h, cnt * sizeof(T));
#endif
    }
#if RIP_GPU
    // The same in the stream s (asynchronous; with pinned host memory). The CPU build has no streams.
    void to_host_async(T* h, size_t off, size_t cnt, cudaStream_t s) const {
        if (cnt) gpu_check(cudaMemcpyAsync(h, p_ + off, cnt * sizeof(T), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync D2H");
    }
    void from_host_async(const T* h, size_t off, size_t cnt, cudaStream_t s) {
        if (cnt) gpu_check(cudaMemcpyAsync(p_ + off, h, cnt * sizeof(T), cudaMemcpyHostToDevice, s), "cudaMemcpyAsync H2D");
    }
    void copy_from_async(const Buf<T>& o, size_t dst_off, size_t src_off, size_t cnt, cudaStream_t s) {
        if (cnt) gpu_check(cudaMemcpyAsync(p_ + dst_off, o.p_ + src_off, cnt * sizeof(T), cudaMemcpyDeviceToDevice, s), "cudaMemcpyAsync D2D");
    }
#endif
    std::vector<T> host() const { std::vector<T> v(n_); to_host(v.data(), 0, n_); return v; }
    void upload(const std::vector<T>& v) { if (v.size() != n_) resize(v.size()); from_host(v.data(), 0, v.size()); }
private:
    void release() {
        if (!p_) return;
#if RIP_GPU
        cudaFree(p_);
#else
        ::operator delete(p_);
#endif
        p_ = nullptr; n_ = 0;
    }
    T* p_ = nullptr;
    size_t n_ = 0;
};

// Host buffer that only grows: pinned (page-locked) in the GPU build, so device copies to and from it run at
// full speed and need no staging inside the driver; ordinary memory in the CPU build. Contents are not kept on growth.
template <class T> class HostBuf {
public:
    HostBuf() = default;
    ~HostBuf() { release(); }
    HostBuf(const HostBuf&) = delete;
    HostBuf& operator=(const HostBuf&) = delete;
    T* ensure(size_t n) {
        if (n > n_) {
            release();
            n_ = std::max(n, n_ + n_ / 2);
#if RIP_GPU
            gpu_check(cudaMallocHost(&p_, n_ * sizeof(T)), "cudaMallocHost");
#else
            p_ = static_cast<T*>(::operator new(n_ * sizeof(T)));
#endif
        }
        return p_;
    }
    T* data() { return p_; }
    size_t size() const { return n_; }
private:
    void release() {
        if (!p_) return;
#if RIP_GPU
        cudaFreeHost(p_);
#else
        ::operator delete(p_);
#endif
        p_ = nullptr; n_ = 0;
    }
    T* p_ = nullptr;
    size_t n_ = 0;
};

// ---------------------------------------------------------------- reductions (one scalar)
#if RIP_GPU
// Block-wise tree reduction: every block reduces its grid-stride share in shared memory and writes one
// partial; the host combines the partials in block order. No atomics on one address (that cost 656 us
// per call for 1 M values, fifth GPU check), no allocation per call, and the result does not depend
// on the scheduling. Both reductions start from 0 (sums; maxima of non-negative values).
struct RipSum { HD double operator()(double a, double b) const { return a + b; } };
struct RipMax { HD double operator()(double a, double b) const { return a > b ? a : b; } };
constexpr int rip_reduce_threads = 256, rip_reduce_blocks = 1024;
template <class F, class Op> __global__ void rip_reduce_kernel(long n, F f, Op op, double* part) {
    __shared__ double s[rip_reduce_threads];
    double v = 0;
    for (long i = long(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += long(gridDim.x) * blockDim.x) v = op(v, f(i));
    s[threadIdx.x] = v;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (int(threadIdx.x) < k) s[threadIdx.x] = op(s[threadIdx.x], s[threadIdx.x + k]);
        __syncthreads();
    }
    if (threadIdx.x == 0) part[blockIdx.x] = s[0];
}
inline Buf<double>& rip_reduce_partials() {          // kept for the whole run (never freed)
    static Buf<double>* b = new Buf<double>(size_t(rip_reduce_blocks));
    return *b;
}
template <class F, class Op> inline double rip_reduce(long n, F f, Op op) {
    if (n <= 0) return 0;
    const int nb = int(std::min<long>(rip_reduce_blocks, (n + rip_reduce_threads - 1) / rip_reduce_threads));
    Buf<double>& part = rip_reduce_partials();
    rip_reduce_kernel<<<nb, rip_reduce_threads>>>(n, f, op, part.data());
    gpu_check(cudaGetLastError(), "reduction launch");
    double h[rip_reduce_blocks];
    part.to_host(h, 0, size_t(nb));
    double r = 0;
    for (int b = 0; b < nb; ++b) r = op(r, h[b]);
    return r;
}
#endif
template <class F> inline double par_sum(long n, F f) {
#if RIP_GPU
    return rip_reduce(n, f, RipSum{});
#else
    double s = 0;
#pragma omp parallel for reduction(+ : s) schedule(static)
    for (long i = 0; i < n; ++i) s += f(i);
    return s;
#endif
}
template <class F> inline double par_max_abs(long n, F f) {
#if RIP_GPU
    return rip_reduce(n, [=] HD(long i) { return fabs(f(i)); }, RipMax{});
#else
    double m = 0;
#pragma omp parallel for reduction(max : m) schedule(static)
    for (long i = 0; i < n; ++i) m = std::max(m, std::fabs(f(i)));
    return m;
#endif
}
