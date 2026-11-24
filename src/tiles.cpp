#include "tiles.hpp"
#include <stdexcept>
#if RIP_GPU
#include <cub/cub.cuh>
#endif

TileGrid tile_grid(const Domain& d, const int cells[3]) {
    TileGrid g;
    for (int a = 0; a < 3; ++a) {
        if (cells[a] < 1) throw std::runtime_error("numerics.tile: cells per tile must be >= 1");
        g.T[a] = cells[a];
        g.nt[a] = (d.n[a] + cells[a] - 1) / cells[a];
    }
    return g;
}

// key[p] = tile of particle p (cell index clamped to the owned box)
void tile_keys_kernel(const double* x, const double* y, const double* z, double lo0, double lo1, double lo2,
                      double ih0, double ih1, double ih2, TileGrid g, int n0, int n1, int n2, int* key, long n) {
    par_for(n, [=] HD(long p) {
        int i = int(floor((x[p] - lo0) * ih0)), j = int(floor((y[p] - lo1) * ih1)), k = int(floor((z[p] - lo2) * ih2));
        i = i < 0 ? 0 : (i >= n0 ? n0 - 1 : i);
        j = j < 0 ? 0 : (j >= n1 ? n1 - 1 : j);
        k = k < 0 ? 0 : (k >= n2 ? n2 - 1 : k);
        key[p] = int((long(k / g.T[2]) * g.nt[1] + j / g.T[1]) * g.nt[0] + i / g.T[0]);
    });
}

#if RIP_GPU
// Device scratch of the sort, kept between calls (never freed).
struct SortScratch { Buf<int> key, key2, idx, idx2, cnt; Buf<unsigned char> tmp; };
static SortScratch& sort_scratch() {
    static SortScratch* s = new SortScratch;
    return *s;
}
static void ensure(Buf<int>& b, size_t n) { if (b.size() < n) b.resize(n); }
#endif

// Stable sort by tile. GPU: CUB's radix sort of (key, index) pairs, which is stable, so the result is
// the same permutation as the host counting sort. The tile starts come from the sorted keys: each
// position q writes start[t] = q for the tiles t in (key[q-1], key[q]], with no atomics (the seventh
// GPU check: int atomics on the counts cost 10-20x the sort). No per-call allocation, and only the
// nt + 1 starts cross to the host. CPU: counting sort.
void sort_by_tile(Species& s, const Domain& d, const TileGrid& g) {
    const long n = long(s.size()), nt = g.count();
    std::vector<int> start(size_t(nt) + 1, 0);
    if (n == 0) { s.tile_start = std::move(start); return; }
#if RIP_GPU
    SortScratch& W = sort_scratch();
    ensure(W.key, size_t(n)); ensure(W.key2, size_t(n)); ensure(W.idx, size_t(n)); ensure(W.idx2, size_t(n));
    ensure(W.cnt, size_t(nt) + 1);
    int* K = W.key.data(); int* I = W.idx.data(); int* C = W.cnt.data();
    tile_keys_kernel(s.ptr<double>(Species::X), s.ptr<double>(Species::Y), s.ptr<double>(Species::Z), d.lo(0),
                     d.lo(1), d.lo(2), 1.0 / d.h[0], 1.0 / d.h[1], 1.0 / d.h[2], g, d.n[0], d.n[1], d.n[2], K, n);
    par_for(n, [=] HD(long p) { I[p] = int(p); });
    int bits = 1;
    while ((1L << bits) < nt) ++bits;
    size_t bytes = 0;
    gpu_check(cub::DeviceRadixSort::SortPairs(nullptr, bytes, K, W.key2.data(), I, W.idx2.data(), int(n), 0, bits),
              "cub sort (size)");
    if (W.tmp.size() < bytes) W.tmp.resize(bytes);
    gpu_check(cub::DeviceRadixSort::SortPairs(W.tmp.data(), bytes, K, W.key2.data(), I, W.idx2.data(), int(n), 0, bits),
              "cub sort");
    const int* K2 = W.key2.data();
    par_for(n, [=] HD(long q) {
        const int k = K2[q], prev = q ? K2[q - 1] : -1;
        for (int t = prev + 1; t <= k; ++t) C[t] = int(q);
        if (q == n - 1)
            for (long t = k + 1; t <= nt; ++t) C[t] = int(n);
    });
    W.cnt.to_host(start.data(), 0, size_t(nt) + 1);
    const int* P = W.idx2.data();
    const double moved = par_sum(n, [=] HD(long q) { return P[q] != q ? 1.0 : 0.0; });
    if (moved > 0) s.permute_device(P);
#else
    std::vector<int> key{};
    key.resize(size_t(n));
    {
        Buf<int> dk{size_t(n)};
        tile_keys_kernel(s.ptr<double>(Species::X), s.ptr<double>(Species::Y), s.ptr<double>(Species::Z), d.lo(0),
                         d.lo(1), d.lo(2), 1.0 / d.h[0], 1.0 / d.h[1], 1.0 / d.h[2], g, d.n[0], d.n[1], d.n[2],
                         dk.data(), n);
        dk.to_host(key.data(), 0, size_t(n));
    }
    // stable counting sort on the host
    for (long p = 0; p < n; ++p) ++start[size_t(key[p]) + 1];
    for (long t = 0; t < nt; ++t) start[size_t(t) + 1] += start[size_t(t)];
    std::vector<int> perm(static_cast<size_t>(n)), next(start.begin(), start.end() - 1);
    for (long p = 0; p < n; ++p) perm[size_t(next[size_t(key[p])]++)] = int(p);
    bool sorted = true;
    for (long q = 0; q < n && sorted; ++q) sorted = perm[size_t(q)] == q;
    if (!sorted) s.permute(perm);
#endif
    s.tile_start = std::move(start);
}

void tile_colours(const TileGrid& g, std::vector<int>& order, std::vector<int>& start) {
    order.clear();
    start.assign(9, 0);
    for (int col = 0; col < 8; ++col) {
        start[col] = int(order.size());
        for (int tz = col >> 2 & 1; tz < g.nt[2]; tz += 2)
            for (int ty = col >> 1 & 1; ty < g.nt[1]; ty += 2)
                for (int tx = col & 1; tx < g.nt[0]; tx += 2) order.push_back((tz * g.nt[1] + ty) * g.nt[0] + tx);
    }
    start[8] = int(order.size());
}

bool tiled_deposit_safe(const TileGrid& g, const Domain& d) {
    for (int a = 0; a < 3; ++a) {
        if (d.dt() > d.h[a]) return false;
        if (g.nt[a] >= 3 && g.T[a] < 6) return false;
    }
    return true;
}

int min_tiles_per_colour(const TileGrid& g) {
    std::vector<int> order, start;
    tile_colours(g, order, start);
    int m = 0;
    for (int col = 0; col < 8; ++col) {
        const int k = start[col + 1] - start[col];
        if (k > 0 && (m == 0 || k < m)) m = k;
    }
    return m;
}
