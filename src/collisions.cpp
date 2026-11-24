// Binary Coulomb collisions (collisions.hpp, docs/specs/collisions.md, decision 94).
#include "collisions.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include "collision_kernels.hpp"
#include "impact.hpp"
#include "fsutil.hpp"
#include "sim.hpp"
#if RIP_GPU
#include <cub/cub.cuh>
#endif

namespace {
constexpr double RE_SI = 2.8179403262e-15;           // classical electron radius [m]
constexpr double LCBAR_SI = 3.8615926796e-13;        // reduced Compton wavelength hbar / (m_e c) [m]
constexpr int MAXS = 16;                             // species in the Debye sum at most

// The cells of the owned box (POD for kernels). Cart3D / 2D: absolute cell index floor(x / h) minus the first owned one (the
// same on any decomposition); RZ: z cells and the radial cells between the nodes of the radial grid.
struct CellMap {
    int n0 = 1, n1 = 1, n2 = 1;
    long c0 = 0, c1 = 0, c2 = 0;
    double ih0 = 1, ih1 = 1, ih2 = 1, xfreeze = 1e300, ivol = 1;
    int rz = 0, flat = 0;
    const double* ivr = nullptr;     // RZ: 1 / V of radial cell j
    RadialView g{};
    long count() const { return long(n0) * n1 * n2; }
    HD long cell(double x, double y, double z) const {
        if (!(x < xfreeze)) return -1;
        const long i = long(floor(x * ih0)) - c0;
        if (i < 0 || i >= n0) return -1;
        if (rz) {
            double u;
            const int j = rz_cell(g, sqrt(y * y + z * z), u);
            if (j < 0 || j >= n1) return -1;
            return long(j) * n0 + i;
        }
        const long j = long(floor(y * ih1)) - c1;
        if (j < 0 || j >= n1) return -1;
        long k = 0;
        if (!flat) {
            k = long(floor(z * ih2)) - c2;
            if (k < 0 || k >= n2) return -1;
        }
        return (k * n1 + j) * n0 + i;
    }
    HD double inv_volume(long c) const { return rz ? ivr[c / n0] : ivol; }
    HD unsigned long long key(long c) const {           // of the absolute cell
        const long i = c % n0, j = (c / n0) % n1, k = c / (long(n0) * n1);
        return ion_mix(ion_mix(ion_mix((unsigned long long)(c0 + i)) ^ (unsigned long long)(c1 + j)) ^ (unsigned long long)(c2 + k));
    }
};

// One species in the kernels: momenta, weights, charge states, and its cell lists (particles of cell c: I[S[c] .. S[c+1]))
struct SpView {
    double *ux = nullptr, *uy = nullptr, *uz = nullptr;
    const double* w = nullptr;
    int* zs = nullptr;               // charge state per particle (null: charge q)
    const int *I = nullptr, *S = nullptr;
    double q = 0, m = 1;
};
struct SpSet { SpView v[MAXS]; int n = 0; };

// impact ionization in collide_block: mode 0 off, 1 count the new electrons per cell (cnt), 2 write them (5 doubles each from
// off[c]: u_x, u_y, u_z, the ion index, the level); efirst: species A of the block is the electron species
struct IonArgs {
    int mode = 0, efirst = 1;
    ImpactTable T;
    const double* ip = nullptr;      // I_k [m c^2]
    int* cnt = nullptr;
    const int* off = nullptr;
    double* B = nullptr;
};

struct CollList {                    // the cell lists of one species
    Buf<int> key, key2, idx, idx2, start;
    Buf<unsigned long long> h;
};
constexpr int ACC = 9;               // per cell and block: pairs, sum s, sum lnL, dE, dPx, dPy, dPz, impact events, sum w_i I
}  // namespace

struct CollBlock {
    CollisionConfig c;
    int a = -1, b = -1;              // species indices
    Buf<double> acc;                 // per cell (cumulative), ACC values
    // ionizing: the species of the new electrons, the tables of the ion species, scratch of the two passes
    int el = -1;
    bool efirst = true;
    Buf<double> cs, te, le, ip, births, save;
    Buf<int> cnt, off, zsave;
    int zn = 0;
};

struct Collisions {
    std::vector<CollBlock> blocks;
    std::vector<int> used;           // species with cell lists
    std::vector<CollList> lists;     // per species index
    bool debye = false;              // a block with the automatic Coulomb log
    Buf<double> debye2;              // per cell
    Buf<double> ivr;                 // RZ
    CollParams P;
    Buf<unsigned char> tmp;
    FILE* f = nullptr;
    ~Collisions() { if (f) std::fclose(f); }
};

namespace {
template <class T> void ensure(Buf<T>& b, size_t n) { if (b.size() < n) b.resize(std::max(n, b.size() + b.size() / 2)); }

CellMap cell_map(Simulation& s, Collisions& C) {
    CellMap M;
    const Domain& d = s.dom;
    M.n0 = d.n[0]; M.c0 = d.cell0(0); M.ih0 = 1.0 / d.h[0]; M.xfreeze = d.xfreeze;
    if (s.cfg.rz) {
        M.rz = 1;
        M.g = s.rz->grid().view();
        M.n1 = M.g.nr;
        M.ivr = C.ivr.data();
    } else {
        M.n1 = d.n[1]; M.c1 = d.cell0(1); M.ih1 = 1.0 / d.h[1];
        M.flat = d.flat ? 1 : 0;
        M.n2 = d.flat ? 1 : d.n[2]; M.c2 = d.flat ? 0 : d.cell0(2); M.ih2 = 1.0 / d.h[2];
        M.ivol = 1.0 / d.cell_volume();
    }
    return M;
}

// ids of the particles that have none: a hash of (species, position) (the load position; as field ionization)
void assign_ids(Species& sp, int index) {
    long long* ID = sp.ptr<long long>(sp.slot("id"));
    const double *X = sp.ptr<double>(Species::X), *Y = sp.ptr<double>(Species::Y), *Z = sp.ptr<double>(Species::Z);
    const unsigned long long base = ion_mix(0x5eedULL + (unsigned long long)index);
    par_for(long(sp.size()), [=] HD(long p) {
        if (ID[p] != 0) return;
        unsigned long long h = base, b;
        double v = X[p]; memcpy(&b, &v, 8); h = ion_mix(h ^ b);
        v = Y[p]; memcpy(&b, &v, 8); h = ion_mix(h ^ b);
        v = Z[p]; memcpy(&b, &v, 8); h = ion_mix(h ^ b);
        ID[p] = (long long)(h == 0 ? 1 : h);
    });
}

// The cell lists of sp: particles sorted by cell, within a cell by the hash of (hk, id) (a random order that does not depend on
// the storage order); cells c in [0, nc), the particles outside in the bucket nc.
void build_list(Species& sp, const CellMap& M, unsigned long long hk, CollList& L, Buf<unsigned char>& tmp) {
    const long n = long(sp.size()), nc = M.count();
    ensure(L.start, size_t(nc) + 2);
    int* S = L.start.data();
    if (n == 0) { L.start.zero(); return; }
    ensure(L.key, size_t(n)); ensure(L.idx, size_t(n)); ensure(L.h, size_t(n));
    int* K = L.key.data(); int* I = L.idx.data(); unsigned long long* H = L.h.data();
    const double *X = sp.ptr<double>(Species::X), *Y = sp.ptr<double>(Species::Y), *Z = sp.ptr<double>(Species::Z);
    const long long* ID = sp.ptr<long long>(sp.slot("id"));
    const CellMap m = M;
    par_for(n, [=] HD(long p) {
        const long c = m.cell(X[p], Y[p], Z[p]);
        K[p] = int(c < 0 ? nc : c);
        H[p] = ion_mix(hk ^ ion_mix((unsigned long long)ID[p]));
        I[p] = int(p);
    });
#if RIP_GPU
    ensure(L.key2, size_t(n)); ensure(L.idx2, size_t(n));
    int bits = 1;
    while ((1L << bits) < nc + 1) ++bits;
    size_t bytes = 0;
    gpu_check(cub::DeviceRadixSort::SortPairs(nullptr, bytes, K, L.key2.data(), I, L.idx2.data(), int(n), 0, bits), "cub sort (size)");
    ensure(tmp, bytes);
    gpu_check(cub::DeviceRadixSort::SortPairs(tmp.data(), bytes, K, L.key2.data(), I, L.idx2.data(), int(n), 0, bits), "cub sort");
    L.key.swap(L.key2); L.idx.swap(L.idx2);
    const int* K2 = L.key.data();
    par_for(n, [=] HD(long q) {
        const int k = K2[q], prev = q ? K2[q - 1] : -1;
        for (int t = prev + 1; t <= k; ++t) S[t] = int(q);
        if (q == n - 1)
            for (long t = k + 1; t <= nc + 1; ++t) S[t] = int(n);
    });
    I = L.idx.data();
#else
    (void)tmp;
    std::fill(S, S + nc + 2, 0);
    for (long p = 0; p < n; ++p) ++S[K[p] + 1];
    for (long c = 0; c <= nc; ++c) S[c + 1] += S[c];
    ensure(L.idx2, size_t(n));
    int* O = L.idx2.data();
    std::vector<int> next(S, S + nc + 1);
    for (long p = 0; p < n; ++p) O[next[size_t(K[p])]++] = int(p);
    L.idx.swap(L.idx2);
    I = L.idx.data();
#endif
    const unsigned long long* HH = H;
    par_for(nc, [=] HD(long c) { coll_heap_sort(I + S[c], S[c + 1] - S[c], HH); });
}

SpView view_of(Species& sp, CollList& L) {
    SpView v;
    v.ux = sp.ptr<double>(Species::UX); v.uy = sp.ptr<double>(Species::UY); v.uz = sp.ptr<double>(Species::UZ);
    v.w = sp.ptr<double>(Species::W);
    v.zs = sp.zslot >= 0 ? sp.ptr<int>(sp.zslot) : nullptr;
    v.I = L.idx.data(); v.S = L.start.data();
    v.q = sp.q; v.m = sp.m;
    return v;
}

// The squared Debye length per cell (Perez; as Smilei 5.1): 1 / lambda_D^2 = sum over the species of n <q>^2 / T, T = m <u^2 / gamma> / 3
// (m c^2), at least the interatomic distance (3 / (4 pi n_max))^(1/3); 0 in empty cells.
void debye_lengths(const CellMap M, const SpSet set, double kre, double* D2) {
    const double cf = 1.0 / (3 * kre);
    par_for(M.count(), [=] HD(long c) {
        const double iv = M.inv_volume(c);
        double inv = 0, nmax = 0;
        for (int t = 0; t < set.n; ++t) {
            const SpView& V = set.v[t];
            const int a = V.S[c], e = V.S[c + 1];
            if (e <= a) continue;
            double dens = 0, ch = 0, T = 0;
            for (int q = a; q < e; ++q) {
                const int p = V.I[q];
                const double u2 = V.ux[p] * V.ux[p] + V.uy[p] * V.uy[p] + V.uz[p] * V.uz[p];
                const double qp = V.zs ? V.q * V.zs[p] : V.q;
                dens += V.w[p]; ch += V.w[p] * qp; T += V.w[p] * u2 / sqrt(1 + u2);
            }
            if (!(dens > 0)) continue;
            ch /= dens; T *= V.m / (3 * dens); dens *= iv;
            inv += T == 0 ? 1e100 : dens * ch * ch / T;
            if (dens > nmax) nmax = dens;
        }
        double d2 = 0;
        if (inv > 0) {
            d2 = 1 / inv;
            const double r = cbrt(cf * nmax), rmin2 = 1 / (r * r);
            if (d2 < rmin2) d2 = rmin2;
        }
        D2[c] = d2;
    });
}

// The collisions of one block in every cell (one thread per cell, the pairs in sequence)
void collide_block(const CellMap M, const SpView A, const SpView B, bool intra, double dte, const CollParams P, const double* D2,
                   unsigned long long bk, double* acc, const IonArgs X) {
    par_for(M.count(), [=] HD(long c) {
        int nborn = 0;
        if (X.mode == 1) X.cnt[c] = 0;
        const int a0 = A.S[c], na = A.S[c + 1] - a0, b0 = B.S[c], nb = B.S[c + 1] - b0;
        long npairs, nnr, ncorr;
        double wc1, wc2;
        if (intra) {
            if (na < 2) return;
            npairs = (na + 1) / 2; nnr = na - npairs; ncorr = 2 * npairs - 1; wc1 = 1; wc2 = 0.5;
        } else {
            if (na == 0 || nb == 0) return;
            npairs = na > nb ? na : nb; nnr = na > nb ? nb : na; ncorr = npairs;
            const long k = npairs / nnr;
            wc1 = 1.0 / double(k); wc2 = 1.0 / double(k + 1);
        }
        const double iv = M.inv_volume(c);
        double n1 = 0, n2 = 0;
        for (int q = a0; q < a0 + na; ++q) n1 += A.w[A.I[q]];
        if (intra) n2 = n1;
        else for (int q = b0; q < b0 + nb; ++q) n2 += B.w[B.I[q]];
        n1 *= iv; n2 *= iv;
        const double c1 = cbrt(n1), c2 = cbrt(n2), n1_23 = c1 * c1, n2_23 = c2 * c2;
        const double dtc = dte * double(ncorr) * iv;
        const double d2 = D2 ? D2[c] : 0.0;
        const unsigned long long ck = ion_mix(bk ^ M.key(c));
        double out[ACC] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (long i = 0; i < npairs; ++i) {
            int ia, ib;
            if (intra) { ia = A.I[a0 + (2 * i) % na]; ib = A.I[a0 + (2 * i + 1) % na]; }
            else if (na > nb) { ia = A.I[a0 + i]; ib = B.I[b0 + i % nb]; }
            else { ia = A.I[a0 + i % na]; ib = B.I[b0 + i]; }
            const double wa = A.w[ia], wb = B.w[ib];
            const double wmin = wa < wb ? wa : wb, wmax = wa < wb ? wb : wa;
            if (!(wmin > 0)) continue;
            const double D = wmax * dtc * ((i % nnr) < (npairs % nnr) ? wc2 : wc1);
            const unsigned long long key = ion_mix(ck ^ ion_mix((unsigned long long)i));
            const double U1 = ion_uniform(key), U2 = ion_uniform(key + 1), U3 = ion_uniform(key + 2);
            const bool up1 = U3 * wa < wb, up2 = U3 * wb < wa;
            double u1[3] = {A.ux[ia], A.uy[ia], A.uz[ia]}, u2[3] = {B.ux[ib], B.uy[ib], B.uz[ib]};
            const double g1 = sqrt(1 + u1[0] * u1[0] + u1[1] * u1[1] + u1[2] * u1[2]);
            const double g2 = sqrt(1 + u2[0] * u2[0] + u2[1] * u2[1] + u2[2] * u2[2]);
            const double qa = A.zs ? A.q * A.zs[ia] : A.q, qb = B.zs ? B.q * B.zs[ib] : B.q;
            double lnL;
            const double s = coll_pair(u1, u2, A.m, B.m, qa, qb, D, n1_23, n2_23, d2, P, U1, U2, up1, up2, lnL);
            if (s > 0) {
                out[0] += 1; out[1] += s; out[2] += lnL;
                if (up1) {
                    const double h1 = sqrt(1 + u1[0] * u1[0] + u1[1] * u1[1] + u1[2] * u1[2]);
                    out[3] += wa * A.m * (h1 - g1);
                    out[4] += wa * A.m * (u1[0] - A.ux[ia]); out[5] += wa * A.m * (u1[1] - A.uy[ia]); out[6] += wa * A.m * (u1[2] - A.uz[ia]);
                    A.ux[ia] = u1[0]; A.uy[ia] = u1[1]; A.uz[ia] = u1[2];
                }
                if (up2) {
                    const double h2 = sqrt(1 + u2[0] * u2[0] + u2[1] * u2[1] + u2[2] * u2[2]);
                    out[3] += wb * B.m * (h2 - g2);
                    out[4] += wb * B.m * (u2[0] - B.ux[ib]); out[5] += wb * B.m * (u2[1] - B.uy[ib]); out[6] += wb * B.m * (u2[2] - B.uz[ib]);
                    B.ux[ib] = u2[0]; B.uy[ib] = u2[1]; B.uz[ib] = u2[2];
                }
            }
            if (X.mode) {                             // impact ionization of the pair (impact.hpp), after the deflection as in Smilei
                const SpView& E = X.efirst ? A : B;
                const SpView& I = X.efirst ? B : A;
                const int ie = X.efirst ? ia : ib, ii = X.efirst ? ib : ia;
                double ue[3] = {E.ux[ie], E.uy[ie], E.uz[ie]};
                const double ui[3] = {I.ux[ii], I.uy[ii], I.uz[ii]};
                const double ge0 = sqrt(1 + ue[0] * ue[0] + ue[1] * ue[1] + ue[2] * ue[2]);
                int z = I.zs[ii];
                double eion;
                const int mode = X.mode, at = X.mode == 2 ? X.off[c] : 0;
                double* BB = X.B;
                auto born = [&](const double un[3], int lev) {
                    if (mode == 2) {
                        double* b = BB + 5 * long(at + nborn);
                        b[0] = un[0]; b[1] = un[1]; b[2] = un[2]; b[3] = double(ii); b[4] = double(lev);
                    }
                    ++nborn;
                };
                const int ni = impact_pair(X.T, X.ip, ue, ui, z, E.w[ie], I.w[ii], D, ion_uniform(key + 3), ion_uniform(key + 4), born, eion);
                I.zs[ii] = z;
                const double ge1 = sqrt(1 + ue[0] * ue[0] + ue[1] * ue[1] + ue[2] * ue[2]);
                out[3] += E.w[ie] * E.m * (ge1 - ge0);
                E.ux[ie] = ue[0]; E.uy[ie] = ue[1]; E.uz[ie] = ue[2];
                out[7] += ni; out[8] += eion * I.w[ii];
            }
        }
        if (X.mode == 1) { X.cnt[c] = nborn; return; }
        for (int t = 0; t < ACC; ++t) acc[ACC * c + t] += out[t];
    });
}

// O[0 .. n) = exclusive sum of C; returns the total
long exclusive_scan(const int* C, int* O, long n, Buf<unsigned char>& tmp) {
    if (n <= 0) return 0;
#if RIP_GPU
    size_t bytes = 0;
    gpu_check(cub::DeviceScan::ExclusiveSum(nullptr, bytes, C, O, int(n)), "cub scan (size)");
    ensure(tmp, bytes);
    gpu_check(cub::DeviceScan::ExclusiveSum(tmp.data(), bytes, C, O, int(n)), "cub scan");
    int last[2] = {0, 0};
    gpu_check(cudaMemcpy(&last[0], O + n - 1, sizeof(int), cudaMemcpyDeviceToHost), "scan last");
    gpu_check(cudaMemcpy(&last[1], C + n - 1, sizeof(int), cudaMemcpyDeviceToHost), "scan last");
    return long(last[0]) + last[1];
#else
    (void)tmp;
    long t = 0;
    for (long q = 0; q < n; ++q) { O[q] = int(t); t += C[q]; }
    return t;
#endif
}

// the momenta of the two species and the charge states of the ions, kept across the counting pass
void save_state(Species& a, Species& b, Species& ion, CollBlock& B, bool restore) {
    const long na = long(a.size()), nb = long(b.size()), ni = long(ion.size());
    if (!restore) { ensure(B.save, size_t(3 * (na + nb))); ensure(B.zsave, size_t(ni)); }
    double* S = B.save.data();
    int* ZS = B.zsave.data();
    for (int t = 0; t < 2; ++t) {
        Species& sp = t ? b : a;
        double *X = sp.ptr<double>(Species::UX), *Y = sp.ptr<double>(Species::UY), *Z = sp.ptr<double>(Species::UZ);
        double* o = S + (t ? 3 * na : 0);
        if (restore) par_for(t ? nb : na, [=] HD(long p) { X[p] = o[3 * p]; Y[p] = o[3 * p + 1]; Z[p] = o[3 * p + 2]; });
        else par_for(t ? nb : na, [=] HD(long p) { o[3 * p] = X[p]; o[3 * p + 1] = Y[p]; o[3 * p + 2] = Z[p]; });
    }
    int* Zs = ion.ptr<int>(ion.zslot);
    if (restore) par_for(ni, [=] HD(long p) { Zs[p] = ZS[p]; });
    else par_for(ni, [=] HD(long p) { ZS[p] = Zs[p]; });
}

// the new electrons of an ionizing block (births: u, ion index, level) at the ions' positions, with the ion's weight and an id
// from the ion's id and the level (as field ionization)
void append_births(Species& el, Species& ion, const double* Bd, long n) {
    if (n == 0) return;
    const long first = long(el.extend(size_t(n)));
    double *EX = el.ptr<double>(Species::X), *EY = el.ptr<double>(Species::Y), *EZ = el.ptr<double>(Species::Z);
    double *EUX = el.ptr<double>(Species::UX), *EUY = el.ptr<double>(Species::UY), *EUZ = el.ptr<double>(Species::UZ);
    double* EW = el.ptr<double>(Species::W);
    long long* EID = el.slot("id") >= 0 ? el.ptr<long long>(el.slot("id")) : nullptr;
    const double *X = ion.ptr<double>(Species::X), *Y = ion.ptr<double>(Species::Y), *Z = ion.ptr<double>(Species::Z);
    const double* W = ion.ptr<double>(Species::W);
    const long long* ID = ion.ptr<long long>(ion.slot("id"));
    par_for(n, [=] HD(long q) {
        const double* b = Bd + 5 * q;
        const long p = long(b[3]), o = first + q;
        EX[o] = X[p]; EY[o] = Y[p]; EZ[o] = Z[p];
        EUX[o] = b[0]; EUY[o] = b[1]; EUZ[o] = b[2];
        EW[o] = W[p];
        if (EID) EID[o] = (long long)(ion_mix((unsigned long long)ID[p] ^ ion_mix(0xe1ec0000ULL + (unsigned long long)b[4])) | 1ULL);
    });
}

void apply_all(Simulation& s, bool force) {
    Collisions& C = *s.collisions;
    bool any = false;
    for (const CollBlock& b : C.blocks) any = any || force || s.istep % b.c.every == 0;
    if (!any) return;
    const CellMap M = cell_map(s, C);
    const long nc = M.count();
    const unsigned long long sk = ion_mix((unsigned long long)s.cfg.seed * 0x9E3779B97F4A7C15ULL + (unsigned long long)s.istep);
    for (int i : C.used) {
        assign_ids(s.species[size_t(i)], i);
        build_list(s.species[size_t(i)], M, ion_mix(sk ^ ion_mix(0xc011ULL + (unsigned long long)i)), C.lists[size_t(i)], C.tmp);
    }
    if (C.debye) {
        ensure(C.debye2, size_t(nc));
        SpSet set;
        for (int i : C.used) {
            Species& sp = s.species[size_t(i)];
            if (sp.q == 0 && sp.zslot < 0) continue;
            set.v[set.n++] = view_of(sp, C.lists[size_t(i)]);
        }
        debye_lengths(M, set, C.P.kre, C.debye2.data());
    }
    for (size_t k = 0; k < C.blocks.size(); ++k) {
        CollBlock& b = C.blocks[k];
        if (!force && s.istep % b.c.every != 0) continue;
        if (b.acc.size() < size_t(ACC * nc)) b.acc.grow_keep(size_t(ACC * nc));
        CollParams P = C.P;
        P.lnL = b.c.coulomb_log; P.angle = b.c.angle;
        if (!b.c.cap) P.lt = 0;
        if (b.c.hquantum) P.bq *= 2 * M_PI;
        Species &sa = s.species[size_t(b.a)], &sb = s.species[size_t(b.b)];
        const SpView A = view_of(sa, C.lists[size_t(b.a)]);
        const SpView B = view_of(sb, C.lists[size_t(b.b)]);
        const unsigned long long bk = ion_mix(sk ^ ion_mix(0xb10cULL + k));
        const double dte = double(b.c.every) * s.dom.dt();
        const double* D2 = P.lnL > 0 ? nullptr : C.debye2.data();
        IonArgs X;
        if (b.c.ionizing) {
            // two passes with the same random numbers: count the new electrons per cell (on a saved state), then write them
            Species& ion = b.efirst ? sb : sa;
            X.efirst = b.efirst ? 1 : 0;
            X.T.cs = b.cs.data(); X.T.te = b.te.data(); X.T.le = b.le.data(); X.T.zn = b.zn;
            X.ip = b.ip.data();
            ensure(b.cnt, size_t(nc)); ensure(b.off, size_t(nc));
            X.cnt = b.cnt.data(); X.off = b.off.data();
            save_state(sa, sb, ion, b, false);
            X.mode = 1;
            collide_block(M, A, B, false, dte, P, D2, bk, b.acc.data(), X);
            save_state(sa, sb, ion, b, true);
            const long total = exclusive_scan(b.cnt.data(), b.off.data(), nc, C.tmp);
            ensure(b.births, size_t(5 * total + 5));
            X.B = b.births.data();
            X.mode = 2;
            collide_block(M, A, B, false, dte, P, D2, bk, b.acc.data(), X);
            append_births(s.species[size_t(b.el)], ion, b.births.data(), total);
        } else collide_block(M, A, B, b.a == b.b, dte, P, D2, bk, b.acc.data(), X);
    }
}

class CollideOp : public Operator {
public:
    const char* name() const override { return "collisions"; }
    Stage stage() const override { return Stage::Migrate; }
    void setup(Simulation& s) override {
        Collisions& C = *s.collisions;
        for (int i : C.used) assign_ids(s.species[size_t(i)], i);   // the load positions
    }
    void apply(Simulation& s) override { apply_all(s, false); }
};
class CollisionIdsOp : public Operator {     // ids of particles loaded later (t_start) at their load positions
public:
    const char* name() const override { return "collision_ids"; }
    Stage stage() const override { return Stage::Inject; }
    void apply(Simulation& s) override {
        for (int i : s.collisions->used) assign_ids(s.species[size_t(i)], i);
    }
};
}  // namespace

std::vector<CollisionConfig> collisions_from_deck(const Deck& d, Config& c) {
    std::vector<CollisionConfig> v;
    for (const std::string& b : d.block_names()) {
        if (b != "collisions" && b.rfind("collisions ", 0) != 0) continue;
        CollisionConfig k;
        k.name = b == "collisions" ? "" : b.substr(11);
        k.sp1 = d.str(b, "species1");
        k.sp2 = d.str(b, "species2", k.sp1);
        k.coulomb_log = d.num(b, "coulomb_log", Kind::None, 0.0);
        k.every = d.integer(b, "every", 1);
        const std::string a = d.str(b, "angle", "nanbu");
        if (a == "nanbu") k.angle = 0;
        else if (a == "smilei5") k.angle = 1;
        else throw std::runtime_error(b + ".angle: nanbu or smilei5");
        k.cap = d.flag(b, "low_t_cap", true);
        const std::string qb = d.str(b, "quantum_bmin", "hbar");
        if (qb == "hbar") k.hquantum = false;
        else if (qb == "h") k.hquantum = true;
        else throw std::runtime_error(b + ".quantum_bmin: hbar (hbar / 2 p*, default) or h (h / 2 p*, as Smilei 5.1)");
        if (k.coulomb_log < 0) throw std::runtime_error(b + ".coulomb_log: > 0 (fixed) or 0 (automatic)");
        if (k.every < 1) throw std::runtime_error(b + ".every: >= 1");
        for (const std::string* nm : {&k.sp1, &k.sp2}) {
            const SpeciesConfig* sc = nullptr;
            for (const SpeciesConfig& x : c.species)
                if (x.name == *nm) sc = &x;
            if (!sc) throw std::runtime_error(b + ": no species " + *nm);
            if (sc->photon) throw std::runtime_error(b + ": " + *nm + " is a photon species");
        }
        k.ionizing = d.flag(b, "ionizing", false);
        if (k.ionizing) {
            auto find = [&](const std::string& nm) -> SpeciesConfig& {
                for (SpeciesConfig& x : c.species)
                    if (x.name == nm) return x;
                throw std::runtime_error(b + ": no species " + nm);
            };
            SpeciesConfig *e = &find(k.sp1), *i = &find(k.sp2);
            auto electron = [](const SpeciesConfig& x) { return !x.ion.on && x.q == -1 && x.m == 1; };
            if (!electron(*e)) std::swap(e, i);
            if (!electron(*e) || k.sp1 == k.sp2)
                throw std::runtime_error(b + ".ionizing: needs an electron species (charge -1, mass 1) and an ion species");
            if (i->ion.element.empty() || i->ion.ip_eV.empty())
                throw std::runtime_error(b + ".ionizing: the ion species " + i->name + " needs an element");
            if (int(i->ion.ip_eV.size()) > 100) throw std::runtime_error(b + ".ionizing: elements up to Z = 100 (data/binding_energies.def)");
            if (!i->ion.on && !i->ion.states) {        // a charge state per particle (impact ionization only)
                if (i->ion.z0 < 0) throw std::runtime_error(b + ".ionizing: the ion species " + i->name + " needs charge_state");
                i->ion.states = true;
                i->ion.zmax = int(i->ion.ip_eV.size());
                i->q = 1;
            }
            k.electrons = d.str(b, "ionization_electrons", e->name);
            bool ok = false;
            for (const SpeciesConfig& x : c.species) ok = ok || (x.name == k.electrons && electron(x));
            if (!ok) throw std::runtime_error(b + ".ionization_electrons: " + k.electrons + " is no electron species");
        } else if (d.has(b, "ionization_electrons")) throw std::runtime_error(b + ".ionization_electrons: needs ionizing = true");
        v.push_back(k);
    }
    return v;
}

void collisions_summary(FILE* f, const Config& c) {
    for (const CollisionConfig& k : c.collisions)
        std::fprintf(f, "  collisions%s%s: %s - %s, Coulomb log %s, every %ld step(s), angles %s%s (Perez et al. 2012)%s\n",
                     k.name.empty() ? "" : " ", k.name.c_str(), k.sp1.c_str(), k.sp2.c_str(),
                     k.coulomb_log > 0 ? std::to_string(k.coulomb_log).c_str()
                                       : (k.hquantum ? "automatic (quantum b_min h / 2p*, Smilei)" : "automatic"), k.every,
                     k.angle == 0 ? "Nanbu" : "Smilei 5.1 fit", k.cap ? "" : ", no low-temperature limit",
                     k.ionizing ? (", impact ionization, electrons into " + k.electrons).c_str() : "");
}

void collisions_register(Simulation& s) {
    if (s.cfg.collisions.empty()) return;
    auto C = std::make_shared<Collisions>();
    const Config& c = s.cfg;
    auto index = [&](const std::string& nm) {
        for (size_t i = 0; i < c.species.size(); ++i)
            if (c.species[i].name == nm) return int(i);
        return -1;
    };
    std::vector<char> use(c.species.size(), 0);
    for (const CollisionConfig& k : c.collisions) {
        CollBlock b;
        b.c = k; b.a = index(k.sp1); b.b = index(k.sp2);
        use[size_t(b.a)] = use[size_t(b.b)] = 1;
        if (k.ionizing) {                             // impact ionization: the roles, the species of the new electrons, the tables
            const SpeciesConfig& e = c.species[size_t(b.a)];
            b.efirst = !e.ion.on && !e.ion.states && e.q == -1 && e.m == 1;
            b.el = index(k.electrons);
            if (b.el < 0) throw std::runtime_error("collisions: no species " + k.electrons);
            use[size_t(b.el)] = 1;
            const IonizationConfig& I = c.species[size_t(b.efirst ? b.b : b.a)].ion;
            std::vector<double> cs, te, le, ip;
            impact_tables(int(I.ip_eV.size()), I.ip_eV, 2 * M_PI / c.lambda0 * RE_SI, cs, te, le);
            for (double x : I.ip_eV) ip.push_back(x / 510998.95);
            b.cs.upload(cs); b.te.upload(te); b.le.upload(le); b.ip.upload(ip);
            b.zn = int(I.ip_eV.size());
        }
        C->debye = C->debye || !(k.coulomb_log > 0);
        C->blocks.push_back(std::move(b));
    }
    if (C->debye)                                     // the Debye length from every charged species (as Smilei)
        for (size_t i = 0; i < c.species.size(); ++i)
            if (!c.species[i].photon) use[i] = 1;
    for (size_t i = 0; i < use.size(); ++i)
        if (use[i]) { C->used.push_back(int(i)); s.species[i].require("id", Attr::I64, 0); }
    if (C->used.size() > size_t(MAXS)) throw std::runtime_error("collisions: at most 16 species (with the automatic Coulomb log: all species)");
    C->lists.resize(c.species.size());
    const double k0 = 2 * M_PI / c.lambda0;
    C->P.kre = k0 * RE_SI;
    C->P.bq = 0.5 * LCBAR_SI * k0;                   // b_min,quantum = hbar / (2 p*); quantum_bmin = h: Smilei's h / (2 p*)
    C->P.lt = std::pow(3 * C->P.kre, -1.0 / 3.0);
    s.collisions = C;
}

void collisions_add_operators(Simulation& s) {
    if (!s.collisions) return;
    Collisions& C = *s.collisions;
    if (s.cfg.rz) {                                   // 1 / V of the radial cells: pi (r_{j+1}^2 - r_j^2) dz
        const RadialGrid& g = s.rz->grid();
        std::vector<double> iv(size_t(g.nr));
        for (int j = 0; j < g.nr; ++j)
            iv[size_t(j)] = 1.0 / (M_PI * (g.r[size_t(j) + 1] * g.r[size_t(j) + 1] - g.r[size_t(j)] * g.r[size_t(j)]) * s.dom.h[0]);
        C.ivr.upload(iv);
    }
    s.pipeline.add(std::make_unique<CollisionIdsOp>());
    s.pipeline.add(std::make_unique<CollideOp>());
}

void collisions_apply(Simulation& s) { if (s.collisions) apply_all(s, true); }

CollisionTotals collisions_totals(Simulation& s, int block) {
    CollisionTotals t;
    if (!s.collisions || block < 0 || block >= int(s.collisions->blocks.size())) return t;
    CollBlock& b = s.collisions->blocks[size_t(block)];
    double v[ACC] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    const long nc = long(b.acc.size() / ACC);
    const double* A = b.acc.data();
    for (int q = 0; q < ACC; ++q) v[q] = par_sum(nc, [=] HD(long c) { return A[ACC * c + q]; });
    s.comm.allreduce_sum(v, ACC);
    t.pairs = v[0]; t.s = v[1]; t.lnL = v[2]; t.dE = v[3]; t.dP[0] = v[4]; t.dP[1] = v[5]; t.dP[2] = v[6];
    t.events = v[7]; t.e_ion = v[8];
    return t;
}

namespace {
// <u_a>, T_a = m <(u_a - <u_a>)^2 / gamma> and <E_kin> = m <gamma - 1> (m_e c^2) of a species over the ranks (collective)
void species_moments(Simulation& s, int i, double out[7]) {
    Species& sp = s.species[size_t(i)];
    const long n = long(sp.size());
    const double *X = sp.ptr<double>(Species::UX), *Y = sp.ptr<double>(Species::UY), *Z = sp.ptr<double>(Species::UZ);
    const double* W = sp.ptr<double>(Species::W);
    double v[4] = {par_sum(n, [=] HD(long p) { return W[p]; }), par_sum(n, [=] HD(long p) { return W[p] * X[p]; }),
                   par_sum(n, [=] HD(long p) { return W[p] * Y[p]; }), par_sum(n, [=] HD(long p) { return W[p] * Z[p]; })};
    s.comm.allreduce_sum(v, 4);
    const double sw = v[0] > 0 ? v[0] : 1, mx = v[1] / sw, my = v[2] / sw, mz = v[3] / sw;
    double t[3] = {par_sum(n, [=] HD(long p) { return W[p] * (X[p] - mx) * (X[p] - mx) / sqrt(1 + X[p] * X[p] + Y[p] * Y[p] + Z[p] * Z[p]); }),
                   par_sum(n, [=] HD(long p) { return W[p] * (Y[p] - my) * (Y[p] - my) / sqrt(1 + X[p] * X[p] + Y[p] * Y[p] + Z[p] * Z[p]); }),
                   par_sum(n, [=] HD(long p) { return W[p] * (Z[p] - mz) * (Z[p] - mz) / sqrt(1 + X[p] * X[p] + Y[p] * Y[p] + Z[p] * Z[p]); })};
    double ek = par_sum(n, [=] HD(long p) { return W[p] * (sqrt(1 + X[p] * X[p] + Y[p] * Y[p] + Z[p] * Z[p]) - 1); });
    s.comm.allreduce_sum(t, 3);
    s.comm.allreduce_sum(&ek, 1);
    out[0] = mx; out[1] = my; out[2] = mz;
    for (int a = 0; a < 3; ++a) out[3 + a] = sp.m * t[a] / sw;
    out[6] = sp.m * ek / sw;
}
}  // namespace

void collisions_diagnostics(Simulation& s, bool root) {
    if (!s.collisions) return;
    Collisions& C = *s.collisions;
    std::vector<CollisionTotals> T;
    for (size_t k = 0; k < C.blocks.size(); ++k) T.push_back(collisions_totals(s, int(k)));
    std::vector<int> sp;                              // the species of the blocks, in order
    for (const CollBlock& b : C.blocks)
        for (int i : {b.a, b.b})
            if (std::find(sp.begin(), sp.end(), i) == sp.end()) sp.push_back(i);
    std::vector<double> mom(7 * sp.size());
    for (size_t q = 0; q < sp.size(); ++q) species_moments(s, sp[q], &mom[7 * q]);
    std::vector<double> zmean;                        // ionizing blocks: <Z> of the ion species (weighted)
    for (const CollBlock& b : C.blocks) {
        if (!b.c.ionizing) continue;
        Species& ion = s.species[size_t(b.efirst ? b.b : b.a)];
        const int* Zs = ion.ptr<int>(ion.zslot);
        const double* W = ion.ptr<double>(Species::W);
        double v[2] = {par_sum(long(ion.size()), [=] HD(long p) { return W[p] * Zs[p]; }), par_sum(long(ion.size()), [=] HD(long p) { return W[p]; })};
        s.comm.allreduce_sum(v, 2);
        zmean.push_back(v[1] > 0 ? v[0] / v[1] : 0.0);
    }
    if (!root) return;
    if (!C.f) {
        bool fresh;
        C.f = open_log(s.cfg.dir + "/collisions.txt", s.restart_step, fresh);
        if (fresh) {
            std::fprintf(C.f, "# step  t");
            for (const CollBlock& b : C.blocks) {
                const std::string nm = b.c.name.empty() ? b.c.sp1 + "-" + b.c.sp2 : b.c.name;
                std::fprintf(C.f, "  pairs_%s  <s>_%s  <lnL>_%s  dE_%s", nm.c_str(), nm.c_str(), nm.c_str(), nm.c_str());
                if (b.c.ionizing) std::fprintf(C.f, "  events_%s  E_ion_%s  <Z>_%s", nm.c_str(), nm.c_str(), nm.c_str());
            }
            for (int i : sp) {
                const char* nm = s.cfg.species[size_t(i)].name.c_str();
                std::fprintf(C.f, "  <ux>_%s <uy>_%s <uz>_%s Tx_%s Ty_%s Tz_%s <Ekin>_%s", nm, nm, nm, nm, nm, nm, nm);
            }
            std::fprintf(C.f, "   (pairs, <s>, <lnL>, dE: since the start of this run, dE the kinetic energy changed by the collisions "
                              "in m c^2 n_c / k0^3; u = p / m c; T_a = m <(u_a - <u_a>)^2 / gamma>, <Ekin> = m <gamma - 1> in m_e c^2)\n");
        }
    }
    std::fprintf(C.f, "%ld %.6e", s.istep, s.t);
    size_t zi = 0;
    for (size_t k = 0; k < T.size(); ++k) {
        const CollisionTotals& t = T[k];
        std::fprintf(C.f, " %.0f %.6e %.6e %.6e", t.pairs, t.pairs > 0 ? t.s / t.pairs : 0.0, t.pairs > 0 ? t.lnL / t.pairs : 0.0, t.dE);
        if (C.blocks[k].c.ionizing) std::fprintf(C.f, " %.0f %.9e %.9e", t.events, t.e_ion, zmean[zi++]);
    }
    for (double x : mom) std::fprintf(C.f, " %.9e", x);
    std::fprintf(C.f, "\n");
    std::fflush(C.f);
}
