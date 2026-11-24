// Multi-GPU performance pieces (decision 92, docs/GPU_TESTING.md "Multi-GPU overlap"): the device migration against the
// former host algorithm, all species in one migration against one by one, the self-periodic guard exchange against the
// copies / sums it must give, and the push in two parts (boundary, interior) against the one-kernel push.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include "common.hpp"
#include "sim.hpp"

static Domain ov_box(int nx, int ny, int nz, double hx, double hy, double hz, bool px) {
    Domain d;
    d.N[0] = nx; d.N[1] = ny; d.N[2] = nz;
    d.h[0] = hx; d.h[1] = hy; d.h[2] = hz;
    d.periodic[0] = px;
    d.ng = 3;
    return d;
}

// a species with all four attribute types; positions partly outside the box (up to 0.9 cells), momenta random
static void ov_fill(Species& sp, const Domain& d, int np, unsigned seed) {
    sp.q = -1; sp.m = 1;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u01(0, 1), uu(-2, 2);
    std::vector<double> h[Species::NC];
    for (int p = 0; p < np; ++p) {
        double r[3];
        for (int a = 0; a < 3; ++a) r[a] = (u01(rng) * (d.N[a] + 1.8) - 0.9) * d.h[a];
        const double v[Species::NC] = {r[0], r[1], r[2], uu(rng), uu(rng), uu(rng), 0.5 + u01(rng)};
        for (int c = 0; c < Species::NC; ++c) h[c].push_back(v[c]);
    }
    sp.append(h);
    const int sf = sp.require("f32", Attr::F32), si = sp.require("i32", Attr::I32), sl = sp.require("i64", Attr::I64);
    const size_t N = static_cast<size_t>(np);
    std::vector<float> f(N); std::vector<int> i(N); std::vector<long long> l(N);
    for (int p = 0; p < np; ++p) { f[size_t(p)] = 0.25f * p; i[size_t(p)] = p; l[size_t(p)] = 3000000000LL + p; }
    sp.set_as(sf, f); sp.set_as(si, i); sp.set_as(sl, l);
}

// All attributes of all particles as rows (host), in storage order
static std::vector<std::vector<double>> ov_rows(const Species& sp) {
    std::vector<std::vector<double>> r(size_t(sp.n_attr()));
    for (int s = 0; s < sp.n_attr(); ++s) {
        const Attr t = sp.info(s).type;
        if (t == Attr::F64) r[size_t(s)] = sp.get_as<double>(s);
        else if (t == Attr::F32) { for (float v : sp.get_as<float>(s)) r[size_t(s)].push_back(v); }
        else if (t == Attr::I32) { for (int v : sp.get_as<int>(s)) r[size_t(s)].push_back(v); }
        else { for (long long v : sp.get_as<long long>(s)) r[size_t(s)].push_back(double(v)); }
    }
    return r;
}

// The former host migration on one rank (sorted list of the leavers, holes below n - m filled from the tail, the last
// staying particle first; route along x, y, z: open x edges lose the particle, periodic y, z wrap it and it arrives
// from "the neighbour" = this rank, appended after the particles kept along that axis)
static std::vector<std::vector<double>> ov_reference(std::vector<std::vector<double>> a, const Domain& d, double& lost) {
    const size_t n = a[0].size(), A = a.size();
    std::vector<size_t> idx;
    for (size_t p = 0; p < n; ++p) {
        bool out = false;
        for (int c = 0; c < 3; ++c) out = out || a[size_t(c)][p] < d.keep_lo(c) || a[size_t(c)][p] >= d.keep_hi(c);
        if (out) idx.push_back(p);
    }
    std::vector<std::vector<double>> cur;                           // rows of the leavers
    for (size_t p : idx) { std::vector<double> row(A); for (size_t s = 0; s < A; ++s) row[s] = a[s][p]; cur.push_back(row); }
    const size_t m = idx.size(), nn = n - m;
    long t = long(n) - 1;
    for (size_t h = 0; h < m && idx[h] < nn; ++h) {
        while (std::binary_search(idx.begin(), idx.end(), size_t(t))) --t;
        for (size_t s = 0; s < A; ++s) a[s][idx[h]] = a[s][size_t(t)];
        --t;
    }
    for (auto& v : a) v.resize(nn);
    for (int ax = 0; ax < 3; ++ax) {
        std::vector<std::vector<double>> keep, send[2];
        for (auto& row : cur) {
            const double x = row[size_t(ax)];
            const int dir = x < d.keep_lo(ax) ? 0 : (x >= d.keep_hi(ax) ? 1 : -1);
            if (dir < 0) { keep.push_back(row); continue; }
            if (!d.periodic[ax]) {
                const double u2 = row[3] * row[3] + row[4] * row[4] + row[5] * row[5];
                lost += row[6] * u2 / (std::sqrt(1.0 + u2) + 1.0);
                continue;
            }
            row[size_t(ax)] += x < 0 ? d.L(ax) : -d.L(ax);
            send[dir].push_back(row);
        }
        for (int dir = 0; dir < 2; ++dir) keep.insert(keep.end(), send[dir].begin(), send[dir].end());
        cur.swap(keep);
    }
    for (auto& row : cur) for (size_t s = 0; s < A; ++s) a[s].push_back(row[s]);
    return a;
}

void test_overlap(Comm& comm) {
    if (comm.size > 1) return;                                       // one-rank tests
    // O1. the device migration = the former host algorithm, exactly (order, all attributes, lost energy)
    {
        Domain d = ov_box(12, 7, 5, 0.3, 0.5, 0.6, false);
        const int want[3] = {1, 1, 1};
        comm.decompose(d, want);
        Species sp;
        ov_fill(sp, d, 5000, 11);
        double lost_ref = 0;
        const auto ref = ov_reference(ov_rows(sp), d, lost_ref);
        migrate(sp, d, comm);
        const auto got = ov_rows(sp);
        bool same = got.size() == ref.size();
        for (size_t s = 0; same && s < ref.size(); ++s) same = got[s] == ref[s];
        check("overlap: device migration = the former host algorithm (order, 4 attribute types, lost energy)",
              same && sp.lost_energy == lost_ref && ref[0].size() < 5000 && ref[0].size() > 3000,
              fmt("%.0f of 5000 kept, lost energy %.3g", double(ref[0].size()), lost_ref));
        // O2. two species in one migration (one message per axis) = each species alone
        Species a, b, a1, b1;
        ov_fill(a, d, 3000, 21); ov_fill(b, d, 2000, 22); ov_fill(a1, d, 3000, 21); ov_fill(b1, d, 2000, 22);
        b.require("extra", Attr::F64, 2.5); b1.require("extra", Attr::F64, 2.5);   // rows of different length
        std::vector<Species*> both{&a, &b};
        migrate(both, d, comm);
        migrate(a1, d, comm); migrate(b1, d, comm);
        check("overlap: all species in one migration = one by one",
              ov_rows(a) == ov_rows(a1) && ov_rows(b) == ov_rows(b1) && a.lost_energy == a1.lost_energy &&
                  b.lost_energy == b1.lost_energy, "");
    }
    // O3. guard exchange of an axis periodic with itself (one kernel): the guards are the copies of the other side; the
    //     sums of add_guards are edge + guard in the order of the general path, the guards then 0
    {
        Domain d = ov_box(9, 7, 6, 1, 1, 1, true);
        const int want[3] = {1, 1, 1};
        comm.decompose(d, want);
        Field f(d), g(d);
        std::mt19937_64 rng(3);
        std::uniform_real_distribution<double> u(-1, 1);
        H hf(f);
        for (double& v : hf.v) v = u(rng);
        hf.put(f);
        H hg(g);
        for (double& v : hg.v) v = u(rng);
        hg.put(g);
        const H f0(f), g0(g);
        comm.exchange(d, {&f});
        const H f1(f);
        int bad = 0;
        auto wrap = [](int i, int n) { return i < 0 ? i + n : (i >= n ? i - n : i); };
        for (int k = -3; k < d.n[2] + 3; ++k)
            for (int j = -3; j < d.n[1] + 3; ++j)
                for (int i = -3; i < d.n[0] + 3; ++i) {
                    const size_t q = f.idx(i, j, k), q0 = f.idx(wrap(i, d.n[0]), wrap(j, d.n[1]), wrap(k, d.n[2]));
                    bad += f1.v[q] != f0.v[q0];
                }
        // add_guards, the reference on the host axis by axis: edge += guard of the far side, then the guards = 0
        std::vector<double> e = g0.v;
        const int ng = 3;
        for (int a = 0; a < 3; ++a) {
            const int n = d.n[a];
            for (int k = -ng; k < d.n[2] + ng; ++k)
                for (int j = -ng; j < d.n[1] + ng; ++j)
                    for (int i = -ng; i < d.n[0] + ng; ++i) {
                        int r[3] = {i, j, k};
                        const int t = r[a];
                        if (t < 0 || t >= ng) continue;
                        auto at = [&](int c) { int s[3] = {r[0], r[1], r[2]}; s[a] = c; return g.idx(s[0], s[1], s[2]); };
                        e[at(t)] += e[at(n + t)];
                        e[at(n - ng + t)] += e[at(-ng + t)];
                        e[at(n + t)] = 0; e[at(-ng + t)] = 0;
                    }
        }
        comm.add_guards(d, {&g});
        const H g1(g);
        int badg = 0;
        for (size_t q = 0; q < e.size(); ++q) badg += g1.v[q] != e[q];
        check("overlap: guard exchange periodic with itself (one kernel) = copies / sums of the general path, exactly",
              bad == 0 && badg == 0, fmt("%.0f wrong copies, %.0f wrong sums", bad, badg));
    }
    // O4. push in two parts (boundary, interior: PushDeposit::apply_overlap) = the one-kernel push: particles exactly (no
    //     particle pushed twice, also those that part 1 moves into the interior), the
    //     currents to round-off (the order of the additions differs), every particle pushed once
    {
        Domain d = ov_box(40, 36, 32, 0.4, 0.5, 0.45, true);
        const int want[3] = {1, 1, 1};
        comm.decompose(d, want);
        Fields f1(d), f2(d);
        std::uniform_real_distribution<double> u(-0.2, 0.2);
        for (Fields* F : {&f1, &f2}) {
            std::mt19937_64 r2(9);
            for (Field* x : {&F->ex, &F->ey, &F->ez, &F->bx, &F->by, &F->bz}) {
                H h(*x);
                for (double& v : h.v) v = u(r2);
                h.put(*x);
            }
            F->zero_currents();
        }
        Species s1, s2;
        ov_fill(s1, d, 20000, 31); ov_fill(s2, d, 20000, 31);
        migrate(s1, d, comm); migrate(s2, d, comm);                  // all inside
        push_and_deposit(s1, f1, d, Pusher::HigueraCary, 2);
        Buf<unsigned char> in(s2.size());
        push_and_deposit_part(s2, f2, d, Pusher::HigueraCary, 2, 1, in.data());
        const auto x1 = s2.get(Species::X);
        push_and_deposit_part(s2, f2, d, Pusher::HigueraCary, 2, 2, in.data());
        const auto a = ov_rows(s1), b = ov_rows(s2);
        const auto x0 = s1.get(Species::X);
        int moved1 = 0;                                              // boundary particles moved by part 1
        for (size_t p = 0; p < x1.size(); ++p) moved1 += x1[p] == x0[p];
        double dj = 0, jm = 0;
        for (Field* F : {&f1.jx, &f1.jy, &f1.jz}) {
            const Field& G = F == &f1.jx ? f2.jx : (F == &f1.jy ? f2.jy : f2.jz);
            const auto p = F->host(), q = G.host();
            for (size_t k = 0; k < p.size(); ++k) { dj = std::max(dj, std::abs(p[k] - q[k])); jm = std::max(jm, std::abs(p[k])); }
        }
        check("overlap: push in two parts (boundary, interior) = one kernel: particles exactly, currents to round-off",
              a == b && dj <= 1e-13 * jm && jm > 0 && moved1 > 1000 && moved1 < 19000,
              fmt("%.0f of 20000 boundary particles, max|dJ| %.1e", moved1, dj / jm) + " of max|J|");
    }
}
