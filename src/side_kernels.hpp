// Kernel helpers of the side TFSF tube (src/side.cpp, src/side_luminal.cpp; docs/specs/side_injection.md): complex
// numbers for device code, the tube geometry by value, the masks of the differences that cross a face, and the
// corrections of one plane-wave component at one cell (the "brackets", without the phase of the cell and time).
#pragma once
#include "grid.hpp"

namespace side {
struct Cx { double r, i; };
HD inline Cx operator*(Cx a, Cx b) { return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r}; }
HD inline Cx operator+(Cx a, Cx b) { return {a.r + b.r, a.i + b.i}; }
HD inline Cx operator-(Cx a, Cx b) { return {a.r - b.r, a.i - b.i}; }
HD inline Cx operator*(double s, Cx a) { return {s * a.r, s * a.i}; }
HD inline Cx conj(Cx a) { return {a.r, -a.i}; }
HD inline Cx& operator+=(Cx& a, Cx b) { a.r += b.r; a.i += b.i; return a; }

constexpr int NCD = 28;                               // doubles per component: k[3], w, v[12], pr[4], mu[2], j[6]
HD inline Cx V(const double* c, int q) { return {c[4 + 2 * q], c[5 + 2 * q]}; }
// the incident value of component q = 0..8 (Ex .. Bz, then the plasma current Jx, Jy, Jz)
HD inline Cx V9(const double* c, int q) { return q < 6 ? V(c, q) : Cx{c[22 + 2 * (q - 6)], c[23 + 2 * (q - 6)]}; }
HD inline Cx PR(const double* c, int q) { return {c[16 + 2 * q], c[17 + 2 * q]}; }
HD inline Cx MU(const double* c) { return {c[20], c[21]}; }

// The tube and the grid, by value into the kernels. cp: the components; tb: per component e^{i kx D}, e^{i ky hy},
// e^{i kz hz} and the tables along y, z (absolute index; S complex); tx: per component the table along x (local i,
// plane waves only; nx complex); ft: the factor of the step per component (plane waves).
struct Geo {
    int lo[3], hi[3];
    bool f[3];
    long o1, o2;                                      // global index of local cell 0 along y, z
    int nx, ny, nz, S, nc;
    double D, hy, hz;
    const double *cp, *tb, *tx, *ft;
    HD double tn(int a, long g) const { return !f[a] || (g >= lo[a] && g <= hi[a]) ? 1.0 : 0.0; }
    HD double th(int a, long g) const { return !f[a] || (g >= lo[a] && g <= hi[a] - 1) ? 1.0 : 0.0; }
    // component c (Ex, Ey, Ez, Bx, By, Bz; Jx, Jy, Jz at the points of Ex, Ey, Ez in y, z): on a half node along y for
    // Ey, Bx, Bz, along z for Ez, Bx, By
    HD double T(int c, long j, long k) const {
        if (c >= 6) c -= 6;
        const bool yh = c == 1 || c == 3 || c == 5, zh = c == 2 || c == 3 || c == 4;
        return (yh ? th(1, j) : tn(1, j)) * (zh ? th(2, k) : tn(2, k));
    }
    HD Cx tab(int m, int q) const { const double* t = tb + 2 * (long(S) * m + q); return {t[0], t[1]}; }
    HD Cx ex(int m) const { return tab(m, 0); }
    HD Cx ey(int m) const { return tab(m, 1); }
    HD Cx ez(int m) const { return tab(m, 2); }
    HD Cx perp(int m, int j, int k) const { return tab(m, 3 + j) * tab(m, 3 + ny + k); }   // e^{i (ky y + kz z)}, local j, k
    HD Cx phase(int m, int i, int j, int k) const {    // plane waves: the whole phase at the step
        const double* x = tx + 2 * (long(nx) * m + i);
        return Cx{ft[2 * m], ft[2 * m + 1]} * Cx{x[0], x[1]} * perp(m, j, k);
    }
};

// Weights (T_out - T_in) of the neighbours in the differences at the cell (jg, kg): predictor and corrector (Ex at the
// node from Bz at j, j-1 and By at k, k-1; Bx at the half node from Ez at j+1, j and Ey at k+1, k) and the transport
// sources (Psi_y from Bx* at k, k-1, Psi_z from Bx* at j, j-1, Phi_y from Ex* at k+1, k, Phi_z from Ex* at j+1, j).
struct Wts { double bz0, bzm, by0, bym, ezp, ez0, eyp, ey0, py0, pym, pz0, pzm, fyp, fy0, fzp, fz0; };
HD inline Wts weights(const Geo& g, long jg, long kg) {
    Wts w;
    const double tE = g.T(0, jg, kg), tB = g.T(3, jg, kg), tY = g.T(1, jg, kg), tZ = g.T(2, jg, kg);
    w.bz0 = tE - g.T(5, jg, kg); w.bzm = tE - g.T(5, jg - 1, kg); w.by0 = tE - g.T(4, jg, kg); w.bym = tE - g.T(4, jg, kg - 1);
    w.ezp = tB - g.T(2, jg + 1, kg); w.ez0 = tB - g.T(2, jg, kg); w.eyp = tB - g.T(1, jg, kg + 1); w.ey0 = tB - g.T(1, jg, kg);
    w.py0 = tY - g.T(3, jg, kg); w.pym = tY - g.T(3, jg, kg - 1); w.pz0 = tZ - g.T(3, jg, kg); w.pzm = tZ - g.T(3, jg - 1, kg);
    w.fyp = tZ - g.T(0, jg, kg + 1); w.fy0 = tZ - g.T(0, jg, kg); w.fzp = tY - g.T(0, jg + 1, kg); w.fz0 = tY - g.T(0, jg, kg);
    return w;
}

// Predictor (times 0.5 D) and corrector (times D; with the mean (1 + e^{-iwD})/2) of one component: a difference across
// a face gets the incident value of the far point. aE, aB: the corrections of Ex*, Bx* (Ex, Bx) before the phase.
HD inline void bracket_pc(const double* c, Cx ey, Cx ez, const Wts& w, double iy, double iz, Cx& aE, Cx& aB) {
    const Cx a = V(c, 5) * (Cx{w.bz0, 0} - w.bzm * conj(ey)), b = V(c, 4) * (Cx{w.by0, 0} - w.bym * conj(ez));
    aE = iy * a - iz * b;
    const Cx p = V(c, 2) * (w.ezp * ey - Cx{w.ez0, 0}), r = V(c, 1) * (w.eyp * ez - Cx{w.ey0, 0});
    aB = Cx{0, 0} - (iy * p - iz * r);
}
// Transport (times 0.5 D): the sources at (a + 1/2), a = i - 1, i; at a = i - 1 the incident value is that of a = i
// times e^{-i kx D}. kind 0: Ex*, Bx* of the incident field as the RIP-CC predictor, 2: the mean of t_n and t_n+1 (iRIP),
// 3: iRIP with o5, the closed mean (stored in pr).
// t: the corrections of Ey, Bz, Ez, By.
HD inline void bracket_t(const double* c, Cx ex, Cx ey, Cx ez, const Wts& w, double iy, double iz, int kind, Cx t[4]) {
    const Cx em = conj(ex), mx = 0.5 * (Cx{1, 0} + ex), one{1, 0};
    const Cx IE = kind != 2 ? PR(c, 0) : V(c, 0) * MU(c), IB = kind != 2 ? PR(c, 1) : V(c, 3) * MU(c);
    const Cx spy = (iz * mx) * (IB * (Cx{w.py0, 0} - w.pym * conj(ez)));
    const Cx spz = (-iy * mx) * (IB * (Cx{w.pz0, 0} - w.pzm * conj(ey)));
    const Cx sfy = (-iz * mx) * (IE * (w.fyp * ez - Cx{w.fy0, 0}));
    const Cx sfz = (iy * mx) * (IE * (w.fzp * ey - Cx{w.fz0, 0}));
    t[0] = (em + one) * spy + (em - one) * sfz;
    t[1] = (em - one) * spy + (em + one) * sfz;
    t[2] = (one + em) * spz + (one - em) * sfy;
    t[3] = (one - em) * spz + (one + em) * sfy;
}
}  // namespace side
