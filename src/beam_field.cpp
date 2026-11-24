#include "beam_field.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

using cplx = std::complex<double>;

// In-place DFT of n values with stride s (sign -1 forward, +1 backward, unnormalised): radix 2 for a
// power of two, the plain sum otherwise (setup only).
static void dft(cplx* a, int n, long s, int sign, std::vector<cplx>& tmp) {
    if (n == 1) return;
    tmp.resize(size_t(n));
    if ((n & (n - 1)) == 0) {
        for (int i = 1, j = 0; i < n; ++i) {                     // bit reversal
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(a[i * s], a[j * s]);
        }
        for (int len = 2; len <= n; len <<= 1) {
            const cplx w = std::polar(1.0, sign * 2 * M_PI / len);
            for (int i = 0; i < n; i += len) {
                cplx wk = 1;
                for (int k = 0; k < len / 2; ++k, wk *= w) {
                    const cplx u = a[(i + k) * s], v = a[(i + k + len / 2) * s] * wk;
                    a[(i + k) * s] = u + v;
                    a[(i + k + len / 2) * s] = u - v;
                }
            }
        }
        return;
    }
    for (int k = 0; k < n; ++k) {
        cplx sum = 0;
        for (int j = 0; j < n; ++j) sum += a[j * s] * std::polar(1.0, sign * 2 * M_PI * double(long(j) * k % n) / n);
        tmp[size_t(k)] = sum;
    }
    for (int k = 0; k < n; ++k) a[k * s] = tmp[size_t(k)];
}

// 2D DFT of every x slice of a[(i * ny + j) * nz + k]
static void dft_yz(std::vector<cplx>& a, int nx, int ny, int nz, int sign) {
    std::vector<cplx> tmp;
    for (int i = 0; i < nx; ++i) {
        cplx* p = &a[size_t(i) * ny * nz];
        for (int j = 0; j < ny; ++j) dft(p + long(j) * nz, nz, 1, sign, tmp);
        for (int k = 0; k < nz; ++k) dft(p + k, ny, nz, sign, tmp);
    }
}

// Periodic x (RIP-CC, iRIP, Yee): the whole problem by a 3D DFT (docs/specs/beam_self_field.md "Periodic boxes").
// Per k, with K_a^2 = (4/h_a^2) sin^2(theta_a/2) and W_a = exp(i theta_a):
//   RIP-CC (phi at the rho point i+1/2):  L = K_x^2/gamma^2 + cos^2(theta_x/2) K_perp^2,
//     Ex = -(1 - 1/W_x) phi / (gamma^2 h_x),  Ey = -(W_y - 1)/h_y (1 + 1/W_x)/2 phi,  Bz = beta Ey,  By = -beta Ez;
//   Yee (phi at the node):                L = K_x^2/gamma^2 + K_perp^2,
//     Ex = -(W_x - 1) phi / (gamma^2 h_x),    Ey = -(W_y - 1)/h_y phi,  Bz, By = +-beta (Ey, Ez) shifted by h_x/2
//     to x_{i+1/2} (exp(i theta_x/2): B at integer time levels, the band-limited value of the moving field).
// phi = rho / L. The k_perp = 0 modes need no special treatment here (E_perp is exactly 0 there). k = 0 is dropped:
// a net charge in a periodic box has no field without a neutralizing background; the Gauss residual then starts at
// -<rho> (uniform), which the run's reference residual absorbs.
static void periodic_beam_field(Fields& f, const std::vector<double>& R, double beta, const Domain& d, Comm& comm) {
    const int NX = d.N[0], NY = d.N[1], NZ = d.N[2];
    const long nyz = long(NY) * NZ;
    const bool yee = d.solver == Solver::Yee;
    const double g2 = 1.0 / (1.0 - beta * beta);
    std::vector<cplx> P(R.begin(), R.end()), T(P.size()), tmp;
    dft_yz(P, NX, NY, NZ, -1);
    for (long q = 0; q < nyz; ++q) dft(&P[size_t(q)], NX, nyz, -1, tmp);
    const double norm = 1.0 / (double(NX) * double(nyz));
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j)
            for (int k = 0; k < NZ; ++k) {
                cplx& p = P[(size_t(i) * NY + j) * NZ + k];
                const double sx = std::sin(M_PI * i / NX), sy = std::sin(M_PI * j / NY), sz = std::sin(M_PI * k / NZ);
                const double Kx2 = 4 * sx * sx / (d.h[0] * d.h[0]);
                const double Kp2 = 4 * sy * sy / (d.h[1] * d.h[1]) + 4 * sz * sz / (d.h[2] * d.h[2]);
                const double L = Kx2 / g2 + (yee ? 1.0 : 1.0 - sx * sx) * Kp2;
                p = (i == 0 && j == 0 && k == 0) || L <= 0 ? cplx(0) : p * norm / L;
            }
    std::vector<double> ex = f.ex.host(), ey = f.ey.host(), ez = f.ez.host(), by = f.by.host(), bz = f.bz.host();
    // component c: 0 Ex, 1 Ey, 2 Ez, 3 By, 4 Bz
    for (int c = 0; c < 5; ++c) {
        for (int i = 0; i < NX; ++i)
            for (int j = 0; j < NY; ++j)
                for (int k = 0; k < NZ; ++k) {
                    const size_t q = (size_t(i) * NY + j) * NZ + k;
                    const double tx = 2 * M_PI * i / NX, ty = 2 * M_PI * j / NY, tz = 2 * M_PI * k / NZ;
                    const cplx Wx = std::polar(1.0, tx), Wy = std::polar(1.0, ty), Wz = std::polar(1.0, tz);
                    const cplx mux = yee ? cplx(1) : 0.5 * (1.0 + 1.0 / Wx);          // RIP-CC: phibar at integer x
                    // Yee: B at x_{i+1/2}; exp(i theta/2) is not 2 pi periodic: the signed wavenumber (i > NX/2: k < 0)
                    const double txs = 2 * M_PI * (2 * i <= NX ? i : i - NX) / NX;
                    const cplx shift = yee ? std::polar(1.0, 0.5 * txs) : cplx(1);
                    cplx v;
                    if (c == 0) v = -(yee ? Wx - 1.0 : 1.0 - 1.0 / Wx) / (g2 * d.h[0]);
                    else if (c == 1 || c == 4) v = -(Wy - 1.0) / d.h[1] * mux * (c == 4 ? beta * shift : cplx(1));
                    else v = -(Wz - 1.0) / d.h[2] * mux * (c == 3 ? -beta * shift : cplx(1));
                    T[q] = v * P[q];
                }
        for (long q = 0; q < nyz; ++q) dft(&T[size_t(q)], NX, nyz, +1, tmp);
        dft_yz(T, NX, NY, NZ, +1);
        std::vector<double>& out = c == 0 ? ex : c == 1 ? ey : c == 2 ? ez : c == 3 ? by : bz;
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i)
                    out[f.ex.idx(i, j, k)] += T[(size_t(d.off[0] + i) * NY + d.off[1] + j) * NZ + d.off[2] + k].real();
    }
    f.ex.upload(ex); f.ey.upload(ey); f.ez.upload(ez); f.by.upload(by); f.bz.upload(bz);
    comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
}

// ---- the exact discrete steady state of RIP-CC (decision 99, docs/specs/beam_self_field.md "Steady init") ----
// One RIP-CC vacuum step (src/fields.cpp Fields::step_core, no hooks) applied to a single Fourier mode
// F(i, j, k) = F^ exp(i (tx i + ty j + tz k)) of every array index: F = (Ex, Ey, Ez, Bx, By, Bz), sources Jx, Jy, Jz.
// Every shift of an index is a factor S_a^(+-1) = exp(+-i t_a); the formulas are those of the kernel, term by term.
struct ModeStep {
    cplx Sx, Sy, Sz;
    double D, iy, iz;
    void apply(const cplx F[6], const cplx J[3], cplx out[6]) const {
        const cplx Ex = F[0], Ey = F[1], Ez = F[2], Bx = F[3], By = F[4], Bz = F[5];
        const cplx Sxi = 1.0 / Sx, Syi = 1.0 / Sy, Szi = 1.0 / Sz, mux = 0.5 * (1.0 + Sx);
        // 1. predictor
        const cplx Exs = Ex + 0.5 * D * ((Bz - Bz * Syi) * iy - (By - By * Szi) * iz - J[0]);
        const cplx Bxs = Bx - 0.5 * D * ((Ez * Sy - Ez) * iy - (Ey * Sz - Ey) * iz);
        // 2. transport: psi, phi at index a = i (factor 1) or a = i - 1 (factor Sxi on the whole bracket)
        const cplx psiy = mux * (1.0 - Szi) * Bxs * iz - J[1];
        const cplx psiz = -mux * (1.0 - Syi) * Bxs * iy - J[2];
        const cplx phiy = -mux * (Sz - 1.0) * Exs * iz;
        const cplx phiz = mux * (Sy - 1.0) * Exs * iy;
        const cplx tpy = Sxi * (Ey + Bz + D * (psiy + phiz));
        const cplx tmy = Sx * (Ey - Bz) + D * (psiy - phiz);
        const cplx tpz = Sx * (Ez + By) + D * (psiz + phiy);
        const cplx tmz = Sxi * (Ez - By + D * (psiz - phiy));
        const cplx nEy = 0.5 * (tpy + tmy), nBz = 0.5 * (tpy - tmy), nEz = 0.5 * (tpz + tmz), nBy = 0.5 * (tpz - tmz);
        // 3. corrector with the time-centred transverse fields
        const cplx bz0 = 0.5 * (Bz + nBz), by0 = 0.5 * (By + nBy), ez0 = 0.5 * (Ez + nEz), ey0 = 0.5 * (Ey + nEy);
        out[0] = Ex + D * ((bz0 - bz0 * Syi) * iy - (by0 - by0 * Szi) * iz - J[0]);
        out[3] = Bx - D * ((ez0 * Sy - ez0) * iy - (ey0 * Sz - ey0) * iz);
        out[1] = nEy; out[2] = nEz; out[4] = nBy; out[5] = nBz;
    }
};

// Solves A x = b for a 6 x 6 complex A (Gaussian elimination, partial pivoting); A and b are overwritten.
static void solve6(cplx A[6][6], cplx b[6]) {
    for (int c = 0; c < 6; ++c) {
        int p = c;
        for (int r = c + 1; r < 6; ++r)
            if (std::abs(A[r][c]) > std::abs(A[p][c])) p = r;
        if (p != c) { for (int q = 0; q < 6; ++q) std::swap(A[c][q], A[p][q]); std::swap(b[c], b[p]); }
        for (int r = c + 1; r < 6; ++r) {
            const cplx m = A[r][c] / A[c][c];
            for (int q = c; q < 6; ++q) A[r][q] -= m * A[c][q];
            b[r] -= m * b[c];
        }
    }
    for (int c = 5; c >= 0; --c) {
        for (int q = c + 1; q < 6; ++q) b[c] -= A[c][q] * b[q];
        b[c] /= A[c][c];
    }
}

// The steady field of a rigid beam moving beta cells per step (c dt = h_x): F^{n+1} = lambda F^n, lambda = exp(-i beta tx)
// with the signed tx, and J^{n+1/2} = lambda^n Jx^ from the discrete continuity equation, Jx^ = (1 - lambda) rho^ / (Sx - 1)
// (rho at i+1/2, Jx at i; Jy = Jz = 0). Per mode, probing ModeStep gives M and the source column c, and
// (M - lambda) F^ = -c Jx^. Closed forms where that matrix is singular:
//   k_perp = 0: only the sheet field Ex^ = h_x rho^ / (Sx - 1) (periodic x; with open x the existing Ex0, zero ahead);
//     the TEM modes there have the eigenvalue 1/Sx = lambda at beta = 1 and no source.
//   tx = 0: the Gauss-law field (x independent: static E_perp, B = beta x E, which the step leaves unchanged exactly).
// Open x: the problem is solved in a periodic box padded to a power of two >= 2 NX (rho zero outside the window) and
// cut back; the beam's field is contracted to ~ r / gamma along x, so the padding decouples the images.
static void steady_beam_field(Fields& f, const std::vector<double>& R, double beta, const Domain& d, Comm& comm) {
    const int NX = d.N[0], NY = d.N[1], NZ = d.N[2];
    const long nyz = long(NY) * NZ;
    const bool per = d.periodic[0];
    int NP = NX;
    if (!per) { NP = 1; while (NP < 2 * NX) NP *= 2; }
    const double g2 = 1.0 / (1.0 - beta * beta), D = d.h[0], hy = d.h[1], hz = d.h[2];
    std::vector<cplx> P(size_t(NP) * nyz, cplx(0)), tmp;
    for (size_t q = 0; q < R.size(); ++q) P[q] = R[q];
    dft_yz(P, NP, NY, NZ, -1);
    for (long q = 0; q < nyz; ++q) dft(&P[size_t(q)], NP, nyz, -1, tmp);
    const double norm = 1.0 / (double(NP) * double(nyz));
    std::vector<cplx> F6(P.size() * 6);
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < NP; ++i)
        for (int j = 0; j < NY; ++j)
            for (int k = 0; k < NZ; ++k) {
                const size_t q = (size_t(i) * NY + j) * NZ + k;
                cplx* F = &F6[6 * q];
                for (int c = 0; c < 6; ++c) F[c] = 0;
                const cplx rho = P[q] * norm;
                if (i == 0 && j == 0 && k == 0) continue;
                const double tx = 2 * M_PI * (2 * i <= NP ? i : i - NP) / NP, ty = 2 * M_PI * j / NY, tz = 2 * M_PI * k / NZ;
                const cplx Sx = std::polar(1.0, tx), Sy = std::polar(1.0, ty), Sz = std::polar(1.0, tz);
                if (j == 0 && k == 0) {                     // k_perp = 0: the sheet field
                    if (per) F[0] = D * rho / (Sx - 1.0);
                    continue;
                }
                if (i == 0) {                               // x independent: the Gauss-law field
                    const double sy = std::sin(0.5 * ty), sz = std::sin(0.5 * tz);
                    const cplx phi = rho / (4 * sy * sy / (hy * hy) + 4 * sz * sz / (hz * hz));
                    F[1] = -(Sy - 1.0) / hy * phi; F[2] = -(Sz - 1.0) / hz * phi;
                    F[5] = beta * F[1]; F[4] = -beta * F[2];
                    continue;
                }
                const ModeStep st{Sx, Sy, Sz, D, 1.0 / hy, 1.0 / hz};
                const cplx lam = std::polar(1.0, -beta * tx);
                const cplx J[3] = {(1.0 - lam) * rho / (Sx - 1.0), 0, 0}, J0[3] = {0, 0, 0};
                cplx A[6][6], b[6], e[6], o[6];
                for (int c = 0; c < 6; ++c) {               // the columns of M - lambda
                    for (int r = 0; r < 6; ++r) e[r] = r == c ? 1.0 : 0.0;
                    st.apply(e, J0, o);
                    for (int r = 0; r < 6; ++r) A[r][c] = o[r] - (r == c ? lam : 0.0);
                }
                for (int r = 0; r < 6; ++r) e[r] = 0;
                st.apply(e, J, o);                          // the source column times Jx^
                for (int r = 0; r < 6; ++r) b[r] = -o[r];
                solve6(A, b);
                for (int c = 0; c < 6; ++c) F[c] = b[c];
            }
    (void)g2;
    std::vector<double> h[6] = {f.ex.host(), f.ey.host(), f.ez.host(), f.bx.host(), f.by.host(), f.bz.host()};
    std::vector<cplx> T(P.size());
    for (int c = 0; c < 6; ++c) {
        for (size_t q = 0; q < T.size(); ++q) T[q] = F6[6 * q + size_t(c)];
        for (long q = 0; q < nyz; ++q) dft(&T[size_t(q)], NP, nyz, +1, tmp);
        dft_yz(T, NP, NY, NZ, +1);
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i)
                    h[c][f.ex.idx(i, j, k)] += T[(size_t(d.off[0] + i) * NY + d.off[1] + j) * NZ + d.off[2] + k].real();
    }
    if (!per) {                                             // the sheet field of the open box: Ex0 = 0 ahead of the beam
        std::vector<double> ex0(size_t(NX) + 1, 0.0);
        for (int i = 0; i < NX; ++i) {
            double s = 0;
            for (long q = 0; q < nyz; ++q) s += R[size_t(i) * nyz + q];
            ex0[size_t(i) + 1] = ex0[size_t(i)] + D * s / double(nyz);
        }
        const double c0 = beta > 0 ? ex0[size_t(NX)] : 0.0;
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) h[0][f.ex.idx(i, j, k)] += ex0[size_t(d.off[0] + i)] - c0;
    }
    f.ex.upload(h[0]); f.ey.upload(h[1]); f.ez.upload(h[2]); f.bx.upload(h[3]); f.by.upload(h[4]); f.bz.upload(h[5]);
    comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
}

void add_beam_field(Fields& f, const Field& rho, double beta, const Domain& d, Comm& comm, bool steady) {
    const int NX = d.N[0], NY = d.N[1], NZ = d.N[2];
    const long nyz = long(NY) * NZ;
    // 1. global rho (each rank its owned cells), summed over the ranks
    std::vector<double> R(size_t(NX) * nyz, 0.0);
    {
        const std::vector<double> h = rho.host();
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i)
                    R[(size_t(d.off[0] + i) * NY + d.off[1] + j) * NZ + d.off[2] + k] = h[rho.idx(i, j, k)];
        for (size_t q = 0; q < R.size(); q += size_t(1) << 28)        // allreduce in chunks of an int count
            comm.allreduce_sum(&R[q], int(std::min<size_t>(size_t(1) << 28, R.size() - q)));
    }
    if (steady && d.solver == Solver::RipCC && beta > 0) { steady_beam_field(f, R, beta, d, comm); return; }
    if (d.periodic[0]) { periodic_beam_field(f, R, beta, d, comm); return; }
    if (d.solver == Solver::Yee) throw std::runtime_error("self_field with solver = yee: periodic boxes only");
    // 2. phi per (ky, kz): tridiagonal along x, phi(-1) = phi(NX) = 0
    std::vector<cplx> P(R.begin(), R.end());
    dft_yz(P, NX, NY, NZ, -1);
    // The k_perp = 0 mode (the slice-mean charge: a charged sheet in the periodic y, z box) separately:
    // its phi ~ gamma^2 rhobar L^2 would cancel only to round-off in E_perp, so it gives Ex alone, from
    // the 1D Gauss law Ex0(i+1) - Ex0(i) = h_x rhobar(i). Its dynamics is dEx/dt = -Jx, so any constant
    // is steady: a moving beam (beta > 0) gets Ex0 = 0 ahead of it (a sheet that came from -x; the moving
    // window needs zero fields at its right edge), a charge at rest the symmetric phi(-1) = phi(NX) = 0.
    std::vector<double> ex0(size_t(NX) + 1, 0.0);
    {
        for (int i = 0; i < NX; ++i) {
            ex0[size_t(i) + 1] = ex0[size_t(i)] + d.h[0] * P[size_t(i) * nyz].real() / double(nyz);
            P[size_t(i) * nyz] = 0;
        }
        double c = ex0[size_t(NX)];                                   // beta > 0: Ex0(NX) = 0
        if (beta == 0) {
            c = 0;
            for (double v : ex0) c += v;
            c /= double(NX + 1);                                      // sum_{i=0..NX} Ex0(i) = 0
        }
        for (double& v : ex0) v -= c;
    }
    // Per mode the boost factor is clamped at the cell-resolved pancake, gamma^2 h_x^2 K <= 4 (a >= 0): beyond it the
    // continuum pancake 1 / (gamma k_perp) is thinner than h_x / 2, and the operator with a < 0 has the undamped
    // homogeneous zigzags (-1)^i (A + B i) that carried a field tail ahead of the beam up to the right edge (cut by the
    // first window shift: 1.8e-4 of max rho at gamma 1e4). At a = 0 the solve is local: phi = rho / K (the slice
    // Poisson problem) and Ex = -(rho(i) - rho(i-1)) h_x / 4 inside the beam only; the Gauss law stays exact.
    // Explicit RIP-CC only (Alexander, 2026-11-11, decision 99): iRIP keeps the unclamped Gauss-law field, which the clamp
    // turned into a 1 -> 12 nm emittance jump of the Fig. 4 beam (the clamped field is RIP-CC's local state, not iRIP's).
    const bool clamp = d.solver == Solver::RipCC;
    const double g2 = 1.0 / (1.0 - beta * beta), ihx2 = 1.0 / (g2 * d.h[0] * d.h[0]);
    std::vector<cplx> Q(P.size());                                    // phi / gamma_eff^2 per mode, for Ex
    std::vector<cplx> cp, dp;
    cp.resize(size_t(NX)); dp.resize(size_t(NX));
    for (int j = 0; j < NY; ++j)
        for (int k = 0; k < NZ; ++k) {
            const double sy = std::sin(M_PI * j / NY), sz = std::sin(M_PI * k / NZ);
            const double K = 4 * sy * sy / (d.h[1] * d.h[1]) + 4 * sz * sz / (d.h[2] * d.h[2]);
            const double ih = clamp ? std::max(ihx2, 0.25 * K) : ihx2;   // 1 / (gamma_eff^2 h_x^2)
            const double a = ih - 0.25 * K, b = -2 * ih - 0.5 * K;
            auto at = [&](int i) -> cplx& { return P[size_t(i) * nyz + long(j) * NZ + k]; };
            // Thomas: a phi(i-1) + b phi(i) + a phi(i+1) = -rho(i)
            cplx den = b;
            cp[0] = a / den; dp[0] = -at(0) / den;
            for (int i = 1; i < NX; ++i) {
                den = b - a * cp[size_t(i - 1)];
                cp[size_t(i)] = a / den;
                dp[size_t(i)] = (-at(i) - a * dp[size_t(i - 1)]) / den;
            }
            at(NX - 1) = dp[size_t(NX - 1)];
            for (int i = NX - 2; i >= 0; --i) at(i) = dp[size_t(i)] - cp[size_t(i)] * at(i + 1);
            for (int i = 0; i < NX; ++i) Q[size_t(i) * nyz + long(j) * NZ + k] = at(i) * (ih * d.h[0] * d.h[0]);
        }
    dft_yz(P, NX, NY, NZ, +1);
    dft_yz(Q, NX, NY, NZ, +1);
    const double norm = 1.0 / double(nyz);
    auto phi = [&](int i, int j, int k) {                   // rho point (i + 1/2, j, k); 0 beyond x
        if (i < 0 || i >= NX) return 0.0;
        j = (j % NY + NY) % NY; k = (k % NZ + NZ) % NZ;
        return P[size_t(i) * nyz + long(j) * NZ + k].real() * norm;
    };
    auto gphi = [&](int i, int j, int k) {                  // phi / gamma_eff^2 at the rho point, 0 beyond x
        if (i < 0 || i >= NX) return 0.0;
        return Q[size_t(i) * nyz + long(j) * NZ + k].real() * norm;
    };
    auto phibar = [&](int i, int j, int k) { return 0.5 * (phi(i - 1, j, k) + phi(i, j, k)); };   // at integer x
    // 3. fields of the owned cells
    std::vector<double> ex = f.ex.host(), ey = f.ey.host(), ez = f.ez.host(), by = f.by.host(), bz = f.bz.host();
    for (int k = 0; k < d.n[2]; ++k)
        for (int j = 0; j < d.n[1]; ++j)
            for (int i = 0; i < d.n[0]; ++i) {
                const int gi = d.off[0] + i, gj = d.off[1] + j, gk = d.off[2] + k;
                const size_t q = f.ex.idx(i, j, k);
                const double Ey = -(phibar(gi, gj + 1, gk) - phibar(gi, gj, gk)) / d.h[1];
                const double Ez = -(phibar(gi, gj, gk + 1) - phibar(gi, gj, gk)) / d.h[2];
                ex[q] += -(gphi(gi, gj, gk) - gphi(gi - 1, gj, gk)) / d.h[0] + ex0[size_t(gi)];
                ey[q] += Ey; ez[q] += Ez;
                bz[q] += beta * Ey; by[q] += -beta * Ez;
            }
    f.ex.upload(ex); f.ey.upload(ey); f.ez.upload(ez); f.by.upload(by); f.bz.upload(bz);
    comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
}

void remove_mean_current(Fields& f, const Domain& d, Comm& comm) {
    Field* J[3] = {&f.jx, &f.jy, &f.jz};
    double s[3];
    const int nx = d.n[0], ny = d.n[1];
    for (int a = 0; a < 3; ++a) {
        const FV V = J[a]->view();
        s[a] = par_sum(d.owned(), [=] HD(long q) {
            const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
            return V(i, j, k);
        });
    }
    comm.allreduce_sum(s, 3);
    const double n = double(d.N[0]) * double(d.N[1]) * double(d.N[2]);
    for (int a = 0; a < 3; ++a) {
        const FV V = J[a]->view();
        const double m = s[a] / n;
        for_owned(d, [=] HD(int i, int j, int k) { V(i, j, k) -= m; });
    }
    comm.exchange(d, {&f.jx, &f.jy, &f.jz});
}
