// Yee solver (docs/specs/yee.md, decision 36), the reference for comparisons with RIP-CC.
//
// Staggering (descriptor Yee in stagger.hpp; array index (i, j, k) = lower corner):
//   Ex (i+1/2, j, k),  Ey (i, j+1/2, k),  Ez (i, j, k+1/2)        J_a at E_a
//   Bx (i, j+1/2, k+1/2),  By (i+1/2, j, k+1/2),  Bz (i+1/2, j+1/2, k)
//   rho at the nodes (i, j, k):  div E = rho  and  div B = 0 are kept to round-off with Esirkepov's J.
// One step n -> n+1, B kept at integer time levels by two half steps of Faraday's law (the same
// leapfrog as B at half levels; the push gathers E^n, B^n as with RIP-CC):
//   B^{n+1/2} = B^n - dt/2 curl E^n,   E^{n+1} = E^n + dt (curl B^{n+1/2} - J^{n+1/2}),
//   B^{n+1} = B^{n+1/2} - dt/2 curl E^{n+1}.
// Stable for c dt < (1/hx^2 + 1/hy^2 + 1/hz^2)^{-1/2} and omega_p dt < 2; numerical dispersion
// (2/dt) sin(omega dt/2) = |K|, K_a = (2/h_a) sin(k_a h_a/2): slower than light, the origin of the
// numerical Cherenkov instability of relativistic beams (Pukhov, JCP 418, 109622 (2020)).
#include "fields.hpp"

namespace {
void half_faraday(Fields& f, const Domain& d, double c) {          // B -= c curl E
    const double ix = 1.0 / d.h[0], iy = 1.0 / d.h[1], iz = 1.0 / d.h[2];
    const FV Ex = f.ex.view(), Ey = f.ey.view(), Ez = f.ez.view(), Bx = f.bx.view(), By = f.by.view(), Bz = f.bz.view();
    for_owned(d, [=] HD(int i, int j, int k) {
        Bx(i, j, k) -= c * ((Ez(i, j + 1, k) - Ez(i, j, k)) * iy - (Ey(i, j, k + 1) - Ey(i, j, k)) * iz);
        By(i, j, k) -= c * ((Ex(i, j, k + 1) - Ex(i, j, k)) * iz - (Ez(i + 1, j, k) - Ez(i, j, k)) * ix);
        Bz(i, j, k) -= c * ((Ey(i + 1, j, k) - Ey(i, j, k)) * ix - (Ex(i, j + 1, k) - Ex(i, j, k)) * iy);
    });
}
}  // namespace

void yee_step(Fields& f, const Domain& d, Comm& comm) {
    const double dt = d.dt(), ix = 1.0 / d.h[0], iy = 1.0 / d.h[1], iz = 1.0 / d.h[2];
    half_faraday(f, d, 0.5 * dt);                                // needs E at i+1 (guards valid between steps)
    comm.exchange(d, {&f.bx, &f.by, &f.bz});
    const FV Ex = f.ex.view(), Ey = f.ey.view(), Ez = f.ez.view(), Bx = f.bx.view(), By = f.by.view(), Bz = f.bz.view();
    const FV Jx = f.jx.view(), Jy = f.jy.view(), Jz = f.jz.view();
    for_owned(d, [=] HD(int i, int j, int k) {
        Ex(i, j, k) += dt * ((Bz(i, j, k) - Bz(i, j - 1, k)) * iy - (By(i, j, k) - By(i, j, k - 1)) * iz - Jx(i, j, k));
        Ey(i, j, k) += dt * ((Bx(i, j, k) - Bx(i, j, k - 1)) * iz - (Bz(i, j, k) - Bz(i - 1, j, k)) * ix - Jy(i, j, k));
        Ez(i, j, k) += dt * ((By(i, j, k) - By(i - 1, j, k)) * ix - (Bx(i, j, k) - Bx(i, j - 1, k)) * iy - Jz(i, j, k));
    });
    comm.exchange(d, {&f.ex, &f.ey, &f.ez});
    half_faraday(f, d, 0.5 * dt);
    comm.exchange(d, {&f.bx, &f.by, &f.bz});
}

// At the cell centres (i+1/2, j+1/2, k+1/2).
double yee_max_div_b(const Fields& f, const Domain& d) {
    const double ix = 1.0 / d.h[0], iy = 1.0 / d.h[1], iz = 1.0 / d.h[2];
    const FV Bx = f.bx.view(), By = f.by.view(), Bz = f.bz.view();
    const int nx = d.n[0], ny = d.n[1];
    return par_max_abs(d.owned(), [=] HD(long q) {
        const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
        return (Bx(i + 1, j, k) - Bx(i, j, k)) * ix + (By(i, j + 1, k) - By(i, j, k)) * iy + (Bz(i, j, k + 1) - Bz(i, j, k)) * iz;
    });
}

// At the nodes (i, j, k), the position of rho.
void yee_gauss_residual(const Fields& f, const Domain& d, Field& out) {
    const double ix = 1.0 / d.h[0], iy = 1.0 / d.h[1], iz = 1.0 / d.h[2];
    const FV Ex = f.ex.view(), Ey = f.ey.view(), Ez = f.ez.view(), R = f.rho.view(), O = out.view();
    for_owned(d, [=] HD(int i, int j, int k) {
        O(i, j, k) = (Ex(i, j, k) - Ex(i - 1, j, k)) * ix + (Ey(i, j, k) - Ey(i, j - 1, k)) * iy +
                     (Ez(i, j, k) - Ez(i, j, k - 1)) * iz - R(i, j, k);
    });
}
