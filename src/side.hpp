// Side injection (docs/specs/side_injection.md): a total-field / scattered-field tube with faces y = const and / or
// z = const (the inner faces of the CPML layers by default). Inside the tube the main grid holds the total field,
// outside it the scattered field, which the CPML absorbs. Every two-point y (z) difference of the step whose two values
// lie on different sides of a face gets the incident value of the far point (+ for a total-field output point, - for a
// scattered one): in the predictor (RIP-CC), in the transport sources and in the corrector. The x transport never mixes
// y, z cells, so nothing else changes; for iRIP the corrections go into the right-hand side of both passes (the
// incident Ex, Bx of t_n+1 are known), the implicit operator is unchanged.
//
// The incident field is a sum of plane waves, each an exact discrete solution of the solver: the vacuum dispersion
//   RIP-CC  sin^2(w D/2) = sin^2(kx D/2) + D^2 Lam/4 cos^2(kx D/2),   iRIP  tan^2(w D/2) = tan^2(kx D/2) + D^2 Lam/4,
//   Lam = 4/hy^2 sin^2(ky hy/2) + 4/hz^2 sin^2(kz hz/2),
// and the polarisation from the null space of the one-step matrix (both polarisations are degenerate). The leakage is
// then at round-off for any angle (research/side_tfsf/). Two incident fields (`side.wave`):
// - plane: one plane wave, continuous or a pulse; the corrections summed over the components every step;
// - flying_focus: a luminal Bessel pulse (all components with w - kx = kappa, src/side_luminal.cpp). In the window
//   moving at c from t = 0 the incident field is a fixed pattern times e^{-i kappa t}: the corrections at the cells
//   next to the faces are tabulated once.
//
// Background plasma (`side.background_density`, decision 63): the incident field is the wave of a uniform plasma, and
// outside the kinetic region (the particles end 2 cells inside the faces) a linear cold fluid carries the plasma
// current, j^{n+1/2} = j^{n-1/2} + D wp^2 F (Ex, mux Ey, mux Ez)^n (F: the fluid fraction of the point, the complement
// of the particles' coverage at the kinetic edge, 0 in the CPML). It is driven by the stored field (total in the tube,
// scattered outside), so the incident wave is an exact discrete solution again:
//   RIP-CC  TE (Ex = 0): sin^2(w D/2) = sin^2(kx D/2) + D^2/4 cos^2(kx D/2) (Lam + wp^2),
//           TM (Bx = 0): the same with wp^2 (1 - D^2 Lam / 4),
//   iRIP    both:        tan^2(w D/2) = tan^2(kx D/2) + D^2/4 (Lam + wp^2 / cos^2(w D/2))
// (research/side_tfsf/check_fluid.py: the one-step matrix to 2e-14). With RIP-CC every component is a TE and a TM
// wave of its own wave vector.
#pragma once
#include <complex>
#include <vector>
#include "comm.hpp"
#include "grid.hpp"

struct SideConfig {
    bool on = false;
    int wave = 0;                            // 0: plane wave, 1: flying focus (luminal Bessel pulse)
    bool faces[3] = {false, false, false};   // axes 1 (y) and 2 (z)
    int lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};   // total-field nodes lo..hi (global index) per axis with faces
    double a0 = 0;                           // field amplitude (as laser.a0); flying focus: the peak on the axis
    double tau = 0;                          // plane: envelope as LaserConfig (exp(-4 ln2 t^2 / tau^2)); 0: continuous
    double tc = 0;                           // plane: time of the peak at the reference point r0
    double phase = 0;                        // plane: carrier phase; flying focus: the carrier-envelope phase
    double angle = 0, elevation = 0;         // plane: direction (cos e cos a, cos e sin a, sin e)
    char pol = 'z';                          // plane 'z': E along n x (z x n), 'y': along z x n; flying focus: E on the axis
    double r0[3] = {0, 0, 0};                // plane: reference point; flying focus: x of the envelope centre at t = 0, axis y, z
    double cone = 0;                         // flying focus: cone angle theta at omega0 (kappa = omega0 (1 - cos theta))
    double len = 0;                          // flying focus: 1/e half length of the field on the axis
    double density = 0;                      // background plasma (n_c, wp^2 in omega0^2): the incident wave of that plasma
    bool fluid_all = false;                  // the cold fluid everywhere (no kinetic region: tests, a linear plasma)
    int order = 1;                           // the particle shape (the coverage of the kinetic edge)
    int absorb[3] = {0, 0, 0};               // CPML cells per axis (no fluid there)
};

class SideTfsf {
public:
    struct Comp {                            // one plane wave: phase k . (i D, j hy, k hz) - w t at the stored index
        double k[3], w;
        double v[12];                        // complex amplitudes (re, im) of Ex, Ey, Ez, Bx, By, Bz at the stored index
        double pr[4];                        // RIP-CC predictor Ex*, Bx* of the mode at t_n (complex)
        double mu[2];                        // (1 + e^{-i w D}) / 2: the mean of t_n and t_n+1
        double j[6];                         // background plasma: the fluid current j^{n-1/2} (Jx, Jy, Jz) at t_n
    };
    // a component as the provider gives it: wave vector, w, amp E along e; cls -1: both polarisations (degenerate),
    // 0 / 1: the TE / TM wave of a pair (RIP-CC in plasma: the TM wave follows its TE wave, the pair fits amp e)
    struct Wave {
        double k[3], w, amp[2], e[3];
        int cls = -1;
    };
    // t_span: plane-wave pulses: the time over which no periodic image may appear (run time + crossing time)
    SideTfsf(const Domain& d, const SideConfig& c, double t_span);
    const SideConfig& config() const { return c_; }
    const std::vector<Comp>& comps() const { return comps_; }
    bool tabulated() const { return c_.wave == 1; }
    double kappa() const { return kappa_; }
    int frequencies() const { return int(gx_.size()); }       // flying focus: frequencies (azimuths = comps / this)
    // At the start of the field step t_n -> t_n+1 (t = t_n): the time factors (host, uploaded).
    void begin(double t);
    // The corrections, added to the owned cells before the exchange that follows each kernel:
    // the RIP-CC predictor Ex*, Bx* (t_n); the transported Ey, Ez, By, Bz (kind 0: Ex*, Bx* of the incident field as the
    // RIP-CC predictor, kind 2: the mean of t_n and t_n+1 as iRIP); the corrector Ex, Bx (mean of t_n and t_n+1).
    void fix_pred(const FV Exs, const FV Bxs) const;
    void fix_transport(const FV nEy, const FV nEz, const FV nBy, const FV nBz, int kind) const;
    void fix_corr(const FV Ex, const FV Bx) const;
    // O5 (decision 76): the incident means (t_n, t_n+1) of Ex, Bx added to the owned tube points (masked), plain (closed =
    // false) or with the closure factor of each component (closed = true: the incident Ex*, Bx* of o5). The time of begin().
    void o5_means(const FV ex, const FV bx, bool closed);
    // The incident field at t added to the owned cells of f: in the tube only (masked) or everywhere (exchange afterwards).
    // jf (background plasma): the fluid currents, which get F j^{t - D/2} of the incident wave.
    void add_incident(const FV f[6], double t, bool masked = true, const FV* jf = nullptr);
    // incident component c (0..5: Ex .. Bz, 6..8: the plasma current j^{t - D/2} at full density) at the global stored
    // index (absolute x index) and time t (host; slow)
    double incident(int c, long i, long j, long k, double t) const;
    bool total(int c, long j, long k) const;            // the point of component c is in the tube
    double polarisation_mismatch() const { return mism_; }   // largest |E_mode - E_wanted| / |E_wanted| of a component
    double mode_residual() const { return res_; }       // largest |G v - lambda v| / |v| of a component (one-step matrix)
    // background plasma: the cold fluid (fluid fraction F of Jx, Jy, Jz at the global (j, k), host); one step of the
    // fluid currents with E^n, added to the deposited j (after the guards are reduced)
    bool fluid() const { return c_.density > 0; }
    double fluid_fraction(int c, long j, long k) const;
    void fluid_current(const FV E[3], const FV Jf[3], const FV J[3]) const;
    // the components of the providers (src/side.cpp, src/side_luminal.cpp); omega: the discrete frequency of the
    // polarisation cls (0 TE, 1 TM; vacuum and iRIP: the same) in a plasma of wp2 = wp^2
    static std::vector<Wave> plane_waves(const Domain& d, const SideConfig& c, double t_span);
    static std::vector<Wave> luminal_waves(const Domain& d, const SideConfig& c, std::vector<int>& groups);
    static double omega(const Domain& d, const double k[3], bool irip, double wp2 = 0, int cls = 0);
    // the polarisations of the solver in the plasma of c: one degenerate pair (-1) or TE and TM apart (RIP-CC, wp2 > 0)
    static bool split(const Domain& d, const SideConfig& c) { return d.solver != Solver::IRip && c.density > 0; }
    void build_table();                      // flying focus: the corrections at the face columns (side_luminal.cpp)
    void fill_luminal(const FV f[6], double t, bool masked, const FV* jf);   // flying focus: the incident field, separable sum
    void means_luminal(const FV ex, const FV bx, bool closed);               // flying focus: o5_means
private:
    const Domain& d_;
    SideConfig c_;
    std::vector<Comp> comps_;
    Buf<double> dev_;                                   // the components on the device (sizeof(Comp) / 8 per component)
    Buf<double> tab_;                                   // per component e^{i kx D}, e^{i ky hy}, e^{i kz hz}, tables in y, z
    Buf<double> tx_;                                    // plane waves: per component the table along x (local i)
    Buf<double> ft_;                                    // plane waves: per component e^{i (kx x0 - w t)} of the current step
    Buf<int> jk_;                                       // the local (j, k) columns next to a face
    int nb_ = 0;
    // flying focus: frequency groups (start index of each group in comps_, and the end), their k_x; the table of the
    // corrections per face column and local x (8 complex: Ex*, Bx*, Ey, Bz, Ez, By, Ex, Bx) and the step factor
    std::vector<int> groups_;
    std::vector<double> gx_;
    Buf<double> H_;
    double kappa_ = 0, ftk_[2] = {1, 0};
    double t_ = 0, mism_ = 0, res_ = 0;
    Buf<double> fl_;                                    // background plasma: the fluid fraction F of Jx, Jy, Jz per local (j, k)
};

// One exact discrete plane wave of the solver of d (side.cpp; also the TFSF x plane with exact plane waves,
// tfsf_modes.cpp): the eigenvector of the one-step matrix for the wave vector k and frequency w (on the discrete
// dispersion, plane_omega) whose E at the physical points is closest to the wanted complex vector E, with the phase
// reference k . (r - r0) - w (t - tc). mism: the largest relative E mismatch so far (updated).
SideTfsf::Comp plane_mode(const Domain& d, const double k[3], double w, const std::complex<double> E[3], const double r0[3],
                          double tc, double& mism);
double plane_omega(const Domain& d, const double k[3]);   // the discrete vacuum frequency of k (RIP-CC or iRIP)
// O5 (d.o5, iRIP): the closure factor of a plane wave, Cx(kx) g_perp(ky, kz) (cx = false: g_perp only); 1 otherwise
double o5_factor(const Domain& d, const double k[3], bool cx = true);
