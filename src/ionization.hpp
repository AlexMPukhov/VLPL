// Field (tunnel) ionization of ions with a charge state (docs/specs/field_ionization.md, decision 86).
//
// An ionizable species carries the attribute charge_state (I32): its particles have the charge q Z (species q = 1, the unit,
// m the ion mass); push and deposit read Z per particle (particles.cpp, rz_particles.cpp). Every step (stage Ionize, before the
// push) each ion samples the ionization of its levels k = Z, Z + 1, ... in E^n at its position: exponential waiting times with
// the static-field rate w_k(|E|), so several levels can go in one step (exact for a field constant over the step). Every
// level that goes creates one macro-electron of the ion's weight and velocity in the species named by ionization_electrons.
// The random numbers are a hash of (seed, step, species, particle id, level): the same on the CPU and the GPU, for any number
// of threads or ranks (the id is a hash of the particle's position when ionization first sees it).
//
// Rate (atomic units, I = I_k / 27.211 eV, F = |E| / 5.142e11 V/m, Z = k + 1 the charge after the ionization):
//   w = C^2 f(l, m) I (2 F0 / F)^(2 n* - |m| - 1) exp(-2 F0 / (3 F)) [ x exp(-alpha Z^2 F / (I F0)) with Tong-Lin ]
//   n* = Z / sqrt(2 I), F0 = (2 I)^(3/2), C^2 = 2^(2 n*) / (n* Gamma(n* + l* + 1) Gamma(n* - l*)) with l* = n* - 1,
//   f(l, m) = (2 l + 1) (l + |m|)! / (2^|m| |m|! (l - |m|)!)
// (Ammosov, Delone, Krainov 1986, the tunnelling limit of Perelomov, Popov, Terent'ev; the form of Smilei), l the orbital of
// the removed electron (data/ionization_energies.def), |m| = 0 by default. Hydrogen: the Landau rate 4 / F exp(-2 / (3 F)).
// Beyond its maximum (barrier suppression) the expression falls again; the rate is kept at its maximum there. Tong-Lin
// (J. Phys. B 38, 2593 (2005), alpha = 6) corrects the over-the-barrier regime.
//
// Energy-conserving ionization current (Nuter et al., Phys. Plasmas 18, 033107 (2011); Mulser et al.): the field pays I_k for
// every ionization. Here the new electron is born displaced by d = I_k E / (q_e |E|^2) (the tunnel exit), and the Esirkepov
// current of that displacement is deposited with the push: its work is exactly q_e w d . E = w I_k, and the charge stays
// conserved (the Nuter current J = I_k E / (|E|^2 dt) of a point dipole), so the Gauss law holds at round-off.
#pragma once
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include "portable.hpp"

struct IonizationConfig {
    bool on = false;
    // a charge state per particle without field ionization: set by a collisions block with ionizing = true for its ion species
    // (impact ionization, collisions.hpp); the species then has q = 1 and the attribute charge_state as with on
    bool states = false;
    std::string element;               // symbol (summary), empty for explicit potentials
    int z0 = 0, zmax = 0;              // initial charge state, highest reachable
    std::vector<double> ip_eV;         // ip_eV[k], k = 0 .. zmax - 1
    std::vector<int> l;                // l[k]
    int m = 0;                         // |m| of all levels (default 0: the largest rate)
    bool tong_lin = false;             // ionization_bsi = tong_lin
    double tl_alpha = 6;
    std::string electrons;             // species of the new electrons
    bool current = true;               // ionization_current: the energy-conserving current
};

// One level k -> k + 1 in code units (E in m c omega0 / e, w in omega0): w = exp(logpre - power ln Ee + expo / Ee - tl Ee),
// Ee = min(E, epeak); 0 below emin (w(emin) dt = 1e-18: never sampled at double resolution).
struct IonLevel {
    double logpre, power, expo, tl, epeak, emin, ip;    // ip: I_k in m c^2
};
HD inline double ion_rate(const IonLevel& L, double E) {
    if (!(E > L.emin)) return 0.0;
    const double e = E < L.epeak ? E : L.epeak;
    return exp(L.logpre - L.power * log(e) + L.expo / e - L.tl * e);
}
// lambda0 [m], dt [1/omega0]
IonLevel ion_level(double ip_eV, int zafter, int l, int m, bool tong_lin, double alpha, double lambda0, double dt);
// rate in SI [1/s] of the same formula (tests): field E [V/m]
double ion_rate_si(double ip_eV, int zafter, int l, int m, bool tong_lin, double alpha, double E_Vm);

// |E| in the rest frame of a particle with momentum u (c = 1, fields in m c omega0 / e) and its gamma:
//   |E_rest|^2 = gamma^2 (|E + v x B|^2 - (v . E)^2) = |gamma E + u x B|^2 - (u . E)^2
// (the boosted frame, decision 90: ions at rest in the lab stream at -beta c there and see the lab field in their own frame).
HD inline double ion_rest_field(const double E[3], const double B[3], double ux, double uy, double uz, double& g) {
    g = sqrt(1.0 + ux * ux + uy * uy + uz * uz);
    const double fx = g * E[0] + uy * B[2] - uz * B[1], fy = g * E[1] + uz * B[0] - ux * B[2], fz = g * E[2] + ux * B[1] - uy * B[0];
    const double ue = ux * E[0] + uy * E[1] + uz * E[2], s2 = fx * fx + fy * fy + fz * fz - ue * ue;
    return s2 > 0 ? sqrt(s2) : 0.0;
}

// counter-based random numbers (splitmix64), uniform in (0, 1)
// Not inlined on the GPU: the ptxas of CUDA 12.9 and 13.0 miscompiles this mix for sm_120 when it is inlined into the QED
// sub-step loops (qed.hpp): the same key gave other random numbers than on the CPU, and the counting and the writing pass
// of the photon emission and the pair decay disagreed (pairs P5 / P6 on the RTX 5090). CUDA 13.5 and the driver's JIT
// compile it correctly; with __noinline__ every CUDA version gives the CPU's numbers.
#ifdef __CUDA_ARCH__
__noinline__
#endif
HD inline unsigned long long ion_mix(unsigned long long z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
HD inline double ion_uniform(unsigned long long key) {
    return (double(ion_mix(key) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
// Levels that go within dt (from charge state z, at most zmax): exponential waiting times, one draw per level k (key + k).
HD inline int ion_sample(const IonLevel* lev, int z, int zmax, double E, double dt, unsigned long long key) {
    double left = dt;
    int k = z;
    while (k < zmax) {
        const double w = ion_rate(lev[k], E);
        if (!(w > 0)) break;
        const double tau = -log(ion_uniform(key + (unsigned long long)k)) / w;
        if (tau >= left) break;
        left -= tau;
        ++k;
    }
    return k - z;
}

class Simulation;
class Operator;
class Species;
struct Config;
// Registers charge_state (I32, initial value z0) and id (I64) on an ionizable species (before it is loaded).
void ionization_register(Species& sp, const IonizationConfig& ic);
// The stage-Ionize operator of the deck's ionizable species (null if none); its setup checks the species and registers
// charge_state and id.
std::unique_ptr<Operator> make_ionization_operator();
// The current of the displacements of the electrons born in this step (stage Push, after the particle deposit, before the
// guards are added; nothing if none or ionization_current = false).
void ionization_currents(Simulation& s);
// dir/ionization.txt at the cadence of diag.txt (collective): energies and the charge-state distributions; wtot = W_field + W_kin.
void ionization_diagnostics(Simulation& s, bool root, double wtot);
void ionization_summary(FILE* f, const Config& c);
// Totals since the start, summed over the ranks (collective): sum I_k w, the work of the ionization current, events, shortened
// displacements.
struct IonizationTotals { double e_ion = 0, e_cur = 0; long events = 0, capped = 0; };
IonizationTotals ionization_totals(Simulation& s);
