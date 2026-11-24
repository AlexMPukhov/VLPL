// Collisional (impact) ionization in the binary collisions (docs/specs/collisions.md, decision 94): the method of Perez et al.,
// Phys. Plasmas 19, 083104 (2012), section III, as in Smilei 5.1 (src/Collisions/CollisionalIonization.cpp).
//
// For an electron-ion pair of a collisions block with `ionizing = true`, in the rest frame of the ion: the electron of kinetic
// energy E ionizes the ion of charge Z* with the cross-section sigma(E) of the relativistic binary-encounter Bethe model
// summed over all orbitals of the ion (occupation N, binding energy B; Kim, Santos, Parente, Phys. Rev. A 62, 052710 (2000)),
// with the probability 1 - exp(-D v' sigma / gamma_e ...) per step (D the density-time factor of the pair, as for the Coulomb
// collision); several ionizations in one step follow Nuter et al., Phys. Plasmas 18, 033107 (2011). The incident electron
// loses the mean lost energy (binding + secondary), the new electron gets the mean secondary energy, along the incident
// direction in the ion frame, at the ion's position, with the ion's weight. Weight rule: the ion is ionized (and the electron
// created) with the probability w_e / w_i, the incident electron slowed with w_i / w_e (both at most 1).
// Tables per charge state on 100 energies, 1 eV .. 10 MeV (log), in code units: sigma k0 r_e-normalized as Smilei
// (r_e omega / 2 c prefactor), energies in m c^2.
// Orbitals: the neutral-atom binding energies of data/binding_energies.def (Smilei's table) shifted so that the outermost one
// of the ion is its NIST ionization energy (data/ionization_energies.def).
#pragma once
#include <vector>
#include "portable.hpp"

constexpr int IMPACT_N = 100;                       // energies per table
constexpr double IMPACT_A1 = 510998.9;              // m c^2 / E_min (E_min = 1 eV)
constexpr double IMPACT_A2 = 6.142165;              // (IMPACT_N - 1) / ln(E_max / E_min), E_max = 10 MeV

struct ImpactTable {                                // device view: [z * IMPACT_N + i], z = 0 .. zn - 1
    const double *cs = nullptr, *te = nullptr, *le = nullptr;
    int zn = 0;
};

// Host: the tables of an element (atomic number Z <= 100, NIST energies ip_eV[0 .. Z-1]), k0 r_e (code units).
// cs: sum over the orbitals of the RBEB cross-section; te: mean secondary energy; le: mean energy lost by the incident
// electron (both in m c^2).
void impact_tables(int Z, const std::vector<double>& ip_eV, double kre, std::vector<double>& cs, std::vector<double>& te,
                   std::vector<double>& le);
// Host, for tests: the cross-section sum of the formula (not the table) for the ion Z* at the electron kinetic energy ek
// (m c^2), in the units of the tables.
double impact_cross_section(int Z, const std::vector<double>& ip_eV, double kre, int zstar, double ek);
bool impact_element_ok(int Z);                      // the binding-energy table has Z

// Interpolation of the tables at the ion-frame Lorentz factor gs (Smilei: linear in x = a2 ln(a1 (gs - 1)), extrapolated
// linearly above 10 MeV for cs). Returns false below 1 eV.
HD inline bool impact_lookup(const ImpactTable& T, int z, double gs, double& cs, double& w, double& e) {
    const double x = IMPACT_A2 * log(IMPACT_A1 * (gs - 1));
    if (!(x >= 0)) return false;
    const double* C = T.cs + z * IMPACT_N; const double* W = T.te + z * IMPACT_N; const double* E = T.le + z * IMPACT_N;
    if (x < IMPACT_N - 1) {
        const int i = int(x);
        const double a = x - i;
        cs = (C[i + 1] - C[i]) * a + C[i]; w = (W[i + 1] - W[i]) * a + W[i]; e = (E[i + 1] - E[i]) * a + E[i];
    } else {
        const double a = x - (IMPACT_N - 1);
        cs = (C[IMPACT_N - 1] - C[IMPACT_N - 2]) * a + C[IMPACT_N - 1]; w = W[IMPACT_N - 1]; e = E[IMPACT_N - 1];
    }
    return true;
}

// One electron-ion pair (Smilei's CollisionalIonization::calculate). ue: electron u (changed when it loses energy), ui: ion u,
// z: the ion's charge state (incremented per ionization when the ion side is selected), we, wi weights, D the density-time
// factor, U1, U2 uniforms. Each ionization calls born(u_new[3], k) (k = the charge state before it) when the ion side is
// selected (U2 < we / wi). Returns the number of ionizations of the ion. eion: sum of the ionization energies (m c^2) of them.
template <class Born>
HD inline int impact_pair(const ImpactTable& T, const double* ip, double ue[3], const double ui[3], int& z, double we, double wi,
                          double D, double U1, double U2, Born born, double& eion) {
    eion = 0;
    if (z >= T.zn) return 0;
    double ge = sqrt(1 + ue[0] * ue[0] + ue[1] * ue[1] + ue[2] * ue[2]);
    const double gi = sqrt(1 + ui[0] * ui[0] + ui[1] * ui[1] + ui[2] * ui[2]);
    double gs = ge * gi - ue[0] * ui[0] - ue[1] * ui[1] - ue[2] * ui[2];   // electron Lorentz factor in the ion frame
    double K = D * sqrt(gs * gs - 1) / gi;
    const bool ion_side = U2 * wi < we, el_side = U2 * we < wi;
    const int kmax = T.zn - z - 1;
    double rate[128], irate[128], cum = 0;
    int n = 0, zl = z;                               // zl: the level of the tables (advances as in Smilei, also on the electron side only)
    for (int k = 0; k <= kmax && k < 128; ++k, ++zl) {
        double cs, w, e;
        if (!impact_lookup(T, zl, gs, cs, w, e)) break;
        if (e > gs - 1) break;
        rate[k] = K * cs / ge;
        irate[k] = 1 / rate[k];
        const double pk = exp(-rate[k]);
        if (k == 0) cum = pk;
        else
            for (int p = 0; p < k; ++p) {
                double cp = 1 - rate[k] * irate[p];
                for (int j = 0; j < p; ++j) cp *= 1 - rate[p] * irate[j];
                for (int j = p + 1; j < k; ++j) cp *= 1 - rate[p] * irate[j];
                cum += (pk - exp(-rate[p])) / cp;
            }
        if (U1 < cum) break;
        const double p2 = gs * gs - 1;
        if (ion_side) {
            const double pr1 = sqrt(w * (w + 2) / p2), pr2 = w + 1 - pr1 * gs;
            const double un[3] = {ue[0] * pr1 + ui[0] * pr2, ue[1] * pr1 + ui[1] * pr2, ue[2] * pr1 + ui[2] * pr2};
            born(un, zl);
            eion += ip[zl];
            ++n;
            ++z;
        }
        if (el_side) {
            double pr = sqrt(((gs - e) * (gs - e) - 1) / p2);
            ue[0] *= pr; ue[1] *= pr; ue[2] *= pr;
            ge *= pr; K *= pr;
            pr = (1 - pr) * gs - e;
            ue[0] += pr * ui[0]; ue[1] += pr * ui[1]; ue[2] += pr * ui[2];
            ge += pr * gi;
            gs -= e;
        }
    }
    return n;
}
