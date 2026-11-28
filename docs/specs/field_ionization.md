# Field ionization (Phase 3, first block)

Status: implemented (branch field-ionization, decision 86); physics choices provisional, for Alexander's review.

## Physics and equations
Ions carry a charge state Z per particle (attribute `charge_state`, I32); their charge is q Z with the species q = 1 (e)
and m the ion mass. Each level k -> k + 1 ionizes with the static-field tunnel rate in the instantaneous |E| at the ion
(the laser carrier is resolved, so no cycle averaging). Atomic units: I = I_k / 27.211 eV, F = |E| / 5.142e11 V/m,
Z = k + 1 (the charge after the ionization):

    w = C^2 f(l, m) I (2 F0 / F)^(2 n* - |m| - 1) exp(-2 F0 / (3 F))          [x exp(-alpha Z^2 F / (I F0)) Tong-Lin]
    n* = Z / sqrt(2 I),  F0 = (2 I)^(3/2),  C^2 = 2^(2 n*) / (n* Gamma(n* + l* + 1) Gamma(n* - l*)),  l* = n* - 1,
    f(l, m) = (2 l + 1) (l + |m|)! / (2^|m| |m|! (l - |m|)!)

ADK (Ammosov, Delone, Krainov, JETP 64, 1191 (1986)), the tunnelling limit of PPT (Perelomov, Popov, Terent'ev 1966), in
the form of Smilei (`ionization_model = "tunnel"`, Nuter et al. 2011): l is the orbital quantum number of the removed
electron (data file, below), |m| = 0 (the largest rate; deck key `ionization_m`). For I = 1/2, l = 0 the formula is the
Landau rate of hydrogen, 4/F exp(-2/(3F)). The expression has a maximum at F = (2/3) F0 / (2 n* - |m| - 1) and falls beyond
it (unphysical, over the barrier); the rate is held at its maximum there (as GEM-PIC-RZ). Option `ionization_bsi =
tong_lin`: the Tong-Lin factor (J. Phys. B 38, 2593 (2005); alpha = 6, `ionization_tl_alpha`) for the barrier-suppression
regime, again held at the maximum of the corrected expression.

Energies: NIST ASD 5.12, every charge state of every element with a complete set, H .. Lr (Z = 1 .. 103) and D, retrieved
2026-11-08, in `data/ionization_energies.def` (X-macro data file, 5357 levels), built by `tools/make_ionization_table.py` from the
NIST downloads (queries in the script; Rf .. Ds lack stages in the ASD and are left out). NIST's brackets of non-measured values
are kept as comments. l per level: the last subshell of NIST's ground configuration of the ion of charge k. Masses: NIST
standard atomic weights (interval midpoints; [A] for elements without one). The energies of the 21 elements of GEM-PIC-RZ's
table are identical (its NIST 5.12 retrieval of 2026-09-27).

Difference to GEM-PIC-RZ: its rate is the s-state form with C^2 = 2^(2n*) / (n* Gamma(n* + 1) Gamma(n*)) (that is l* = 0,
although its comment says l* = n* - 1) and f = 1. Ratio this / GEM-PIC-RZ: H 1.000, He 1.278, N0 3.10 (l = 1), N4+ 0.390
(n* = 1.86), N5+ 1.059, Ar0 3.22, Ar7+ 0.192 (n* = 2.46), Kr8+ 1.36 (l = 2, C^2 ratio 0.27), Xe0 2.83, Au0 0.80. Because of
the exponential field dependence, the ionization threshold moves only by a few % in field.

## Discretisation
Stage Ionize, before the push, every step. For every ion: E^n gathered at x^n with the push's gather (Cart3D/2D: the
staggered shapes; RZ: the mode sum, unsmoothed); the levels k = Z, Z + 1, ... ionize by exponential waiting times in the
constant field over the step: tau_k = -ln(u_k) / w_k(|E|), taken while their sum stays below dt. Exact sampling of the
rate equations for a field constant over the step, any w dt (several levels per step). Below emin (w dt = 1e-18) the rate
is zero (no exp evaluated). Random numbers: u_k = hash(seed, step, species, id, k) (splitmix64), id = a hash of the ion's
position when ionization first sees it (its load position: the same on any decomposition, kept through migrations and
checkpoints). The same draws on the CPU and the GPU, any thread or rank count, any particle order.

New electrons: one macro-electron per ionized level, weight of the ion, velocity of the ion (u = gamma v), appended to the
species `ionization_electrons` in the order of their ions (count, exclusive scan (CUB on the GPU), fill: reproducible).
Charge-neutral: the ion gains + w, the electron brings - w at the same point; the Gauss law holds at round-off.

Ionization current (`ionization_current = true`, default): the energy I_k w comes from the field. The electron is born at
x_ion + d with d = I_k E / (q_e |E|^2) (the tunnel exit, ~0.5 nm for H at its threshold), and the Esirkepov current of the
displacement x_ion -> x_ion + d (RZ: one charge-conserving segment per mode) is deposited into J^{n+1/2} with the push. Its
work is q_e w d . E^n = w I_k exactly (to the time centring E^n vs (E^n + E^{n+1})/2), and the current is the Nuter et al.
current J = sum I_k w E / (|E|^2 dt) of a point dipole, so charge conservation and the Gauss law stay exact. |d| is capped
at half a cell (counted in ionization.txt; never reached at physical fields). E_ion = sum I_k w and E_cur = the work of the
displacements are accumulated; W_field + W_kin + E_cur is conserved.

Push and deposit read Z per particle (q Z / m in the push, q Z w in the current and the charge); species without the
attribute are unchanged (bit-identical).

## Boosted frame (decision 90, provisional)
In a run with `boost` (docs/specs/boosted_frame.md) the ions at rest in the lab stream at -beta c, and the field of the
simulation frame is not the field that ionizes them. The ADK rate is a rest-frame rate, so per ion (momentum u^{n-1/2},
gamma_i = sqrt(1 + u^2)):
- the field is |E| in the ion's rest frame, from E^n and B^n gathered at the ion (the push's gather; RZ: the mode sum):

      |E_rest|^2 = gamma_i^2 (|E + v x B|^2 - (v . E)^2) = |gamma_i E + u x B|^2 - (u . E)^2,

  for an ion at rest in the lab exactly the lab field (a laser of amplitude E_lab has E' = gamma (1 - beta) E_lab in the
  boosted frame; the streaming ion sees E_lab again);
- the time is its proper time over the step, dt / gamma_i (the waiting times are sampled in the rest frame);
- the new electrons get the ion's velocity, as in the lab;
- the energy-conserving current (corrected 2026-11-28): in the ion's rest frame the tunnel-exit dipole
  d_r = I_k E_r / (q_e |E_r|^2) takes from the field the energy I_k and the momentum q_e d_r x B_r = I_k S_r,
  S_r = E_r x B_r / |E_r|^2 (c = 1; |S_r| = 1 for a plane wave: the absorbed photons' momentum I_k / c). Seen from the
  simulation frame this 4-momentum has the energy

      I'_k = (gamma_i + u . S_r) I_k     (ion_energy_factor in src/ionization.hpp; E_r, B_r by the Lorentz transformation),

  gamma (1 - beta) I_k for a forward laser in a frame boosted by gamma (the Doppler factor of the laser energy itself),
  gamma (1 + beta) I_k for a backward wave. The displacement d = I'_k E / (q_e |E|^2) along the simulation-frame E does
  exactly that work, q_e w d . E = I'_k w, and the Gauss law stays at round-off. E_ion = sum I'_k w is the energy in the
  simulation frame (E_cur / E_ion = 1 as in the lab). Until 2026-11-28 the code took gamma_i I_k (the transfer (I_k, 0),
  without the photons' momentum): 1 / (1 - beta) times too much for a forward laser, 7.5 at gamma = 2, 17.5 at gamma = 3.
  This went unnoticed because the charge states (B4) do not depend on it; it showed as a laser depletion about three
  times too large in a gamma = 3 run through helium. B4 now checks I'_k / I_k against the lab run. The mass change of the ion (m_e, I_k) is neglected everywhere. The column W_field + W_kin + E_cur of ionization.txt is no check in
  the boosted frame: the plasma brings kinetic energy through the injection plane and takes it out at the left edge, and a
  new electron is born with the kinetic energy (gamma_i - 1) m_e c^2 of the streaming ion's velocity (the ion keeps its mass).
- ids: the plasma enters through the injection plane in every step; the hash of the load position also mixes in the step
  of the ion's first ionization pass (the step after its injection, the same on any decomposition), so two lattice points
  that happen to be loaded at the same x' in different steps get different ids.

Lab runs (no `boost`) are unchanged and bit-identical: the simulation-frame |E| and dt, the hash without the step.
Choices for Alexander (decision 90): the rest-frame rate only in the boosted frame (in a lab run the ions move slowly; a
lab run with relativistic ions would need the same transformation); (gamma_i + u . S_r) I_k as the energy taken (gamma_i I_k
until 2026-11-28, a bug); the step in the id.

## Deck interface
```
species N {
  element              = N           # symbol: mass (standard atomic weight) and the NIST energies, l per level
  charge_state         = 0           # initial (without ionization: the charge of the species, instead of `charge`)
  ionization           = adk         # none | adk
  ionization_electrons = electrons   # species of the new electrons (q < 0)
  max_charge_state     = 7           # default: Z of the element (the number of potentials)
  ionization_current   = true        # the energy-conserving current
  ionization_bsi       = none        # none | tong_lin
  ionization_tl_alpha  = 6
  ionization_m         = 0           # |m| of all levels
  ionization_potentials = 552.07, 667.05 eV   # optional: replace the table (k = 0, 1, ...); ionization_l likewise
  density              = 1e-3        # atom density (n_c)
}
```
Output: `ionization.txt` at the cadence of diag.txt (E_ion, E_cur, W_field + W_kin + E_cur, capped displacements, per species
the events and the charge-state distribution as sums of weights). Boosted frame: see above (examples/ionization_boost.deck,
rz_ionization_boost.deck). Not yet: the implicit mover (O6), self_field, ionization of
the bound-electron medium (O7c), collisional ionization (next block).

## Tests
- tests/test_ionization.cpp (I1-I7): rates (Landau identity, Python reference, the cap), the MC scheme in a constant field
  (H: exp(-w t); Ar0..4+ at w_1 dt = 0.5: the rate equations), order independence, C0 -> C4+ after a Gaussian pulse against
  tools/ionization_ref.py, the energy balance of the current, the Gauss law with electrons created (Cart3D, 2D, RZ), the
  charge per particle in the push, deck checks. Numbers: section "Results".
- tests/mpi.sh: examples/ionization_2d.deck and examples/rz_ionization.deck on 1 and NP ranks (diag.txt and ionization.txt);
  the boosted frame: examples/ionization_boost.deck (2D) and rz_ionization_boost.deck.
- Boosted frame: tests/test_boost.cpp B4 (`build/tests test_boost`; `BOOST_ION_ONLY=1` runs B4 alone): the rest-frame field
  against the Lorentz transformation; a laser through a nitrogen slab, the charge-state fractions of the boosted run
  (gamma = 2) against the lab run within the statistics (2D and RZ at a0 = 0.0105, 2D at a0 = 0.03 with up to three levels
  per ion), the Gauss law, E_cur = E_ion.
- bench/ionization_smilei/: against Smilei 5.1 (hydrogen slab absorbing ~20 % of the pulse, carbon tracer).

## Results (2026-11-08, the CPU cluster CPU; cuda RTX 5090 GPU)
- Rates: Landau identity 5.6e-16, code tables vs SI 3.8e-15, tools/ionization_ref.py 3.3e-11.
- Constant field: H vs exp(-w t) 1.11 sigma; Ar0..4+ at w_1 dt = 2.3 and 0.5 vs the RK4 rate equations 1.39 sigma; the
  same counts with the ion order reversed (0 265 30778 1725 0).
- C0 -> C4+ after a Gaussian pulse: C2+ / C3+ 0.5632 / 0.4368 vs Python 0.5608 / 0.4392 (1.23 sigma).
- Energy: field energy lost = E_cur to 4.6e-4 (771 events, immobile electrons), E_cur / E_ion - 1 = 7e-16; with mobile
  electrons the current removes 0.994 of E_ion from W + K. Gauss law with electrons created 1e-16 (Cart3D, 2D, RZ).
- MPI (tests/mpi.sh, 1 vs 8 ranks): ionization.txt identical (2D: 208877 events, E_ion 3.222084582253e-3; RZ: 321337
  events, E_ion 1.755332875147e-2), diag.txt 2.6e-13 (2D) / 0 (RZ), Gauss drift 1.3e-14 / 6.2e-15.
- GPU (RTX 5090, sm_120; the CUDA build compiles for sm_90): test_ionization identical to the CPU (the same event counts),
  ionization.txt of examples/ionization_2d.deck and rz_ionization.deck identical to the CPU's.
- Cost: ionization 7 % of the step (2D example), 16 % (RZ example).
- Smilei 5.1 (bench/ionization_smilei/README.md): C2+ / C3+ 0.98981 / 0.01019 vs 0.98969 / 0.01031 (0.3 sigma), field at
  the end 0.96226 vs 0.96342 of the pulse energy; without the ionization current 0.99302.

## Results, boosted frame (2026-11-09, the CPU cluster CPU; decision 90)
- Rest-frame field vs the Lorentz transformation (1000 random E, B in [-1, 1], u in [-3, 3] per component): 1.8e-15;
  a lab plane wave seen from the boosted frame by a lab ion: |E_rest| = E_lab to 2.2e-16.
- Nitrogen slab, lab vs boosted (gamma 2, iRIP, lambda/32; sigma from N_eff of both runs, ~3e4 / 6e4 ions each):
  2D a0 = 0.0105: N0 / N1+ 0.4762 / 0.5238 lab, 0.4756 / 0.5244 boosted (0.15 sigma);
  RZ a0 = 0.0105: 0.5422 / 0.4578 lab, 0.5448 / 0.4552 boosted (0.91 sigma);
  2D a0 = 0.03: N2+ / N3+ 0.9839 / 0.0161 lab, 0.9834 / 0.0166 boosted (0.40 sigma); RZ a0 = 0.03 (scan): 0.68 sigma.
  Gauss law <= 3.4e-17, E_cur / E_ion - 1 <= 2.2e-16; 748 boosted steps against 3088 lab steps.
- Energy per ionization, after the correction (2026-11-28, the CPU cluster epyc, Slurm job 18333, commit ba719a7): ion_energy_factor
  vs the photon Doppler factor gamma (1 - V . k), 1000 random plane waves and frames: 3.2e-15; gamma 2: forward 0.267949,
  backward 3.732051 (exact). B4 runs: E_ion / sum I_k w = 1 to 1e-12 in the lab, 1.00009 (2D), 1.00008 (RZ), 1.00038
  (2D, a0 = 0.03) times gamma (1 - beta) = 0.26795 in the boosted runs (before: gamma = 2, 7.46 times too much). Charge
  states lab vs boosted after the correction: 0.3 sigma (2D), 0.7 sigma (RZ), 0.4 sigma (2D, a0 = 0.03).
- Lab runs bit-identical to main: examples/ionization_2d.deck and rz_ionization.deck with 1 thread: diag.txt and
  ionization.txt byte-identical; with 32 threads ionization.txt identical and diag.txt identical except the max|dGauss|
  column, which differs at 1e-14 between two runs of main itself (the order of the threads' atomic adds).
- MPI (tests/mpi.sh, 1 vs 8 ranks): 2D boosted ionization.txt identical (17163 events), diag.txt identical; RZ (8 ranks along
  z) ionization.txt to 1.1e-13 (39236 events in both), energies 3.5e-13; Gauss drift 1.2e-17 / 2.8e-18.
- CUDA build compiles (sm_120, sm_90); RTX 5090 (cuda, sm_120): B4 passes with the same fractions and event counts as on
  the CPU.

## References
M. V. Ammosov, N. B. Delone, V. P. Krainov, Sov. Phys. JETP 64, 1191 (1986); A. M. Perelomov, V. S. Popov, M. V. Terent'ev,
Sov. Phys. JETP 23, 924 (1966); R. Nuter, L. Gremillet, E. Lefebvre, A. Levy, T. Ceccotti, P. Martin, Phys. Plasmas 18,
033107 (2011); P. Mulser, F. Cornolti, D. Bauer, Phys. Plasmas 5, 4466 (1998); X. M. Tong, C. D. Lin, J. Phys. B 38, 2593
(2005); M. Chen et al., J. Comput. Phys. 236, 220 (2013); J. Derouillat et al. (Smilei), Comput. Phys. Commun. 222, 351 (2018).
