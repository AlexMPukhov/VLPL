# Binary collisions and impact ionization (Phase 3)

Status: implemented (branch collisions, decision 94); physics choices provisional, for Alexander's review.

## Physics and equations
Coulomb collisions between macro-particles of the same cell: the relativistic binary Monte Carlo of Pérez et al., Phys.
Plasmas 19, 083104 (2012), the method of Smilei, with Nanbu's distribution of the cumulative small-angle deflection (Nanbu,
Phys. Rev. E 55, 4642 (1997)). For a pair (masses m1, m2, charges q1 q2 e, u = p / m c) the momenta are transformed to the
centre-of-mass frame (gamma_C, p* = p1* = -p2*, energies E1*, E2*), p* is rotated by chi (from Nanbu's distribution with the
parameter s) and a uniform phi, and transformed back; E1*, E2* are kept, so momentum and energy of the pair are conserved to
round-off. In code units (n in n_c, dt in 1/omega0, k0 r_e = 2 pi r_e / lambda0):

    s = D k0r_e lnL (q1 q2)^2 gamma_C p* / (m1 gamma1 m2 gamma2 (m1 gamma1 + m2 gamma2)) (E1* E2* / p*^2 + 1)^2
    s <= s' = D (3 k0r_e)^(-1/3) (m1 + m2) v_rel / max(m1 n1^(2/3), m2 n2^(2/3)),   v_rel = |v1* - v2*|   (low-temperature cap)

(Pérez eqs. for s and s'; for m2 -> infinity, s = n dt k0r_e Z^2 lnL / (gamma^2 beta^3), the Rutherford rate of <theta^2>/2.)
D is the density of the collision partners times the time: Smilei 5.1's normalization D = every dt N_c max(w1, w2) / V with
N_c = 2 N_pairs - 1 (intra) or N_pairs (inter) and the correction 1/k or 1/(k+1) for the partners that repeat (below).

Nanbu's chi: cos chi = ln(e^-A + 2 U sinh A) / A with coth A - 1/A = e^-s, so <cos chi> = e^-s; A from Pérez's fits (s < 0.1:
cos chi = 1 + s ln U; 1/A a quintic for s < 3; A = 3 e^-s for s < 6; isotropic beyond). Option `angle = smilei5`: the fit of
Smilei 5.1, sin^2(chi/2) = a U / sqrt(1 - U + a^2 U), a = 0.37 s - 0.005 s^2 - 0.0064 s^3 (s < 4, isotropic beyond).

Coulomb logarithm: fixed per block (`coulomb_log > 0`) or automatic (`coulomb_log = 0`, Pérez; per pair the Lee-More form):
lnL = max(2, 0.5 ln(1 + lambda_D^2 / b_min^2)), b_min = max(hbar / (2 p*), b0) with b0 = |q1 q2| r_e gamma_C / (m1 gamma1 + m2
gamma2) (E1* E2* / p*^2 + 1) (the classical distance of closest approach), and per cell 1 / lambda_D^2 = sum over all species of
n <q>^2 / T, T = m <u^2 / gamma> / 3, at least the ion-sphere radius a = (4 pi n_max / 3)^(-1/3) (as Smilei 5.1). The quantum
b_min is the diffraction limit hbar / (2 p*) (Lee and More 1984, GMS, BPS; ln(lambda_D / (hbar / 2 m v)) at v^2 = 3 T / m is
the NRL formulary's quantum e-i log to 0.1); Smilei 5.1 uses h / (2 p*) (Collisions.cpp: coeff1 = h omega / (2 m_e c^2)), 2 pi
larger, which lowers lnL by ln 2 pi = 1.84 wherever the quantum limit sets b_min (T_e >~ 10 eV); `quantum_bmin = h` keeps it
for comparisons with Smilei (decision 94, "Dense plasmas" below).

Unequal weights (Nanbu and Yonemura, J. Comput. Phys. 145, 639 (1998), as Smilei): particle 1 is deflected with the
probability w2 / max(w1, w2), particle 2 with w1 / max(w1, w2) (one uniform for both). Momentum and energy are then conserved
on average only.

Impact ionization (`ionizing = true`; Pérez section III, Smilei's CollisionalIonization): for an electron-ion pair, in the
ion frame (gamma_s = gamma_e gamma_i - u_e . u_i), the cross-section sum of the relativistic binary-encounter Bethe model over
all orbitals of the ion of charge Z* (occupation N, binding B = U; Kim, Santos, Parente, Phys. Rev. A 62, 052710 (2000)), the
mean energy of the secondary electron and the mean energy lost by the incident one, tabulated per Z* on 100 energies 1 eV ..
10 MeV (log). Orbitals: the binding energies of the neutral atom (data/binding_energies.def, Smilei's table) shifted so that
the outermost one of the ion is its NIST ionization energy I(Z*) (data/ionization_energies.def): B_k(Z*) = I(Z*) - B_{Z-Z*-1}
+ B_k. Probability per step 1 - exp(-D sqrt(gamma_s^2 - 1) sigma / (gamma_i gamma_e)); several ionizations in one step by the
cumulative probabilities of Nuter et al. (2011). The incident electron loses the lost energy (along its direction in the ion
frame), the new electron (weight w_i, at the ion's position) gets the secondary energy along the same direction; weight rule:
the ion is ionized with the probability w_e / w_i, the electron slowed with w_i / w_e. The ion's recoil is neglected (as
Smilei): momentum and energy are conserved on average only; the charge exactly (ion +w_i, electron -w_i at the same point).

## Discretisation
Stage Migrate, after the migration and the tile sort (all particles in the cells of their rank; the collisions change u^{n+1/2}
at x^{n+1}), every `every` steps with dt_eff = every dt. Cells: Cart3D / 2D the grid cells (absolute index floor(x / h), V = hx
hy hz, 2D: hz = the slab thickness dz); RZ the annuli between the radial nodes, V = pi (r_{j+1}^2 - r_j^2) dz, all azimuthal
angles together (as Smilei's AM geometry). Particles in the entry columns of a moving window (x >= xfreeze) do not collide.

Per species and step: cell lists (key = cell, counting sort on the CPU, CUB radix sort on the GPU), and within a cell the order
of a hash of (seed, step, species, particle id) (heap sort per cell): a random shuffle that does not depend on the storage
order. Pairing per cell (Smilei 5.1): intra-species pairs (L[2i], L[2i+1]), an odd number pairs L[0] twice (with the weight
correction 1/2 on its two pairs); inter-species every particle of the larger group with one of the smaller, L_small[i mod N_small]
(those repeat k or k + 1 times: corrections 1/k, 1/(k+1)). One thread per cell does its pairs in sequence (a particle that
collides twice sees its updated momentum). Random numbers of a pair: hashes of (seed, step, block, absolute cell, pair index).
So the collisions are the same for any number of ranks or threads, on CPU and GPU, and any particle order (test C7); 1 vs NP
ranks differ only by the round-off of the PIC fields. The ids are a hash of the load position (as field ionization; registered
on the colliding species when there are collision blocks); new electrons of field and impact ionization get theirs from the ion's
id and the level.

Impact ionization creates particles inside the per-cell kernel: it runs twice with the same random numbers, first counting the
new electrons per cell on a saved copy of the momenta and charge states (restored afterwards), then, after a scan of the counts,
writing them; they are appended to the electron species after the kernel (count, scan, fill: reproducible).

`mobile = false` species in a collision block are frozen (Smilei's time_frozen): never pushed, no current, momenta from the
deck (u, uth allowed) and changed only by the collisions; they do not count in the omega_p dt limits. The benchmarks use it.

Species without collision blocks: nothing changes (no id attribute, no operator; bit-identical, checked on 8 example decks).

## Deck interface
```
collisions <name> {                  # any number of blocks; the name only labels collisions.txt
  species1    = electrons
  species2    = ions                 # default: species1 (collisions within one species)
  coulomb_log = 0                    # 0: automatic (Perez), > 0: fixed
  every       = 1                    # collide every `every` steps with every dt
  angle       = nanbu                # nanbu | smilei5
  low_t_cap   = true                 # Perez's low-temperature limit s <= s' (false: none; see "Dense plasmas")
  quantum_bmin = hbar                # automatic lnL: quantum b_min hbar / (2 p*) | h (h / (2 p*), as Smilei 5.1)
  ionizing    = false                # impact ionization (an electron species: charge -1, mass 1; an ion species with element
                                     #   and charge_state, which then carries a charge state per particle)
  ionization_electrons = electrons   # species of the new electrons (default: the electron species of the block)
}
```
Output `dir/collisions.txt` at the cadence of diag.txt: per block the pairs collided, the mean s and lnL, the kinetic energy
changed by the collisions (0 to round-off with equal weights); ionizing blocks also the events, sum w_i I and <Z>; per species
of the blocks <u_x>, <u_y>, <u_z>, T_a = m <(u_a - <u_a>)^2 / gamma> and <E_kin> (m_e c^2).

## Tests
- tests/test_collisions.cpp (C1-C9, I0-I2), numbers in "Results".
- bench/collisions_epoch/: solid-density e-p relaxation against EPOCH 4.20.1, Smilei 5.1, NRL and the dense-plasma Coulomb
  logarithms (dense_plasma.py: Lee-More, GMS, BPS; the statistics of the cap).
- tests/mpi.sh: examples/collisions_2d.deck (dense Al plasma, e-e, e-i with impact ionization, i-i; NP ranks along x) and
  examples/rz_collisions.deck (hydrogen, RZ, ranks along z), mobile species: diag.txt and collisions.txt.
- bench/collisions_smilei/: against Smilei 5.1 on its own collision benchmarks (thermalisation, isotropization, beam relaxation,
  stopping power, impact-ionization rate), frozen species.

## Results (2026-11-10, the CPU cluster CPU)
- C1 one collision, 20000 random pairs (e-e, e-p, e-He, |u| 1e-3 .. 1e2, s 1e-6 .. isotropic): |dP| / (|P| + E) <= 2.2e-14,
  |dE| / E <= 6.6e-14 (round-off; the largest at |u| = 100 with m2 / m1 = 7300).
- C2 <cos chi> = e^-s: Pérez's fits 1.2e-3 (s = 0.01 .. 4); the Smilei 5.1 fit 2.1e-2.
- C3 isotropization (T_perp = 2 T_par, 1 keV, lnL = 10) vs NRL at 1.3 e-folds: -5.0 % (dt = 100) and -6.0 % (dt = 25); the NRL
  formula assumes bi-Maxwellians, which the strongly anisotropic plasma leaves (the deficit does not depend on dt). 2D -5.6 %, RZ
  -7.1 %, RZ inner / outer radial half -8.6 % / -7.1 % (cell volumes 2 pi r dr dz): the same within the noise of 1-2 %.
- C4 e-i relaxation (m_i = 4, T_e = 5 T_i, intra lnL = 1000 to keep Maxwellians, as Smilei's benchmark) vs NRL (Spitzer) over
  6 e-folds: max deviation 0.098 (dt = 100, s ~ 0.1-0.4 per step) and 0.051 (dt = 25): the time-step error (a pair exchanges
  1 - e^-s instead of s); w_i = 4 w_e: 0.048, energy changed 2.8 % of the exchanged energy (Nanbu-Yonemura); equal weights 7e-11.
- C5 a gamma = 3 electron beam on cold electrons (beam weight 1e-4): <gamma> vs d gamma / dt = -n k0r_e lnL / beta to 9e-4 of the
  energy lost (gamma 3 -> 2.03); the background heats by 1e-4. A gamma = 2 beam on heavy ions: <cos theta> vs exp(-nu t), nu =
  n k0r_e lnL / (gamma^2 beta^3), 2.7e-3 at nu t = 2 (32768 electrons).
- C6 Maxwell-Jüttner at theta = 1: KS distance of the gamma distribution 0.0026 after sum s = 1.6 per particle (95 % level
  0.0058); a Gaussian in u relaxes to the Maxwell-Jüttner distribution of its energy (theta 0.557): KS 0.0695 -> 0.0038.
- C7 the momenta per id are bit-identical with the particles stored in reverse order (automatic lnL); three steps of e-e and
  e-i with equal weights conserve P to 1.5e-15 and E exactly.
- I0 the RBEB cross-section of H at 100 eV: 6.04e-17 cm^2 (measured 5.7e-17, Shah et al. 1987).
- I1 C0 in a 1 keV electron beam (u_x = 0.2, n_e = 10 n_C): <Z>(t) vs the rate equations with R_k = n_e <sigma_k v> over the
  electrons (the formula, not the tables): <Z> 4.770 vs 4.762 (0.3 %) after 703361 events; the charge conserved to 3e-11.
- C9 the automatic lnL of an electron beam on cold protons (b_max = a_i) vs the Lee-More form evaluated in SI: |difference|
  <= 4.5e-4 (u = 0.01: classical b_min, 2.908; u = 0.03: quantum, 4.383; with quantum_bmin = h: 2.000 (the floor), 2.548).
- Solid density (bench/collisions_epoch; "Dense plasmas" below): with lnL = 5 fixed the 1/e time of T_e - T_i is 488 /omega
  (EPOCH 4.20.1 Nanbu-Pérez 487, full PIC, 488 at dt / 4; Smilei 513; NRL 409); without the low-temperature limit 455, 425 at
  dt / 4. With the automatic lnL 937 at dt / 4 (GMS 865, BPS 944).
- MPI, GPU, Smilei, cost: see STATUS.md / bench/collisions_smilei/README.md.

## Dense plasmas: the low-temperature limit and the Coulomb log (decision 94)
The benchmark case of bench/collisions_epoch, hydrogen at n_e = n_i = 1.1e23 cm^-3, T_e = 100 eV, T_i = 10 eV
(dense_plasma.py):
- lambda_De 0.223 nm, lambda_Di 0.070 nm, lambda_D (e + i) 0.067 nm, a_i 0.129 nm; N_D = 5.2 electrons per Debye sphere.
- Gamma_ee = Gamma_ei = 0.11, Gamma_ii = 1.12 (the protons moderately coupled).
- E_F = 8.5 eV, so T_e / T_F = 11.8: not degenerate. No degeneracy correction is needed here.
- Coulomb logarithms at t = 0: NRL formulary 2.07, Landau-Spitzer 2.74, Lee-More 2.79, GMS (model 6, as in Glosli et al.,
  PRE 78, 025401 (2008)) 2.56, BPS 2.34. The lnL = 5 of the benchmark is not physical here.

1/e time of T_e - T_i (1/omega; for the models lnL re-evaluated with T(t)):

| model | 1/e time |
|---|---|
| Spitzer/NRL rate with lnL = 5 | 409 |
| with the NRL-formulary lnL | 1078 |
| with Landau-Spitzer | 826 |
| with Lee-More | 756 |
| with GMS6 | 865 |
| with BPS | 944 |
| VLPL, automatic lnL, cap on (dt / 4) | 950 (937) |
| VLPL, automatic lnL, cap off (dt / 4) | 898 (885) |
| VLPL, automatic lnL with Smilei's h / 2p*, cap on (dt / 4) | 1137 (1136) |
| VLPL, the same, cap off (dt / 4) | 1098 (1056) |

How often the limit s <= s' acts (pairs sampled from the initial Maxwellians with the kernel's formulas; "exchange" =
sum of s p*^2, the momentum transfer the pairs carry):

| pairs | lnL = 5: pairs capped | lnL = 5: exchange removed | automatic lnL: pairs capped | automatic lnL: exchange removed |
|---|---|---|---|---|
| e-p | 6.7 % | 13 % | 3.5 % | 7.1 % |
| e-e | 4.1 % | 10 % | 2.1 % | 5.6 % |
| p-p | 62 % | 61 % | 41 % | 46 % |

For e-p the limit sets in below v_rel = 0.66 (lnL = 5) or 0.52 (automatic) of the rms relative speed.

Reading:
- The limit is a bound on the cross-section at small relative speed, sigma <~ a_i^2: the mean free path is not shorter than
  the interatomic distance. It is not a stability limit; Nanbu's sampling is stable for any s.
- With a fixed lnL = 5 it removes 13 % of the e-p exchange: the slow pairs, for which b0(v) approaches a_i and the true
  log would be ~1, not 5. With the automatic lnL (smaller for slow pairs) it removes half of that, 7 %.
- With the physical log the limit changes the relaxation time by 6 % (937 vs 885 at dt / 4). That is less than the 9 % spread
  between GMS and BPS, which molecular dynamics of dense hydrogen supports within its error bars (Glosli et al. 2008). With the
  limit VLPL is at BPS (-0.7 %) and 8 % above GMS; without it, 2 % above GMS and 6 % below BPS. Lee-More (756) and the NRL
  formula (1078) bracket the result; the NRL formula is not valid at lnL ~ 2.
- The larger error was the quantum b_min. Smilei's h / (2 p*) puts every e-p pair of this case on the floor lnL = 2 and makes
  the relaxation 20 % slower than BPS and 31 % slower than GMS. With hbar / (2 p*) the automatic lnL is the Lee-More form per
  pair, and its result lies between GMS and BPS.
- Decision (delegated by Alexander): keep `low_t_cap = true` as the default. It is a physical bound and not an artefact; at
  the physical lnL its effect is below the model spread; it stays the only limit where a Coulomb-log description fails
  (Gamma_ii ~ 1: 46 % of the p-p exchange); and it keeps VLPL comparable with Smilei and EPOCH, which both apply it (EPOCH:
  `s12 = MIN(s12, s_prime)`).
- Change: the automatic lnL now uses hbar / (2 p*); `quantum_bmin = h` restores Smilei's form. Runs with a fixed
  `coulomb_log`, i.e. all tests except C7 and C9 and all benchmarks, do not change.
- With lnL fixed at a value that is too large for the plasma, the cap acts as a partial correction. The physical way is the
  automatic lnL.

## Not yet / open
Species groups in one block (Smilei's lists); collisions in the boosted frame and with the implicit mover; the energy-conserving
weighting of Higginson et al. (J. Comput. Phys. 413, 109450 (2020)) for unequal weights; a degeneracy correction of the
Coulomb logarithm (T_e <~ T_F: the Fermi temperature in lambda_D and b_min, Lee-More), not needed for T_e / T_F = 12 above;
electron-only (dynamic) screening for e-i pairs (GMS: b_max^2 = lambda_De^2 + a_i^2; per pair it would give ~850 by the estimate of the sampled pairs, i.e. GMS); electron recoil-free ionization models other than RBEB (Lotz); three-body recombination;
the tallies across restarts.

## References
F. Pérez, L. Gremillet, A. Decoster, M. Drouin, E. Lefebvre, Phys. Plasmas 19, 083104 (2012); K. Nanbu, Phys. Rev. E 55, 4642
(1997); K. Nanbu, S. Yonemura, J. Comput. Phys. 145, 639 (1998); T. Takizuka, H. Abe, J. Comput. Phys. 25, 205 (1977);
Y.-K. Kim, J. P. Santos, F. Parente, Phys. Rev. A 62, 052710 (2000); R. Nuter et al., Phys. Plasmas 18, 033107 (2011);
J. Derouillat et al. (Smilei), Comput. Phys. Commun. 222, 351 (2018); J. D. Huba, NRL Plasma Formulary (2019).
