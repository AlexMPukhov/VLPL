# Spin-resolved strong-field QED (Phase 4, spin part)

Status: implemented (decision 93, merged; Breit-Wheeler pairs: decision 96, branch breit-wheeler). Physics choices of
decision 93 provisional, for Alexander's review; the five choices of decision 96 (mixed pair spins, selection on, energy
exact with the momentum to the field, threshold 2 m c^2 only, x from the full spectrum) accepted by Alexander, 2026-11-11;
the transverse pair spin follows the published sign since 2026-11-11 ("Comparison with the literature" below). Field solvers
RIP-CC and iRIP (explicit mover; "Field solvers and the time step" below); in the boosted frame since decision 117 (defaults
accepted by Alexander, 2026-11-23; "Boosted frame" below); not with the implicit mover.

## Physics and equations
Code units: c = 1, t in 1/omega0, fields in m c omega0 / e, u = gamma v, energies in m_e c^2; hw = hbar omega0 / (m c^2)
(3.03e-6 at 0.8 um).

**T-BMT** (every species with `spin`; anomaly a, 1.15965218e-3 for e-/e+):

    dS/dt = (q/m) S x Omega,   Omega = (a + 1/gamma) B - a gamma/(gamma + 1) (beta.B) beta - (a + 1/(gamma + 1)) beta x E

In a constant B perpendicular to v, S turns at (a + 1/gamma) B = (a gamma + 1) omega_c and relative to u at a gamma omega_c.

**Photon emission** (nonlinear inverse Compton scattering, locally constant field approximation), resolved in the lepton
spin S (|S| <= 1, mixed state) and the photon Stokes vector xi; u = eps_gamma / eps_e, chi = gamma hw sqrt((E + beta x B)^2 -
(E.beta)^2):

    d2W/(du dt) = C0/4 (w + <g, S> + <pf, Sf> + <pxi, xi>),   C0 = alpha m^2 c^4 / (sqrt3 pi hbar eps_e) = alpha / (sqrt3 pi hw gamma)
    w   = (u^2 - 2u + 2)/(1 - u) K23(y) - IK13(y),   y = 2u / (3 (1 - u) chi),   IK13(y) = int_y^inf K13(x) dx
    g   = -u K13 e2
    pf  = -u/(1-u) K13 e2 + (2 K23 - IK13) S + u^2/(1-u) (K23 - IK13) <S, ev> ev
    pxi = ( u/(1-u) K13 <S, e1>,  ((2u - u^2)/(1-u) K23 - u IK13) <S, ev>,  K23 - u/(1-u) K13 <S, e2> )

Basis: ev = u/|u|, e1 = the normalized transverse part of the Lorentz force q (E + beta x B), e2 = ev x e1 (along B for an
electron in a magnetic field). xi = (xi1, xi2, xi3) are Stokes parameters in the photon frame with the axes (e1, e2) (xi3 = +1:
linear polarization along e1, xi2: circular); e1 is stored with the photon. Summed over Sf and xi the rate is C0 (w + <g, S>).
After an emission (mixed state): S <- pf / p0, xi = pxi / p0, p0 = w + <g, S>; pure state: S <- +-pf/|pf| with probability
(p0 +- |pf|) / (2 p0), likewise xi. Without an emission in dt (the selection effect):

    S <- [S (1 - C0 dt W) - C0 dt G] / (1 - C0 dt (W + <G, S>)),   W = int_0^1 w du,   G = int_0^1 g du = -Gs e2

For S along e2 the mean spin follows dS/dt = C0 [(Xi - W) S + K - G] (Xi = int (2 K23 - IK13), K = int -u/(1-u) K13), so
S(t) = (S0 - S_inf) exp(C0 (Xi - W) t) + S_inf with S_inf = -(K - G)/(Xi - W) -> -8/(5 sqrt3) = -0.924 for chi -> 0: the
Sokolov-Ternov polarization, antiparallel to B for electrons (along -e2). The mean energy loss is
P = gamma C0 int u w du = P_alpha chi^2 g(chi), P_alpha = 2 alpha^2 m c^2 / (3 tau_e) (= 2/3 alpha chi^2 / hw in code units).

Corrections to the manuscripts (QV3D draft, arXiv:2605.19814): S_f = pf / (w + <g, S>) without C_rad; e1 normalized; the
photon Stokes vector has the components (1,0,0), (0,1,0), (0,0,1) in Stokes space (not spatial e1, e2, ev); the
non-radiative update normalized by 1 - C0 dt (W + <G, S>) (arXiv Eq. 11; the QV3D Eq. 15 lacks it, Eq. 13 needs it); the
Sokolov-Ternov limit applies to the spin component along e2, not the longitudinal one; y = 2u/(3(1-u) chi) (not chi^3).

**Breit-Wheeler pairs** (decision 96; photon energy om = eps_gamma / m c^2 > 2 along n, chi_gamma = om hw |E_perp + n x B|,
x = eps_+ / eps_gamma, x_- = 1 - x, rho = 2 / (3 chi_gamma x x_-); basis n, e1 = the transverse part of E + n x B (the force on
the positron), e2 = n x e1; xi the photon Stokes vector in (e1, e2)):

    d2W/(dx dt) = Cg (B0 - xi3 K23(rho)),   B0 = (x^2 + x_-^2)/(x x_-) K23 + IK13,   Cg = alpha / (sqrt3 pi hw om)
    b_+ = -(K13/x - xi3 K13/x_-) e2 - xi1 K13/x_- e1 + xi2 ((x - x_-)/(x x_-) K23 + IK13/x) n        (positron spin)
    b_- = (K13/x_- - xi3 K13/x) e2 + xi1 K13/x e1 + xi2 ((x_- - x)/(x x_-) K23 + IK13/x_-) n         (electron spin)

Mixed state: S_+- = b_+- / p0, p0 = B0 - xi3 K23 (pure: +-b/|b|). As Seipt & King (2020) and Chen et al. (2022) (comparison
below). Since e2 = -B'/|B'| (B' = B - n x E, the magnetic field in the rest frame of a lepton along n), an unpolarized photon
gives the positron preferably parallel and the electron antiparallel to B' (the lower magnetic energy, the orientation of the
Sokolov-Ternov limit of the emission). First derived from the emission rate above by crossing and charge conjugation (the QV3D
draft has no pair formulas); that derivation had the transverse sign wrong (below), corrected 2026-11-11 (Alexander). Checks: |b_+-| <= p0 (max 0.99994 over x, chi, unit xi), the unpolarized rate and Erber's limits
(W = alpha m^2 c^4 / (hbar eps_gamma) T(chi): 3 sqrt3 / (16 sqrt2) chi e^(-8/(3 chi)) for chi << 1, 0.3796 chi^(2/3) for chi >> 1),
the rate ratio of photons polarized along e2 and e1, (T0 + T3) / (T0 - T3) -> 2 (chi -> 0) and 3/2 (chi -> inf), helicity transfer
to the harder lepton. Integrals T0 = int B0 dx, T3 = int K23 dx: total rate Cg (T0 - xi3 T3). Without a decay in dt (a = Cg dt)
the photon state is selected (dichroism): xi <- [xi (1 - a T0) + a T3 (0,0,1)] / (1 - a (T0 - xi3 T3)); unpolarized photons become
xi3 = tanh(Cg T3 t). Kinematics: eps_+ = x eps_gamma, eps_- = eps_gamma - eps_+ (energy exact, each >= m c^2), both along n, |p| =
sqrt(eps^2 - 1): the field takes m c (1/(2 eps_+) + 1/(2 eps_-)) of the momentum. The rest energy 2 m c^2 per pair leaves the
kinetic energies of diag.txt (W_kin).

**Comparison with the literature** (2026-11-11, from the arXiv sources: D. Seipt, B. King, PRA 102, 052805 (2020),
arXiv:2007.11837, their LCFA pair rate (rate.pp.final.stokes) and Compton rate (RNLCfinal); Y.-Y. Chen, K. Z. Hatsagortsyan,
C. H. Keitel, R. Shaisultanov, PRD 105, 116013 (2022), arXiv:2201.10863, their G_0..G_3 and the electron polarization (PPE)).
With Ai1(z) = IK13/(pi sqrt3), Ai(z)/sqrt z = K13/(pi sqrt3), Ai'(z)/z = -K23/(pi sqrt3), z = (chi_gamma x x_-)^(-2/3), and
Seipt-King's tau_k = xi3, s = x:
- Agree: the rate Cg (B0 - xi3 K23) (both papers, term by term); the magnitudes of all spin terms (e2: K13/x - xi3 K13/x_- for
  the positron, K13/x_- - xi3 K13/x for the electron; e1: xi1 K13/x_-, xi1 K13/x; denominator p0); the longitudinal (helicity)
  terms xi2 ((x - x_-)/(x x_-) K23 + IK13/x) n and xi2 ((x_- - x)/(x x_-) K23 + IK13/x_-) n with their sign (Chen's electron:
  zeta . v = xi2 (IK13/x_- - (x^2 - x_-^2)/(x x_-) K23) / p0).
- Differed (first version, 635e1a2 .. 67a5a96, merged in PR #124): the sign of the transverse part (the e1 and e2 terms) of b_+ and b_-; now
  corrected (b_perp -> -b_perp, both leptons; the formulas above are the corrected ones). Seipt-King's spin axis follows from their
  Compton rate, which carries an overall minus sign: matched to the verified emission rate of decision 93 (g = -u K13 e2,
  Sokolov-Ternov along -e2), their axis zeta sign(h') is +e2 of an electron in the same field, i.e. along the rest-frame
  magnetic field B'. Their pair rate (no overall minus) then favours the positron along +B' and the electron along -B' (their
  text: the pair is preferably produced in the lower magnetic energy state, positron parallel, electron antiparallel to B').
  Here e2 = n x e1 = -B'/|B'| (e1 along the force on the positron), so they give the positron along -e2 and the electron along
  +e2 (= -e2_- in its own basis), the orientation of the Sokolov-Ternov limit of the emission; the first version gave the
  opposite. Chen et al. agree with Seipt-King when their s (the transverse acceleration of the created electron) is -e1:
  electron zeta = -[xi1 K13/x s + (K13/x_- - xi3 K13/x) b - xi2 (...) v] / p0 with b = v x s = -e2; their xi1 term then also
  had the opposite sign to the first version.
- Where the crossing lost the sign: it mapped the spin of the incoming lepton onto the outgoing positron as S_perp -> +S_+perp
  (only the longitudinal component got a sign, from ev -> -n). An outgoing antifermion carries the reversed spin: its
  two-component spinor describes -zeta_+ (Chen et al., below their Eq. for R_p: zeta' = -zeta_+, the hole picture). So
  S_perp -> -S_+perp: every term of b_+ along e1 and e2 changes sign, and with it (charge conjugation) those of b_-. The longitudinal xi2 n terms came out with
  the published sign (helicity transfer to the harder lepton) and were confirmed against Chen et al., not re-derived. Effect of
  the correction: |b|, the rates, the photon selection, the yields and the longitudinal terms unchanged; <S+.e2>, <S-.e2-> and
  the pair-spin spectra of qed.txt / qed_pairs.txt change sign; the pairs' later emission changes slightly through <g, S>.
  Test: tests/test_pairs.cpp P4 (per x bin, the published sign) and P4b (positron parallel, electron antiparallel to B'
  computed from E and B, in a magnetic and in a crossed field).
- Not compared: the pair spin correlations (Seipt-King's sigma_p sigma_q terms, Chen's zeta_- . zeta_+ terms; not kept here).

**Independent first-principles check of the sign** (2026-11-11, Alexander: "check the signs carefully! Seipt & Chen also might make
a mistake"; tools/volkov_spin.py, numpy/scipy, no formula from a paper). Volkov states and explicit Dirac spinors in a plane wave
a(phi) = a0 (1 + tanh(phi/tau)) along -z (a0 = 10, tau = 5: a' > 0 everywhere, so B' = gamma (B - v x E) is along +y for every
lepton moving along +z, computed from E and B), q = -1 for the Dirac field, metric (+,-,-,-), Dirac representation, gamma5 =
i g0 g1 g2 g3; spins along +-y in the rest frame reached by a pure boost; the positron by hole theory (the negative-energy solution
with spinor spin -n has physical spin n). Building blocks: both Volkov solutions satisfy the Dirac equation to 1e-12 (relative);
the hole-theory positron spinor equals the charge conjugate i g2 u* (|overlap| = 1.000000); v vbar = (pslash - m)(1 + g5 sslash)/2
with s = the PHYSICAL positron spin (the projector of Seipt-King, so their positron label is physical). The amplitudes are the
phi integrals of ubar Gamma(a) v e^{i Phi} at fixed light-front momenta (created / emitted at phi = 0, chi = 1), integrated over
the transverse momentum perpendicular to the field.
- Emission (the same code; the C-conjugate field for the positron): all four spin channels W(i -> f) along B' agree with
  VLPL's verified emission rate (decision 93) to 1e-3 at u = 0.2, 0.5, 0.8; (W_down->up - W_up->down)/sum along +B' =
  -0.6811 / -0.8500 / -0.9478 (VLPL -0.6815 / -0.8501 / -0.9477) for the electron, the opposite for the positron: electrons
  polarize antiparallel, positrons parallel to B' (Sokolov-Ternov).
- Pairs (unpolarized photon, chi = 1): <S_+ . B'^> = +0.9406 / +0.6700 / +0.2351, <S_- . B'^> = -0.2351 / -0.6700 / -0.9406
  at x = 0.2 / 0.5 / 0.8; the corrected b_+- / p0 give +-0.9405 / 0.6699 / 0.2351. Photon along the field force (xi3 = +1):
  no transverse spin (formula: 0); rate ratio of the two linear polarizations 2.095 (formula 2.093); circular photon of helicity
  +1: helicity +0.9453 of the harder, +0.1242 of the softer lepton (x = 0.8 / 0.2), as the xi2 n terms.

| source | method; conventions | positron transverse spin | electron transverse spin |
|---|---|---|---|
| this check (tools/volkov_spin.py) | Volkov + Dirac spinors; q = -1, metric (+,-,-,-), physical spins (hole theory = C), B' = gamma (B - v x E) of the lepton | parallel to B' | antiparallel to B' |
| Seipt & King, PRA 102, 052805 (2020) | Volkov / Furry; a = eA with e < 0; spin 4-vector zeta_p = beta - (p.beta / p.kappa) kappa (rest-frame B direction), projectors (pslash -+ m)(1 + sigma g5 zeta-slash)/2 (physical spin, checked above); photon basis Lambda_1 = laser E, Lambda_2 = laser B; sign(h') fixed by their Compton rate against VLPL's | parallel ("lower magnetic energy") | antiparallel |
| Chen, Hatsagortsyan, Keitel, Shaisultanov, PRD 105, 116013 (2022) | Baier-Katkov operator method (independent of Seipt-King); antiparticle spinor phi(zeta' = -zeta_+) (hole sign explicit); basis v, s = transverse acceleration of the created electron, b = v x s; photon e1 = (n x H + E_perp) normalized | parallel (with s = -e1) | antiparallel |
| Torgrimsson, NJP 23, 065001 (2021) | light-front Volkov; e a -> a with e < 0, so his B^ is antiparallel to B (his footnote); Stokes n = u^+ Sigma u / 2 "and similarly for positrons"; R_2 = +(q1/s2) Ai/sqrt(xi) B^, R_3 = +(q1/s3) Ai/sqrt(xi) B^; states agreement with Seipt-King | parallel if his positron vector is that of v (as charge conjugation requires opposite e+ and e- orientations) | antiparallel |
| Baier, Katkov, Strakhovenko (book, 1998) | not read here; Chen et al. state that their positron-summed result reproduces it | - | antiparallel (via Chen) |
| VLPL before the correction (635e1a2 .. 67a5a96) | crossing of decision 93, S_perp -> +S_+perp | antiparallel (wrong) | parallel (wrong) |
| VLPL now | b_+- above | parallel | antiparallel |

Charge conjugation (an e+ in B behaves as an e- in -B): the e+ and e- transverse spins are opposite in all rows; it fixes the
relative sign only, the direct calculation fixes the absolute one. The pair orientation is that of the Sokolov-Ternov polarization.

**Deterministic option** (`model = deterministic`): d gamma/dt = -P_alpha chi^2 g(chi) with g from the same tables, the
direction of u kept (the quantum-corrected Landau-Lifshitz drag of an ultrarelativistic particle), no spin effects.

## Discretisation
- Push and move split (decision 15) for the species with spin or emission: stage Push gathers E^n, B^n at x^n, pushes
  u^{n-1/2} -> u^{n+1/2} and stores (E^n, B^n, u^{n-1/2}) per particle (`push_store`, `rz_push_store`); stage Qed runs T-BMT
  (exact rotation by |omega| dt about omega = -(q/m) Omega with u^n = (u^{n-1/2} + u^{n+1/2})/2) and then the emission on
  u^{n+1/2}; stage MoveDeposit moves and deposits with the unchanged kernels (Pusher::None: no gather, no push). The other
  species keep the fused kernel; positions are untouched by the QED operator, so the Gauss law stays at round-off.
  Geometry-independent: E, B, u, S are Cartesian in the particle axes (RZ: (z, x, y)).
- Emission (Algorithm 1 of the QV3D draft): r uniform, u = r^3, emission if r' < 3 r^2 C0 (w + <g, S>) dt_s. Sub-steps
  dt_s = dt / n with n such that C0 M dt_s <= 1 (M = max_r 3 r^2 (w + u K13)) and C0 (W + Gs) dt_s <= `substep_prob` (0.1); E, B
  held, chi and the basis recomputed after each emission. Events with probability > 1 (outside M) are counted (column P>1).
- Lepton after an emission: gamma <- max((1 - u) gamma, 1) along ev (mass shell; the photon gets the energy lost exactly).
  Photon: u_gamma = (energy) ev at x^n, weight of the lepton; into the photon species if above `photon_min_energy`.
- Tables (host, built once, 0.3 s): ln K13, ln K23, ln IK13 on 4096 points in ln y (1e-14 .. 720; quadratic interpolation,
  beyond: zero; below: the end slope) from the integral representation K_nu(y) = int_0^inf exp(-y cosh t) cosh(nu t) dt; ln W,
  ln Gs, ln M, ln P on 801 points in ln chi (1e-6 .. 1e4), integrated in ln u and ln(1 - u) with the same interpolation. The
  (chi, u) dependence of the rate enters only through y and the u-prefactors, so the y tables suffice.
- Random numbers: hash of (seed, step, species, particle id, sub-step, draw) (splitmix64, as ionization): any thread count,
  rank count, particle order, CPU or GPU give the same events. id: a hash of the position when the qed operator first sees
  the particle.
- Photons (`kind = photon`): q = 0, never gathered or deposited, moved x += dt u/|u| (stage MoveDeposit), migrated and
  written like any species (attributes xi1 xi2 xi3, pe0 pe1 pe2 = e1); their energy in diag.txt is sum w |u|.
- Photons are created by count, scan, fill (the counting pass replays the same random numbers).
- Pairs (stage Qed, after the emission; all photons, also those of this step): E^n, B^n gathered at the photons (push_store
  with Pusher::None), count, scan, fill (qed_pair<Count>: x uniform, a decay if r' < Cg p0 dt_s; sub-steps so that Cg Mb dt_s <= 1
  and Cg (T0 + T3) dt_s <= substep_prob), then the decayed photons are removed (stable order). The pairs are appended to
  `pair_electrons` / `pair_positrons` at the photon's x^n with its weight (neutral: the Gauss law stays) and get ids from the
  photon's random-number stream; photon ids come from the emitting lepton's stream (pairs on) or a position hash. The pair
  species are split species: moved and deposited in stage MoveDeposit of the same step.
- Breit-Wheeler tables: ln T0, ln T3, ln Mb (1.05 max_x (B0 + K23)) plus 8/(3 chi) on 701 points in ln chi_gamma (0.01 .. 1e4),
  integrated in ln x over [x_lo, 1/2] (symmetric) with the y tables. Below chi_gamma = 0.01: no pairs (rate ~ e^-267).

## Boosted frame (decision 117; the eight defaults accepted by Alexander, 2026-11-23)
With a `boost` block the QED operator runs unchanged in the boosted frame: the code units stay those of the lab omega0 (hw
fixed), chi and chi_gamma are invariants, and the rate per boosted time C0 = c0f / gamma' gives W dt' = invariant x proper
time; the numbers of photons and pairs do not depend on the frame. Frame-dependent and therefore converted:
- photon_min_energy, the energies of qed.txt (E_rad, E_photons, E_pairs), dN/d eps of the photons and the lepton spectra of
  qed_pairs.txt are LAB energies: eps_lab = gamma_b eps' (1 + beta_b n'_x) for a photon along n' (x the boost axis; RZ z),
  gamma_b (eps' + beta_b p'_x) for a lepton. Counts, u and x are kept. The rows of qed.txt stay at the boosted steps; W, <S>,
  <chi>, chi_max, N_photons and E_photons_now are the box at the boosted time (chi_max is no lab maximum: a beam moving against
  the boost crosses the pulse peak between two rows). The lab state at a lab time: the lab particle snapshots.
- Spins: the deck `spin` is the rest-frame spin seen from the lab; the code keeps the rest-frame spin reached by the pure boost
  from the simulation frame. qed_spin_boost converts through the spin 4-vector a = (u.s, s + (u.s) u / (gamma + 1)), boosted,
  s' = a' - a'^0 u' / (gamma' + 1) (the Wigner rotation included; |s| kept): at setup with the species' mean lab momentum (exact
  for a cold beam; the identity along the boost axis), per particle for the lab snapshots.
- Photons: the Stokes vector is unchanged (the basis (e1, n x e1) goes over covariantly, the helicity is invariant); the axis e1
  is the polarization 4-vector (0, e1) boosted, the gauge fixed transverse: e1' = eps' - eps'^0 k' / |k'| (qed_pol_boost).
- Lab particle snapshots (lab_diagnostics.species, src/boost_diag.cpp): photons are massless (velocity u / |u|,
  u_x = gamma_b (u'_x + beta_b |u'|)); species with spin get the columns s0 s1 s2, photons xi1 xi2 xi3 pe0 pe1 pe2 after
  x y z ux uy uz w (named in the .txt); lab_beams.txt of a photon species uses gamma = sqrt(1 + u^2) ~ |u|.
- Pair threshold om > 2 in the simulation frame (where the pair is made); the sub-steps per boosted step as before.
- A beam moving against the boost is gamma_b^2 (1 + beta_b)^2 shorter in boosted cells (14 at gamma_b = 2, 98 at 5): ppc_x
  must grow by that factor to keep its macro-particles (the boosted lattice may miss a beam shorter than a cell entirely).
- The gamma_b = 5 deficit at lambda/20 (table below) is the laser, not QED: the pulse from the lab-fixed antenna reaches its lab
  amplitude only a few um downstream of the antenna (0.82 / 0.94 / 0.99 of the lab run's on the axis 1.9 / 2.7 / 3.6 um from
  it; gamma 2: 0.999); the collisions of the QED decks are at 2.7 um, so the beam sees chi 7 % low. Halving c dt' removes it,
  halving h_perp does not (decision 117). Keep the interaction beyond the measured
  d_1% = 1.1 (gamma_b (1 + beta_b))^2 h_lab downstream of the antenna (printed by the boost summary; QED
  decks, h_lab = 0.04 um: 4.3 um at gamma_b 5, 0.61 um at 2) or refine h_lab (boosted_frame.md, near field).
- Limits of the accepted defaults: (a) the deck spin is converted at load with the species' MEAN lab momentum (exact for a
  cold beam; the per-particle Wigner rotation of a momentum spread is neglected at load, of the order of the spread angle; the
  lab snapshots convert every particle exactly); (b) the qed.txt rows are at boosted times: counts and lab energies are sums since
  the start, the instantaneous columns are the box at a boosted time, no lab quantities (chi_max 0.20 boosted vs 1.17 lab at
  gamma_b = 5 in qed_collision); lab-time states from the lab snapshots.

Lab run (gamma 1) vs boosted, iRIP and the lab-fixed antenna in all runs, 4 seeds each (mean +- standard error; the CPU cluster,
2026-11-23), examples/qed_cascade.deck (10 GeV, a0 = 80):

| lab cell | gamma_b | pairs / beam e- | photons / e- | E_rad / E_0 (lab) | <S+.e2> | lab spin <S2> of the e- at 29.35 fs |
|---|---|---|---|---|---|---|
| lambda/20 | 1 | 0.3144 +- 0.0013 | 9.416 | 0.7245 | -0.375 | 0.5950 (qed.txt of the lab run) |
| lambda/20 | 2 | 0.3134 +- 0.0021 | 9.364 | 0.7252 | -0.371 | 0.5941 (lab snapshot) |
| lambda/20 | 5 | 0.2532 +- 0.0015 | 8.418 | 0.6896 | -0.377 | 0.6210 |
| lambda/40 | 1 / 2 / 5 | 0.3166 / 0.3204 / 0.3114 | 9.509 / 9.516 / 9.321 | 0.7294 / 0.7296 / 0.7246 | -0.375 / -0.380 / -0.377 | 0.593 / 0.591 / 0.590 |
| lambda/80 | 1 / 5 | 0.3198 / 0.3150 (+- 0.0012 / 0.0010) | 9.538 / 9.529 | 0.7284 / 0.7291 | -0.381 / -0.377 | 0.591 / 0.591 |

examples/qed_collision.deck (1 GeV, a0 = 100), photons per electron: 14.83 / 14.87 / 12.98 (lambda/20, gamma 1 / 2 / 5),
14.89 / 14.95 / 14.53 (lambda/40), 15.03 / - / 14.87 (lambda/80); the lab spin <S2> at 29.35 fs 0.786 / 0.778 / 0.810,
0.783 / 0.780 / 0.782, 0.777 / - / 0.783. Spectra (dN/du, dN/d eps_lab, pair dN/dx): chi^2 per bin 0.8 - 3.5 against the lab
run at gamma 2 (lambda/20), 0.7 - 3.9 at gamma 5 (lambda/80). Energy at the lab time from the lab snapshot (kinetic + photons
+ 2 m c^2 per pair, relative to the beam's): -2.056e-5 vs -2.067e-5 in the lab run (cascade, gamma 2; the work of the laser).
P>1 = 0 in all runs. gamma 2 agrees with the lab run on its grid within ~2.5 sigma; gamma 5 approaches it with the lab cell;
the lambda/20 deficit is the laser near the antenna (above), not QED.

## Deck interface
```
species e  { ... spin = 0 0 1          # initial S (particle axes; |S| <= 1); spin_anomaly = 1.15965218e-3 }
species ph { kind = photon }            # empty at the start
qed {
  species = e                           # emitting leptons (|q| = 1, m = 1)
  model = monte_carlo                   # monte_carlo | deterministic | none (none: only T-BMT)
  photons = ph                          # optional photon species; without: tallies and spectra only
  photon_min_energy = 0                 # MeV, photons below are only tallied
  spin_state = mixed                    # mixed (|S| <= 1, default) | pure (|S| = 1, stochastic flips)
  spin_precession = true                # T-BMT on/off (also for species without emission)
  spin_flip = true                      # radiative spin change (after and without emission) on/off
  substep_prob = 0.1
  spectrum_bins = 200  spectrum_u_min = 1e-8  spectrum_e_min = 1e-3  spectrum_e_max = 1e7   # MeV
  spectrum_every = 0                    # steps; 0: dir/qed_spectrum.txt at the end only
  pair_electrons = pe                   # Breit-Wheeler pairs from `photons` (decision 96): electrons (q = -1, m = 1) ...
  pair_positrons = pp                   # ... and positrons (q = 1, m = 1), both mobile; both or none; model = monte_carlo
  pair_selection = true                 # the photon Stokes vector changes without a decay (dichroism)
}
```
Outputs: dir/qed.txt at the cadence of diag.txt (per species: weight, <S>, <|S|>, <chi>, chi_max, emitted number and energy,
mean Stokes vector of the emitted photons, P>1 events, energy into the photon species; then the photon species' count and
energy), dir/qed_spectrum.txt (dN/du and dN/d eps_gamma with <xi> per bin, since the start). With pairs, qed.txt adds N_pairs
(sum w), E_pairs (sum w eps_gamma of the decayed photons), P>1 events, the mean pair spins in their own bases (<S_+.e2>,
<S_+.n>, <S_-.e2_->, <S_-.n>) and the mean xi3 of the decayed photons; dir/qed_pairs.txt: dN/dx (x = eps_+ / eps_gamma) with
<xi3> and <S_+> per bin, dN/d eps of the electrons and positrons. Energy balance: sum W_kin + E_photons + 2 m c^2 N_pairs.
The pair species need `density = 0` (empty at the start; spin = 0 0 0 if their spins are wanted). Tallies restart from zero after
a restart (spins, photons and ids are in the checkpoint).

## Tests (tests/test_qed.cpp; references independent of the tables: std::cyl_bessel_k, own quadrature)
- Q1 tables vs std::cyl_bessel_k; W(chi) and g(chi) vs quadrature and the Baier-Katkov form (K_5/3); S_inf(chi = 1e-3) =
  -8/(5 sqrt3).
- Q2 dN/du = C0 w at chi = 0.1, 1, 10 (chi^2 per bin); Q3 rate(+e2) - rate(-e2) = 2 C0 g; Q4 <xi> per u bin vs Eq. 21 and
  the xi1, xi2 terms, mixed and pure.
- Q5 spin relaxation along e2 vs the analytic solution at chi = 1, 0.1 (mixed and pure); Q6 energy loss vs P_alpha chi^2 g.
- Q7 T-BMT: rotation vs RK4 of the T-BMT equation for random E, B, u; precession in a constant B in Cart3D and RZ:
  (a gamma + 1) omega_c and a gamma omega_c relative to u (exact up to round-off and the known Boris angle).
- Q8 energy balance electrons + photons (species and tallies); Q9 identical results with the particles in reverse order;
  Q10 deck checks. tests/mpi.sh: examples/qed_collision.deck on 1 vs 8 ranks.
- Q11 (decision 117) the frame changes of the boosted frame: the Wigner rotation of two perpendicular boosts, cos theta =
  (g1 + g2) / (1 + g1 g2) (4e-15); the spin along the boost axis unchanged; there and back (6e-13), |s| kept; the polarization
  axis of a photon along y polarized along x, (1/gamma, beta, 0); transverse and unit after random boosts, there and back.
  tests/mpi.sh "QED in the boosted frame": the small-box cascade at gamma 2 (iRIP, antenna) on 1 vs 8 ranks (energies 3.8e-13,
  qed.txt 1.7e-11, the lab snapshot of the fields and of the four species with spin, Stokes and polarization columns: column
  sums 7.3e-14); restarted from a checkpoint on 8 ranks (11 of 11 lab files identical) and on 3 ranks (9.8e-14).

## Tests of the pairs (tests/test_pairs.cpp; Bessel functions by qed_bessel_exact, quadrature here, Erber in closed form)
- P1 T0, T3 vs quadrature (chi_gamma 0.05 .. 1000); Erber's limits with the expected corrections (~ chi, ~ chi^(-2/3)); the
  polarization ratio -> 2 and 3/2; |b_+-| <= p0 for all x, chi, unit xi.
- P2 dN/dx = Cg B0 at chi = 0.3, 1, 10 (chi^2 per bin), energy exact per event, momentum deficit = m c (1/(2 eps_+) + 1/(2 eps_-)).
- P3 decay rates of xi3 = +1 / -1 photons; survival and the selection xi3 = tanh(Cg T3 t) of unpolarized photons (mixed, pure).
- P4 pair spins per x bin vs b_+- / p0 (unpolarized, linear along e1, e2, 45 deg, circular; mixed and pure).
- P5 cascade in a constant B (chi_e = 20, Cart3D and RZ): energy balance with the rest energy, charge, pairs = positrons;
  P6 identical pairs with the beam in reverse order; P7 deck checks. tests/mpi.sh: examples/qed_cascade.deck on 1 vs 8 ranks.
- P5b the cascade of P5 with `numerics.solver = irip` (Cart3D and RZ): at the step of P5 the RIP-CC events (pairs, emitted and
  decayed photon energy to 4.4e-16); at 4 times the step with h_perp = c dt / 4 (beyond the RIP-CC limit) the same pairs per
  beam electron, emitted and decayed photon energy as RIP-CC within 4 standard errors of 4 seeds each (Cart3D 10.09 vs 10.22,
  -2.3 sigma; RZ 10.18 vs 10.18), energy balance 8e-15, no event with probability > 1. tests/mpi.sh: the pair cascade with iRIP
  at 4 times the RIP-CC step on 1 vs 8 ranks (ranks along x).

## Field solvers and the time step (iRIP, 2026-11-23)
The QED operator acts on the particles only (E^n, B^n gathered in stage Push), so it works with RIP-CC and with iRIP
(`numerics.solver = irip`, explicit mover; the implicit mover replaces the push / move split and is rejected with spin and
QED, decision 93). With iRIP the transverse cells may be
smaller than c dt = h_x. Comparison on the CPU cluster (CPU, 4 seeds per case, the same physical time; mean +- standard error):

| examples/qed_cascade.deck (10 GeV, a0 = 80) | pairs / beam e- | E_rad / E_0 | <S+.e2> | P>1 |
|---|---|---|---|---|
| RIP-CC, h_x = lambda/20 (the deck) | 0.3144 +- 0.0012 | 0.7244 +- 0.0011 | -0.374 +- 0.003 | 0 |
| iRIP, h_x = lambda/20 | 0.3144 +- 0.0013 | 0.7245 +- 0.0010 | -0.375 +- 0.003 | 0 |
| iRIP, h_x = lambda/20, h_perp = h_x (RIP-CC rejects it) | 0.3147 +- 0.0005 | 0.7267 +- 0.0002 | -0.378 +- 0.001 | 0 |
| RIP-CC / iRIP, h_x = lambda/40 | 0.3167 / 0.3166 | 0.7295 / 0.7294 | -0.375 / -0.375 | 0 |
| RIP-CC / iRIP, h_x = lambda/10 | 0.2910 / 0.2906 | 0.7114 / 0.7113 | -0.379 / -0.378 | 0 |
| iRIP, h_x = lambda/5 = h_perp (RIP-CC rejects it) | 0.2050 +- 0.0010 | 0.6598 +- 0.0009 | -0.391 +- 0.003 | 0 |

examples/qed_collision.deck (1 GeV, a0 = 100) the same way, RIP-CC / iRIP: E_rad / E_0 0.6411 / 0.6405 at lambda/20 (iRIP with
h_perp = h_x 0.6447), 0.6438 / 0.6435 at lambda/40, 0.6260 / 0.6242 at lambda/10, iRIP at lambda/5 0.5517; <S2> after the pulse
0.785 / 0.786 at lambda/20 (h_perp = h_x 0.782), 0.782 / 0.783 at lambda/40, 0.789 / 0.790 at lambda/10, 0.837 at lambda/5.
The photon and pair spectra of iRIP and RIP-CC at the same step agree (chi^2 per bin 0.01 - 0.11 against the seed scatter).
Gauss residual <= 1.1e-12 in all runs. So:
- iRIP and RIP-CC give the same QED results at the same step (same random numbers, the same fields at the particles to
  round-off), also with h_perp < c dt.
- The step c dt = h_x is set by the laser and the orbits, not by the QED sub-steps: the results converge in h_x in the same way
  for both solvers (lambda/20 within 1 % of lambda/40, lambda/10 7.5 % fewer pairs, lambda/5 35 % fewer; at lambda/5 the
  quadratic shape gathers a lambda/5 wave with sinc^3(pi/5) = 0.82 of its amplitude; the largest <chi> is 0.82 / 0.85 of
  that at lambda/20 in the two decks). The sub-steps
  (`substep_prob`) keep every event probability below 1 at all steps (P>1 = 0); in a constant field (P5b) 4 times the step
  gives the same pair yield. Use iRIP to refine the transverse grid (small foci, RZ near the axis), not to coarsen h_x.

## References
- O. Mathiak, A. Pukhov et al., QED module for QV3D (in preparation); O. Mathiak, L. Reichwein, A. Pukhov, L. Ji,
  M. Buescher, arXiv:2605.19814 (2026).
- H.-H. Song, W.-M. Wang, Y.-T. Li, Phys. Rev. Lett. 129, 035001 (2022); Y.-F. Li et al., Phys. Rev. Lett. 122, 154801 (2019)
  and 124, 014801 (2020); D. Seipt, D. Del Sorbo, C. P. Ridgers, C. D. Murphy, M. Marklund, Phys. Rev. A 98, 023417 (2018).
- V. N. Baier, V. M. Katkov, V. M. Strakhovenko, Electromagnetic Processes at High Energies in Oriented Single Crystals (1998);
  A. A. Sokolov, I. M. Ternov, Sov. Phys. Dokl. 8, 1203 (1964); V. Bargmann, L. Michel, V. L. Telegdi, PRL 2, 435 (1959).
- C. P. Ridgers et al., J. Comput. Phys. 260, 273 (2014) (g(chi), LCFA Monte Carlo).
- Pairs: T. Erber, Rev. Mod. Phys. 38, 626 (1966); V. I. Ritus, J. Sov. Laser Res. 6, 497 (1985); for the spin- and
  polarization-resolved rates (compared above, 2026-11-11): D. Seipt, B. King, Phys. Rev. A 102, 052805 (2020),
  arXiv:2007.11837; Y.-Y. Chen, K. Z. Hatsagortsyan, C. H. Keitel, R. Shaisultanov, Phys. Rev. D 105, 116013 (2022),
  arXiv:2201.10863.
