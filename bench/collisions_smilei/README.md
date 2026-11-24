# Binary collisions and impact ionization: VLPL against Smilei 5.1

Five of Smilei's own collision benchmarks (Smilei-5.1/benchmarks/collisions/: thermalisation_ei1, temperature_isotropization1,
beam_relaxation1, Stopping_power1, ionization_rate1), with their parameters (lambda = 1 um, densities in n_c, temperatures in
m_e c^2) and fewer cells. All species are frozen (Smilei time_frozen, VLPL mobile = false): only the collisions act. Smilei: 1D,
VLPL: 2D (cartesian2d), the same time step and about the same number of macro-particles.

```
sbatch run_smilei.sbatch        # out_smilei_<case>/ (Smilei 5.1 on the CPU cluster: . $WORK/smilei/smilei_env.sh)
EXE=../../build/vlpl sbatch run_vlpl.sbatch   # out_vlpl_<case>/collisions.txt, and out_vlpl_<case>_s5 with angle = smilei5
. $WORK/smilei/smilei_env.sh; python3 compare.py
```

| case | physics | VLPL (Nanbu) | VLPL (angle = smilei5) | Smilei 5.1 | theory |
|---|---|---|---|---|---|
| thermal | e-i (m_i = 10) T_e 102 eV, T_i 92 eV, lnL 5 (intra 1000) | T_e - T_i 5.78 eV at t = 25 | 6.07 | 6.18 | NRL 4.79 |
| | max \|VLPL - Smilei\| / (T_e - T_i)_0 | 0.052 | 0.024 | | |
| iso | e-e, T_x = 1.1 T_perp (5 eV), lnL 2 | (T_x - T_perp)/initial 0.394 at t = 0.5 | 0.449 | 0.462 | NRL 0.301 |
| | max \|VLPL - Smilei\| / initial | 0.069 | 0.040 | | |
| beam | e beam v = 0.05 on ions (m = 10), lnL 3 | <u_x> 0.0117 at t = 251 | 0.0125 | 0.0123 | |
| | max \|VLPL - Smilei\| / <u_x>_0 | 0.012 | 0.004 | | |
| stop | e beams 10 / 20 / 30 keV on 5 keV electrons, lnL 3 | energy lost 0.217 / 0.273 / 0.247 keV | 0.211 / 0.270 / 0.246 | 0.207 / 0.270 / 0.232 | NRL drag 0.223 / 0.278 / 0.252 |
| ion | e (v = 0.03) on Al3+, lnL 3, impact ionization | ionized per ion 0.7739 | | 0.7673 (VLPL / Smilei 1.009) | |

- With the same angle law (angle = smilei5) VLPL and Smilei agree to 2.4 % (thermal), 4.0 % (iso; the statistical noise of
  (T_x - T_perp) / initial is a few % at 2e5 particles) and 0.4 % (beam) of the initial value. VLPL's default (Nanbu's
  distribution with Pérez's fits, <cos chi> = e^-s to 1.2e-3; the Smilei fit 2e-2) relaxes a bit faster at these time steps.
- At Smilei's benchmark time steps s per pair is 0.1-0.4 (mean <s>_ei = 0.40 in thermal): both codes relax slower than NRL, the
  time-step error of the binary method (a pair exchanges 1 - e^-s instead of s; VLPL's thermal at dt / 4 follows NRL closer,
  tests/test_collisions.cpp C4 converges with dt).
- Stopping: VLPL / NRL drag (test particle in a Maxwellian, d eps / dt = -2 (psi(x) - psi'(x)) nu0 eps with the relativistic
  cold-target rate n k0 r_e lnL / beta) 0.973 / 0.980 / 0.981, Smilei / NRL 0.931 / 0.972 / 0.920. VLPL's spread between seeds
  (7, 8, 9, 10): 0.215-0.222, 0.273-0.281, 0.244-0.254 keV.
- Impact ionization: <Z> of Al3+ -> 3.774 (VLPL) / 3.767 (Smilei) after 100 periods; the statistical error of 9600 ions is
  ~0.5 %. VLPL uses the NIST ionization energies (Smilei its own table) with Smilei's orbital table.
- Cost (VLPL, 12 threads on one cluster node): thermal 50k + 50k particles, 3 blocks: 4.9 ms per step (collisions 73 % of the run);
  ion: 1.3 ms per step for 2 x 9600 particles (84 %, two passes of the ionizing block).
