# Beam diagnostics: moments, energy spectrum, particle output

Status: implemented (`src/beam_diag.*`, deck block `diagnostic <name>`, decision 30)

## Physics and equations
For the particles of one species with γ >= γ_min (the accelerated electrons of an LWFA, a witness
beam), every `every` steps:
- macro-particle count; charge Q = q e Σ w n_c / k0³, with w in units n_c / k0³ (the loader's
  w = n dV / ppc) and n_c = ε0 m_e ω0² / e²;
- charge-weighted means and rms values (⟨a⟩ = Σ w a / Σ w):
  - kinetic energy E = (γ - 1) m c² (m in units of m_e): ⟨E⟩, rms spread σ_E / ⟨E⟩, maximum;
  - position: ⟨x⟩, σ_x, σ_y, σ_z;
- normalised rms emittances, with u = γ v / c:
  ε_n,y = sqrt(⟨δy²⟩ ⟨δu_y²⟩ - ⟨δy δu_y⟩²), and ε_n,z the same along z;
- spectrum dQ/dE: `bins` equal bins on [0, E_max], in pC / MeV.

## Discretisation
- The selection and all sums run on the device, as `par_sum` / `par_max_abs` kernels over the
  species.
- MPI: sums with `allreduce_sum`, the maximum with `allreduce_max`.
- Two passes: the means first, then the centred second moments Σ w (a - ⟨a⟩)². One-pass
  Σ w a² - (Σ w a)² / Σ w loses the energy spread of a GeV beam to cancellation (GEM-PIC-RZ: 27 % at
  σ_E/E = 4e-6, γ = 1000).
- Spectrum: one FP64 atomic add per selected particle into the bins. On the GPU and with OpenMP, the
  bin sums depend on the summation order at round-off.
- Particles (`particles = true`):
  - the selected indices are listed on the device (int atomic counter);
  - their x y z ux uy uz w are gathered into one compact buffer and copied to the host;
  - they are written in index order, so each rank's file is deterministic.

## Output (in output.dir)
- `beam_<name>.txt`, one line per output:
  - `step t[fs] N_macro Q[pC] E_mean[MeV] dE/E_rms E_max[MeV] eps_ny[um] eps_nz[um]`
  - `x_mean[um] sigma_x[um] sigma_y[um] sigma_z[um]`
- `spectrum_<name>.txt`: a header line with the bin centres in MeV, then one line per output with
  `step t[fs]` and dQ/dE [pC/MeV] per bin.
- `particles/<name>_<step>_r<rank>.bin`:
  - float64 columns x y z ux uy uz w, in code units: lengths 1/k0, u = γ v / c, w n_c / k0³;
  - `particles/<name>.txt` lists step, t and the count of each rank, and `particles/meta.txt` the
    unit factors.

## Deck interface
```
diagnostic injected {
  species   = electrons
  every     = 600          # steps (default output.every)
  gamma_min = 20           # select gamma >= gamma_min (default 1: all)
  bins      = 200          # spectrum bins (default 200; 0: no spectrum)
  e_max_mev = 500          # upper end of the spectrum in MeV (default 1000)
  particles = false        # also write the selected particles
}
```

## Tests
- Fast (`tests/test_diag.cpp`):
  - D1: random particles; the moments, charge and count above γ_min equal a direct host
    computation to 1e-12.
  - D2: particles on an ellipse y = a cos φ, u_y = b sin φ, with N equally spaced φ: ε_n,y = a b / 2
    exactly.
  - D3: weights at bin centres; the spectrum is exact, and its sum is Q.
  - D4: a run with `particles = true`; the files read back give the selected particles of the species.
- MPI (`tests/mpi.sh`, beam case): the beam file on 1 and 8 ranks agrees to round-off.

## References
- K. Floettmann, Phys. Rev. ST Accel. Beams 6, 034202 (2003): rms emittance definitions.
