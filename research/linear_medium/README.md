# Linear-response media (Drude / Lorentz) in box iRIP: 2D prototype (2026-10-28)

Spec: `docs/specs/linear_medium.md`. numpy/scipy, periodic 2D box (x, y; d/dz = 0), staggering as
`research/irip_rz/nci2d.py`.

- `box2d.py`: box iRIP field step with a medium.
  - The CN current is driven by the Theorem-2 work field, with local elimination of the medium's own source.
  - Ex^{n+1}, Bx^{n+1} are solved by GMRES, preconditioned by the vacuum Fourier solver with the local Ex diagonal.
- `test_energy.py`: W + K + dissipated conserved to 1e-14 ... 1e-11, with random fields and random media (wp dt <= 30), dy = 0.4 dx.
- `setup.py`:
  - Gaussian beam pulses (forward Fourier components, staggered phases);
  - media built from sub-cell area fractions.
- `conv2d.py <mirror_p|mirror_s|lossy_p|lossy_s|glass_p|glass_s> <cells per lambda>`: oblique incidence in a 12 x 12 lambda box.
  - Compared with Fresnel, weighted over the beam spectrum.
  - Results are in `conv2d.jsonl`; the table is in the spec.

Not committed (cloud session only): the 1D scripts (`drude1d.py`, `drude_disc.py`, `conv.py`) and the movie
scenarios (`run.py`, `render.py`; 20 x 20 lambda, lambda/20, dy = dx/2). The movies were given to Alexander on 2026-10-28.
