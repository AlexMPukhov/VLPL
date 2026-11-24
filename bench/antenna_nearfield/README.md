# Near field of the lab-fixed antenna in a boosted run

Status: measured 2026-11-24 (branch `antenna-near-field`, commit 2dede95 for the runs; epyc job 17300).
Question (from the QED-in-the-boosted-frame work, where a collision 2.7 um downstream of the antenna at gamma 5, lab lambda/20
lost 19 % of its pairs): how far downstream of the antenna (`laser.injection = antenna`, docs/specs/boosted_frame.md
"Antenna") is the emitted laser field still too weak, and what does that distance scale with?

## Result

The emitted field reaches the lab field only **one boosted step after its emission**. In lab units the deficient zone is

    d_1% = 1.1 gamma (1 + beta) c dt'  =  1.1 (gamma (1 + beta))^2 h_lab   (RIP-CC / iRIP, c dt' = h'_x)

downstream of the antenna (the lab distance light covers while one boosted step passes on the antenna's world line). Beyond
d_1% the amplitude is within 1 % of the lab run (0.990 at N = 1.0 - 1.1 boosted steps, then 0.994 - 1.005, 1.000 from
N = 1.8 on at lambda/40 and finer). Inside it is weaker: 0.58 - 0.66 at N < 0.2, 0.78 at 0.3, 0.90 at 0.55, 0.97 at 0.85.

The deficit **collapses onto one curve in N = x / ((gamma (1 + beta))^2 h_lab)** for gamma 5, 6, 10 and lab cells lambda/20,
/40, /80 (the scatter between the curves is the binning, +-0.03 at small N). It is **fixed in distance**: two lab snapshots 40 fs
apart, a pulse emitted 40 fs later, all give the same curve (not a start-up in time). It does **not** depend on the transverse
cell (dy 0.2 / 0.4 um), the particle shape (1 / 2), the current filter (0 / 2 passes: the antenna's current is added after the
filter) or the solver (iRIP / RIP-CC): identical to 3 digits. So the hypothesis is **supported**: the deficit is the extent of
the moving source in the boosted frame (the sheet's stretched spline spans (1 + beta) c dt' / h_x x 1.5 cells; a point is fed by
the completed sheet only once it is about one boosted step of light travel away from it). The antenna physics is unchanged here.

| case (iRIP, shape 2, no filter) | lab um per boosted step | N 0-0.25 | 0.25-0.5 | 0.5-0.75 | 0.75-1 | 1-1.25 | 1.25-1.5 | N >= 4 |
|---|---|---|---|---|---|---|---|---|
| gamma 2, lambda/20 | 0.56 | - | - | 0.855 | 0.957 | 1.003 | 1.003 | 0.997 |
| gamma 2, lambda/40 | 0.28 | - | 0.955 | 0.983 | 0.988 | 0.997 | 1.004 | 1.000 |
| gamma 5, lambda/20 | 3.92 | 0.655 | 0.789 | 0.906 | 0.970 | 0.994 | 0.999 | 0.997 |
| gamma 5, lambda/40 | 1.96 | 0.620 | 0.789 | 0.893 | 0.964 | 0.996 | 1.004 | 1.000 |
| gamma 5, lambda/80 | 0.98 | 0.637 | 0.754 | 0.891 | 0.970 | 0.994 | 1.004 | 1.000 |
| gamma 6, lambda/20 | 5.67 | 0.649 | 0.790 | 0.909 | 0.971 | 0.995 | 0.999 | 0.997 |
| gamma 6, lambda/40 | 2.84 | 0.613 | 0.780 | 0.897 | 0.966 | 0.995 | 1.002 | 1.000 |
| gamma 10, lambda/20 | 15.9 | 0.627 | 0.787 | 0.898 | - | 1.000 | 1.001 | 0.999 |
| gamma 5, lambda/20, shape 1 / filter 2 / RIP-CC (dy 0.8) / dy 0.2 / delay 3 tau | 3.92 | 0.655 | 0.789 | 0.906 | 0.970 | 0.994 | 0.999 | 0.997 |
| gamma 6, lambda/20, shape 1 + filter 2 | 5.67 | 0.649 | 0.790 | 0.909 | 0.971 | 0.995 | 0.999 | 0.997 |

A_boost / A_lab on the axis, pooled over 4 lab snapshots (pulse centre at 0, 12, 24, 36 um from the antenna), points with
A_lab > 0.3 of the snapshot maximum. At gamma 2 the lab snapshot planes (2 lab cells apart) resolve the first quarter step only
poorly (empty bins). The far-field value 0.997 at lambda/20 (1.000 at lambda/40) is a separate, resolution-converging offset of
the boosted run against the lab run, present at every distance.

Consistency with the QED case (gamma 5, lambda/20, 3.92 um per step): the measured carrier amplitude 0.82 / 0.94 / 0.99 with the
pulse centre at 1.9 / 2.7 / 3.6 um (N = 0.48 / 0.69 / 0.92) matches this curve (0.85 / 0.93 / 0.98).

What it means for decks:
- QED collision decks at gamma 5, lambda/20: d_1% = 4.3 um; the collision point must lie further downstream (or c dt' smaller).
- A boosted RZ LWFA deck (gamma 6, lambda0/32, h_lab = 0.025 um): d_1% = 3.9 um at the start of an 8 mm ramp where the
  density is < 0.001 n0: no effect.
- The boost summary prints d_1% for every antenna laser.

## Tight focus: a separate effect

The global peak of a tightly focused pulse is too low at large c dt', also far from the antenna (the focus here is 40 um
downstream, 10 boosted steps at gamma 5, lambda/20). On-axis envelope peak at the focus, boosted / lab (2D, iRIP, tau 15 fs):

| waist | gamma 2, lambda/20 | gamma 5, lambda/20 | gamma 5, lambda/40 | gamma 5, lambda/80 |
|---|---|---|---|---|
| 1.5 um | 0.9965 | 0.9668 | 0.9955 | 0.9993 |
| 3 um | - | 0.9998 | 1.0001 | - |

It depends strongly on the divergence (1.5 um: 3.3 %, 3 um: 0.02 % at gamma 5, lambda/20) and converges with c dt' (about third
order), not with the distance from the antenna. Reading (not verified further): the beam's angular spectrum, theta ~ lambda /
(pi w), is opened to theta' ~ gamma (1 + beta) theta in the boosted frame (2 rad for 1.5 um at gamma 5), and these nearly
transverse components have a large phase advance per boosted step. The 3D collision deck of the QED work showed -11 %; 2D has
one transverse dimension less. Remedy: smaller c dt' (a finer lab cell) or a smaller gamma_boost for tight foci.

## Files

- tests/test_boost_antenna.cpp: `nf_run` (lab run + boosted run with the antenna, lab snapshots, on-axis envelopes), the scan
  driver (`VLPL_NF_SCAN=1 build/tests test_boost_antenna`, parameters `NF_GAMMA NF_N NF_SOLVER NF_SHAPE NF_FILTER NF_DY NF_W
  NF_FOCUS NF_TAU NF_DELAYF NF_L NF_LY NF_COUNT NF_TFIRST NF_TEVERY NF_OUT`) and the suite test A3.
- nf.sbatch: the scan of this README (one epyc node, 21 cases, 42 min); pool.awk: the binned ratios (`awk -v bw=0.25 -v
  name=CASE -f pool.awk CASE/scan.txt`).
