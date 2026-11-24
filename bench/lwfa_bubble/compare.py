"""Compare VLPL (rip.deck) with FBPIC (fbpic_bubble.py) in the bubble regime.

python3 bench/lwfa_bubble/compare.py out_bubble_rip out_bubble_fbpic [tol_laser tol_length tol_l2]
Per output time, on axis (RIP: y = ny/2 of the slice at z = nz/2; FBPIC: theta = 0, all modes):
- the peak laser field |E_pol| and the position of the laser (peak of the field envelope);
- the bubble: its back = the minimum of Ex in the first bucket behind the laser (the accelerating
  spike), its length = laser position - back, the peak accelerating field -min Ex there, and the
  smallest electron density n_e / n_0 between the bubble centre (Ex = 0) and the laser (cavitation);
- the relative L2 difference of Ex over the first bucket (back - 1 um .. laser);
then, if VLPL wrote beam_injected.txt, the charge, mean and maximum energy of the electrons with
gamma >= 20 in both codes (reported, not gated).
Ex is averaged over one laser wavelength before the bubble is located (the laser's own Ex oscillates).
Times without plasma (n_e < n_0 / 2) or a wake (< 0.02 m c w0 / e) behind the laser are skipped. Exits 1 if the
laser field, the bubble length or the first-bucket L2 differ by more than the tolerances (defaults
0.10, 0.05, 0.25, relative).
"""
import os, sys
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

rip, fb = sys.argv[1], sys.argv[2]
tol_las, tol_len, tol_l2 = [float(v) for v in sys.argv[3:6]] if len(sys.argv) > 5 else (0.10, 0.05, 0.25)
lam = 0.8e-6
cl = 2.99792458e8
k0 = 2 * np.pi / lam
w0 = k0 * cl
E0 = 9.1093837015e-31 * cl * w0 / 1.602176634e-19                     # m c w0 / e [V/m]
n_c = 8.8541878128e-12 * 9.1093837015e-31 * w0**2 / 1.602176634e-19**2
n0 = 0.005                                                              # plateau density / n_c

meta = dict(l.split(None, 1) for l in open(os.path.join(rip, "slices", "meta.txt")))
nx, ny, dx = int(meta["nx"]), int(meta["ny"]), float(meta["dx"])
index = [l.split() for l in open(os.path.join(rip, "slices", "index.txt"))]

def rip_axis(name, step):
    a = np.fromfile(os.path.join(rip, "slices", "%s_%06d.f32" % (name, step)), dtype=np.float32).reshape(ny, nx)
    return a[ny // 2].astype(float)

ts = OpenPMDTimeSeries(os.path.join(fb, "hdf5"), check_all_files=False)

def fb_axis(field, comp, it):
    f, info = ts.get_field(field, comp, iteration=it, m="all", theta=0, slice_across="r")
    return info.z, f

def envelope(f, w=16):
    """running max of |f| over w cells (one wavelength at dx = lambda / 16)"""
    a = np.abs(f)
    return np.array([a[max(0, i - w // 2):i + w // 2 + 1].max() for i in range(len(a))])

def smooth(f, w=16):
    """running mean over w cells (one laser wavelength): removes the laser's own Ex, keeps the wake"""
    return np.convolve(f, np.ones(w) / w, mode="same")

def bubble(x, ex_raw, ne, epol):
    """laser position, bubble back, length, peak accelerating field, cavity density, first-bucket mask
    (found on the wavelength-averaged Ex)"""
    ex = smooth(ex_raw)
    env = envelope(epol)
    il = int(np.argmax(env))
    i = il
    while i > 0 and ex[i] <= 0: i -= 1          # in the laser: skip to the decelerating front (Ex > 0)
    while i > 0 and ex[i] > 0: i -= 1           # the bubble centre: Ex crosses zero
    ic = i
    while i > 0 and ex[i] <= 0: i -= 1          # accelerating half, up to the back
    ib = i + int(np.argmin(ex[i:ic + 1]))
    cav = float(np.min(ne[ic:il + 1])) if il > ic else float("nan")
    m = (x >= x[ib] - 1e-6) & (x <= x[il])
    return x[il], x[ib], x[il] - x[ib], -float(ex[ib]), cav, float(env[il]), m

fails = []
print("%7s | %8s %8s | %7s %7s | %8s %8s | %6s %6s | %6s" % ("t [fs]", "E_las RIP", "FB", "L_b RIP", "FB",
      "E_acc RIP", "FB", "n_cav", "FB", "L2"))
for step_s, t_s, x0_s in index:
    step = int(step_s)
    if step == 0 or step not in ts.iterations:
        continue
    x = (float(x0_s) + dx * np.arange(nx)) / k0                         # absolute x [m]
    ex, ez, ne = rip_axis("Ex", step), rip_axis("Ez", step), rip_axis("n_electrons", step) / n0
    zf, exf = fb_axis("E", "z", step)
    _, epf = fb_axis("E", "x", step)
    _, rhof = fb_axis("rho_electrons", None, step)
    exf, epf = np.interp(x, zf, exf) / E0, np.interp(x, zf, epf) / E0
    nef = np.interp(x, zf, -rhof / 1.602176634e-19 / n_c) / n0
    r = bubble(x, ex, ne, ez)
    f = bubble(x, exf, nef, epf)
    behind = x < r[0] - 2e-6
    if behind.sum() < 10 or np.max(nef[behind]) < 0.5 or np.max(np.abs(smooth(exf)[behind])) < 0.02:
        continue                                                        # no plasma or no wake behind the laser yet
    m = r[6] | f[6]
    l2 = np.sqrt(np.sum((ex[m] - exf[m]) ** 2) / np.sum(exf[m] ** 2))
    t_fs = float(t_s) / w0 * 1e15
    print("%7.0f | %8.3f %8.3f | %7.2f %7.2f | %8.4f %8.4f | %6.3f %6.3f | %6.3f" % (t_fs, r[5], f[5], r[2] * 1e6,
          f[2] * 1e6, r[3], f[3], r[4], f[4], l2))
    for what, a, b, tol in (("laser field", r[5], f[5], tol_las), ("bubble length", r[2], f[2], tol_len)):
        if abs(a - b) > tol * abs(b): fails.append("%s at %.0f fs: %.4g vs %.4g" % (what, t_fs, a, b))
    if l2 > tol_l2: fails.append("first-bucket L2 at %.0f fs: %.3f" % (t_fs, l2))
# injected electrons (gamma >= 20), not gated: self-injection is chaotic in this regime
bf = os.path.join(rip, "beam_injected.txt")
if os.path.exists(bf):
    rows = {int(float(l.split()[0])): [float(v) for v in l.split()] for l in open(bf) if not l.startswith("#")}
    print("\ninjected electrons, gamma >= 20 (VLPL: beam_injected.txt; FBPIC: particles)")
    print("%7s | %9s %9s | %9s %9s | %9s %9s" % ("t [fs]", "Q RIP pC", "FB", "<E> RIP", "FB", "E_max RIP", "FB"))
    mec2 = 0.51099895
    for step_s, t_s, x0_s in index:
        step = int(step_s)
        if step not in rows or step not in ts.iterations:
            continue
        r = rows[step]
        ux, uy, uz, w = ts.get_particle(["ux", "uy", "uz", "w"], species="electrons", iteration=step)
        g = np.sqrt(1 + ux**2 + uy**2 + uz**2)
        m = g >= 20
        q = np.sum(w[m]) * 1.602176634e-19 * 1e12
        ef = (g[m] - 1) * mec2
        emf = np.sum(w[m] * ef) / np.sum(w[m]) if m.any() else 0.0
        print("%7.0f | %9.4g %9.4g | %9.4g %9.4g | %9.4g %9.4g" % (float(t_s) / w0 * 1e15, r[3], q, r[4], emf, r[6],
              ef.max() if m.any() else 0.0))
print("tolerances: laser %.2f, bubble length %.2f, first-bucket L2 %.2f" % (tol_las, tol_len, tol_l2))
for s in fails: print("  exceeds: " + s)
sys.exit(1 if fails else 0)
