"""Remark 'The current in the characteristic form' (Sec. 3): shared vs per-characteristic current sources.
2D (d/dz = 0), RIP-CC, E-polarisation (E_z, B_y, B_x), prescribed current J_z, h_y = 1.5 h_x, periodic box.
The transverse sources of the pair T^pm = E_z -+ B_y get the current in three ways:
  centre : J_z at the cell centre (x_{i+1/2}, t^{n+1/2}) for both members
  box    : the (x,t)-cell average of J_z for both members (what a face-flux deposition gives)
  diag   : each member gets the line integral of J_z along its own diagonal of the (x,t) cell
Checks: div B = delta_x B_x + delta_y mu_x B_y (Eq. gauss) stays at round-off for the shared sources and not for
'diag'; all three are second order for a smooth current."""
import numpy as np
def dyp(F, dy): return (np.roll(F, -1, 1) - F) / dy
def dym(F, dy): return (F - np.roll(F, 1, 1)) / dy
def mux(F): return 0.5 * (F + np.roll(F, -1, 0))
def smooth(x, y, t, Ly):        # oscillating antenna, switched on smoothly
    return np.exp(-((x - 20)**2 + (y - Ly/2)**2) / 4.0) * np.sin(1.3*t) * (1 - np.exp(-(t/3.0)**2))
def sheet(x, y, t, Ly):         # particle-like: a sheet of width 0.02 moving at 0.6 c in x
    return np.exp(-((x - 5 - 0.6*t)/0.02)**2) / (0.02*np.sqrt(np.pi)) * np.exp(-(y - Ly/2)**2/4.0)
def sources(J, n, D, dy, nx, ny, mode, nq):
    Ly = ny * dy; s, w = np.polynomial.legendre.leggauss(nq); s = 0.5*(s + 1); w = 0.5*w
    X, Y = np.meshgrid((np.arange(nx) + 0.5)*D, np.arange(ny)*dy, indexing="ij")
    if mode == "centre":
        j = J(X, Y, (n + 0.5)*D, Ly); return j, j
    if mode == "box":
        j = sum(a*b*J(X - D/2 + p*D, Y, (n + q)*D, Ly) for p, a in zip(s, w) for q, b in zip(s, w)); return j, j
    return (sum(a*J(X - D/2 + p*D, Y, (n + p)*D, Ly) for p, a in zip(s, w)),     # + diagonal
            sum(a*J(X + D/2 - p*D, Y, (n + p)*D, Ly) for p, a in zip(s, w)))     # - diagonal
def run(J, D, T, mode, nq=8, L=40.0):
    dy = 1.5*D; nx = int(round(L/D)); ny = int(round(L/dy))
    Ez, By, Bx = (np.zeros((nx, ny)) for _ in range(3)); bmax = 0.0
    for n in range(int(round(T/D))):
        jp, jm = sources(J, n, D, dy, nx, ny, mode, nq)
        dBx = dym(mux(Bx - 0.5*D*dyp(Ez, dy)), dy)                  # predictor, closure mu_x B_x^*
        Up1 = np.roll(Ez - By + D*(-dBx - jp), 1, 0)                 # T^+ moves to +x
        Um1 = np.roll(Ez + By, -1, 0) + D*(-dBx - jm)                # T^- moves to -x
        Ez1, By1 = 0.5*(Up1 + Um1), 0.5*(Um1 - Up1)
        Bx = Bx - D*dyp(0.5*(Ez + Ez1), dy)                          # corrector
        Ez, By = Ez1, By1; bmax = max(bmax, np.abs(By).max(), np.abs(Bx).max())
    divB = (np.roll(Bx, -1, 0) - Bx)/D + dyp(mux(By), dy)
    return Ez, D*np.abs(divB).max()/bmax                           # div B relative to max|B|/h
T = 24.0
res = {}
for name, J, nq in (("smooth", smooth, 8), ("sheet", sheet, 64)):
    for mode in ("centre", "box", "diag"):
        res[name, mode] = run(J, 0.5, T, mode, nq)[1]
        print(f"{name:6s} {mode:6s} h = 0.5: h max|div B| / max|B| = {res[name, mode]:.1e}")
d2 = run(smooth, 0.25, T, "diag")[1]; print(f"smooth diag   h = 0.25: {d2:.1e}")
assert max(res[k] for k in res if k[1] != "diag") < 1e-13
assert res["sheet", "diag"] > 1e-2 and res["smooth", "diag"] > 1e-5
ref = run(smooth, 0.0625, T, "box")[0]; r = ref[int(round(30/0.0625)), int(round(20/(1.5*0.0625)))]
for mode in ("centre", "box", "diag"):
    e = [abs(run(smooth, D, T, mode)[0][int(round(30/D)), int(round(20/(1.5*D)))] - r) for D in (0.5, 0.25, 0.125)]
    print(f"smooth {mode:6s} E_z error h = 0.5/0.25/0.125: " + " ".join(f"{x:.1e}" for x in e)
          + f"  orders {np.log2(e[0]/e[1]):.2f} {np.log2(e[1]/e[2]):.2f}")
    assert np.log2(e[0]/e[2]) / 2 > 1.9
