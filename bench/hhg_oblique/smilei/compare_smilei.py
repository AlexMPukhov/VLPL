# Reflected-field spectrum of the Smilei oblique HHG run (probe plane x' = 0.5 um, Probes0.h5).
#   python3 compare_smilei.py <plasma_dir> <vacuum_dir> <scale> [out_prefix]
# scale = a0 parameter of the plasma run / a0 parameter of the vacuum run (vacuum fields scale linearly).
# Reflected field = plasma - scale * vacuum (exact in vacuum on the same grid and timestep).
# Outbound (-x) components T1 = Ey - Bz, T2 = Ez + By; flux along -x = (T1^2 + T2^2)/4 (normalised units, c = 1).
# Inbound (+x) components of the vacuum run: I1 = Ey + Bz, I2 = Ez - By; flux along +x = (I1^2 + I2^2)/4.
# Energies in units of n_c m_e c^2 (c/omega0)^3 (= 2.944e-7 J at 0.8 um), like Smilei's scalars.
# Spectrum written exactly like VLPL tools/xflux.py: F = rfft(T, axis=time) * dt for T1, T2 at every probe point,
# S(omega) = sum over points and both components |F|^2 * dy_probe * dz_probe / 4, omega = 2 pi rfftfreq(n, dt);
# by Parseval S = pi dW/domega (no window). Files: <out>.txt (reflected), <out>_vacuum.txt (vacuum, inbound components).
import sys, re, h5py, numpy as np
from math import pi
pdir, vdir, scale = sys.argv[1], sys.argv[2], float(sys.argv[3])
out = sys.argv[4] if len(sys.argv) > 4 else "spectrum_smilei"
lam = 0.8; um = 2*pi/lam; fs = 2*pi*0.299792458/lam; Junit = 2.944e-7
log = open(f"{pdir}/smilei.log").read()
dt = float(re.search(r"timestep = ([0-9.]+)", log).group(1))
cl = [float(v) for v in re.search(r"cell_length\s*:\s*([0-9.e+-]+),\s*([0-9.e+-]+),\s*([0-9.e+-]+)", log).groups()] \
     if re.search(r"cell_length\s*:", log) else None
fp, fv = h5py.File(f"{pdir}/Probes0.h5", "r"), h5py.File(f"{vdir}/Probes0.h5", "r")
names = fp.attrs["fields"].decode().split(",")
iEy, iEz, iBy, iBz = (names.index(n) for n in ("Ey", "Ez", "By", "Bz"))
pos = fp["positions"][()] / um
ys, zs = np.unique(np.round(pos[:, 1], 6)), np.unique(np.round(pos[:, 2], 6))
dA = (ys[1] - ys[0]) * (zs[1] - zs[0]) * um**2         # area per probe point, (c/omega0)^2
ks = sorted(set(k for k in fp.keys() if k.isdigit()) & set(k for k in fv.keys() if k.isdigit()))
nt, npnt = len(ks), pos.shape[0]
print(f"dt = {dt} /omega0 ({dt/fs:.4f} fs), {nt} steps, {npnt} points ({len(ys)} x {len(zs)}), "
      f"spacing {ys[1]-ys[0]:.4f} x {zs[1]-zs[0]:.4f} um, x' = {pos[0,0]:.3f} um")
T1 = np.empty((nt, npnt), np.float32); T2 = np.empty_like(T1)
I1 = np.empty_like(T1); I2 = np.empty_like(T1)
Wout_vac = 0.
for j, k in enumerate(ks):
    p = fp[k][()]; v = fv[k][()] * scale
    r = p - v
    T1[j] = r[iEy] - r[iBz]; T2[j] = r[iEz] + r[iBy]
    I1[j] = v[iEy] + v[iBz]; I2[j] = v[iEz] - v[iBy]
    Wout_vac += np.sum((v[iEy] - v[iBz])**2 + (v[iEz] + v[iBy])**2)
Wout_vac *= dA * dt / 4
W_ref = dA * dt / 4 * float(np.sum(T1.astype(np.float64)**2) + np.sum(T2.astype(np.float64)**2))
W_inc = dA * dt / 4 * float(np.sum(I1.astype(np.float64)**2) + np.sum(I2.astype(np.float64)**2))
# spectra: sum over points of |rfft|^2, in column chunks
def spectrum(A, B):
    S = np.zeros(nt//2 + 1)
    for c in range(0, npnt, 4096):
        for X in (A, B):
            F = np.fft.rfft(X[:, c:c+4096].astype(np.float64), axis=0)
            S += np.sum(np.abs(F)**2, axis=1)
    return S * dt**2 * dA / 4                             # RIP xflux.py normalisation: S = pi dW/domega
om = 2*pi*np.fft.rfftfreq(nt, dt)                        # omega / omega0
dom = om[1] - om[0]
Sr, Si = spectrum(T1, T2), spectrum(I1, I2)
hdr = (f"Smilei v5.1, oblique HHG code check (n0 = 6 n_c, a0 = 2, dt = {dt:.5g} / omega0), probe x' = 0.5 um\n"
       "normalisation as VLPL tools/xflux.py: F = rfft(T)*dt, S = sum |F|^2 dy dz / 4 (= pi dW/domega)\n"
       "units: omega0, E and B in m c omega0/e, t in 1/omega0, lengths in c/omega0\n")
np.savetxt(f"{out}.txt", np.column_stack([om, Sr]), fmt="%.6e", header=hdr +
           f"reflected field (plasma - {scale} x vacuum), T1 = Ey - Bz, T2 = Ez + By; W_ref = {W_ref:.6e}\n"
           "columns: omega/omega0  S")
np.savetxt(f"{out}_vacuum.txt", np.column_stack([om, Si]), fmt="%.6e", header=hdr +
           f"vacuum run x {scale}, INBOUND components Ey + Bz, Ez - By (approximate for the 45 deg beam); W_inc = {W_inc:.6e}\n"
           "columns: omega/omega0  S")
Sr, Si = Sr/pi, Si/pi                                    # dW/domega for the band energies below
print(f"check Parseval: sum(S_ref) domega = {Sr.sum()*dom:.6e} vs time domain {W_ref:.6e}; "
      f"incident {Si.sum()*dom:.6e} vs {W_inc:.6e}")
print(f"incident energy (inbound through the probe, vacuum x scale) W_inc = {W_inc:.5e} = {W_inc*Junit*1e3:.4f} mJ")
print(f"reflected energy W_ref = {W_ref:.5e} = {W_ref*Junit*1e3:.4f} mJ, W_ref / W_inc = {W_ref/W_inc:.5f}")
print(f"vacuum outbound (noise floor / boundary reflection) = {Wout_vac:.3e} = {Wout_vac/W_inc:.2e} W_inc")
with open(f"{out}_bands.txt", "w") as g:
    g.write("# N, W in [N-0.5, N+0.5] omega0 (reflected), fraction of W_inc, fraction of W_ref\n")
    print(" N    W_band        /W_inc       /W_ref")
    for N in range(1, 33):
        s = (om >= N - 0.5) & (om < N + 0.5)
        Wb = Sr[s].sum() * dom
        g.write(f"{N:3d} {Wb:.6e} {Wb/W_inc:.6e} {Wb/W_ref:.6e}\n")
        print(f"{N:3d}  {Wb:.4e}  {Wb/W_inc:.4e}  {Wb/W_ref:.4e}")
    g.write(f"# total reflected {W_ref:.6e}, incident {W_inc:.6e}, ratio {W_ref/W_inc:.6f}\n")
# vacuum energy from DiagScalar (Uelm at its maximum, x scale^2)
a = np.loadtxt(f"{vdir}/scalars.txt")
cols = [l.split()[2] for l in open(f"{vdir}/scalars.txt") if l.startswith("#") and len(l.split()) > 2 and l.split()[1].isdigit()]
U = a[:, cols.index("Uelm")] * scale**2
print(f"vacuum DiagScalar: max Uelm x scale^2 = {U.max():.5e} at t = {a[np.argmax(U), 0]/fs:.1f} fs; "
      f"W_ref / Uelm_max = {W_ref/U.max():.5f}; inbound-probe W_inc / Uelm_max = {W_inc/U.max():.4f}")
