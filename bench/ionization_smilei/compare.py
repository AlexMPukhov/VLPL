#!/usr/bin/env python3
"""VLPL against Smilei 5.1: field ionization (bench/ionization_smilei/README.md).

  . $WORK/smilei/smilei_env.sh; python3 compare.py [out_rip out_smilei]

Prints the charge-state fractions of hydrogen and carbon at the end (VLPL ionization.txt; Smilei ParticleBinning),
their statistical errors, and the energies: the field energy at the end relative to the injected pulse energy, the electron
kinetic energy, and the ionization energy (VLPL E_ion, E_cur; Smilei: injected - field - kinetic).
"""
import sys
import numpy as np
import h5py

rip = sys.argv[1] if len(sys.argv) > 1 else 'out_rip'
sm = sys.argv[2] if len(sys.argv) > 2 else 'out_smilei'

# ---- VLPL
hdr = [l for l in open(rip + '/ionization.txt') if l.startswith('#')][0].split()
ion = np.loadtxt(rip + '/ionization.txt')
cols = ["step", "t", "E_ion", "E_cur", "W_tot", "capped"] + hdr[hdr.index("capped") + 1:]   # data columns
cols = cols[:cols.index('(charge')]
last = ion[-1]
def rstates(name, z):
    w = np.array([last[cols.index('%s_%d' % (name, k))] for k in range(z + 1)])
    return w / w.sum()
diag = np.loadtxt(rip + '/diag.txt')
# xflux-free accounting: the injected energy is the field energy once the pulse is fully in the box and before it meets the gas
# (column 2), reported by `wref` below
rip_H, rip_C = rstates('hydrogen', 1), rstates('carbon', 6)

# ---- Smilei
s = np.loadtxt(sm + '/scalars.txt')
names = [l.split()[2] for l in open(sm + '/scalars.txt') if l.startswith('# ') and l.split()[1].isdigit()]
S = {n: s[:, i] for i, n in enumerate(names)}
def sstates(f, z):
    with h5py.File(f) as h:
        k = sorted(h.keys())[-1]
        w = h[k][()]
    return w / w.sum(), k
sm_H, kH = sstates(sm + '/ParticleBinning0.h5', 1)
sm_C, kC = sstates(sm + '/ParticleBinning1.h5', 6)

def show(label, a, b, n_a, n_b):
    print(label)
    for k, (x, y) in enumerate(zip(a, b)):
        sig = np.sqrt(x * (1 - x) / n_a + y * (1 - y) / n_b) if 0 < x < 1 or 0 < y < 1 else 0
        print('  Z = %d   VLPL %.5f   Smilei %.5f   difference %+.5f  (%.1f sigma)' % (k, x, y, x - y, abs(x - y) / sig if sig > 0 else 0))

# macro-particle numbers: VLPL-RIP 2D (ny = 4 cells): 1280 x 4 cells x ppc; Smilei 1D: 1280 cells x ppc
cells = 1280
show('hydrogen at the end (charge-state fractions)', rip_H, sm_H, cells * 4 * 4, cells * 4)
show('carbon at the end', rip_C, sm_C, cells * 4 * 64, cells * 64)
print('mean charge: hydrogen %.5f / %.5f, carbon %.4f / %.4f (VLPL / Smilei)' % (
    np.dot(np.arange(2), rip_H), np.dot(np.arange(2), sm_H), np.dot(np.arange(7), rip_C), np.dot(np.arange(7), sm_C)))

# ---- energies (relative to the injected pulse energy)
uinj = S['Uelm_bnd'][-1]
print('Smilei: injected %.6e, field at the end %.6e (%.5f), electron kinetic %.3e (%.5f), injected - field - kinetic %.3e (%.5f)' % (
    uinj, S['Uelm'][-1], S['Uelm'][-1] / uinj, S['Ukin_electrons'][-1], S['Ukin_electrons'][-1] / uinj,
    uinj - S['Uelm'][-1] - S['Ukin'][-1], (uinj - S['Uelm'][-1] - S['Ukin'][-1]) / uinj))
t = diag[:, 1]
wref = diag[np.argmin(abs(t - 45 * 2.354564459136067)), 2]      # 45 fs: the pulse is in the box, the gas starts at 12 um
print('VLPL: field at 45 fs %.6e, at the end %.6e (%.5f), kinetic %.3e (%.5f), E_ion %.3e (%.5f), E_cur %.3e (%.5f)' % (
    wref, diag[-1, 2], diag[-1, 2] / wref, diag[-1, 3], diag[-1, 3] / wref, last[2], last[2] / wref, last[3], last[3] / wref))
