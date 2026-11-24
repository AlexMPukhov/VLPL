# Charge-conserving deposition for RIP-CC in quasi-3D RZ with azimuthal modes (docs/specs/rz_modes.md).
# Grid (RIP-CC staggering, x -> z, y -> r, z -> phi): rho, J_phi at (i+1/2, r_j); J_z at (i, r_j);
# J_r at (i+1/2, r_{j+1/2}); r_j = j dr. Cell volumes V_j = 2 pi r_j dr dz (axis: pi (dr/2)^2 dz), so the
# solver's divergence per mode at a rho node is
#   div_m J = (Jz(i+1) - Jz(i))/dz + (r_{j+1/2} Jr_{j+1/2} - r_{j-1/2} Jr_{j-1/2})/(r_j dr) + (i m / r_j) Jphi_j
# (axis j = 0: 4 Jr_{1/2}/dr). Modes: F = F_0 + 2 Re sum_m F_m e^{i m phi}, F_m = (1/2 pi) int F e^{-i m phi}.
# A particle of charge q, linear shapes S_z (on the rho grid i+1/2), S_r (on r_j): W = S_z S_r, e = e^{-i m phi}.
#   Delta(W e) = (Delta W) e_bar + W_bar (Delta e),  e_bar = (e0+e1)/2, W_bar = (W0+W1)/2   (exact)
# Delta W: 2D Esirkepov in (z, r) -> fluxes through z faces and r surfaces (cumulative from the axis),
# times e_bar; W_bar Delta e -> J_phi = i r_j W_bar Delta e q / (m dt V_j)  (balances (i m / r_j) J_phi).
# Check: (rho_m^{n+1} - rho_m^n)/dt + div_m J = 0 per mode at every node (axis nodes: m = 0 only).
import numpy as np
rng = np.random.default_rng(3)
import os
Nz, Nr, M = 8, 10, 4
dz, dr, dt = 0.5, 0.4, 0.5
if os.environ.get('RZ_STABLE'):          # explicit RIP-CC stable: D^2 (lambda_max) < 1 for m < M
    dz, dr, dt = 0.08, 0.4, 0.08
V = 2 * np.pi * np.arange(Nr + 2) * dr * dr * dz; V[0] = np.pi * (dr / 2) ** 2 * dz
rh = (np.arange(Nr + 2) + 0.5) * dr

def shapes(p):
    """linear weights on the rho grid in z (nodes z_{i+1/2}) and on r_j: (iz, wz[2]), (jr, wr[2]), phi"""
    z, x, y = p
    r, phi = np.hypot(x, y), np.arctan2(y, x)
    sz = z / dz - 0.5; iz = int(np.floor(sz)); fz = sz - iz
    sr = r / dr; jr = int(np.floor(sr)); fr = sr - jr
    return iz, np.array([1 - fz, fz]), jr, np.array([1 - fr, fr]), phi

def deposit(p0, p1, q, rho0, rho1, Jz, Jr, Jp):
    iz0, wz0, j0, wr0, ph0 = shapes(p0); iz1, wz1, j1, wr1, ph1 = shapes(p1)
    # common 3 x 3 node window (moves < 1 cell): z indices ib..ib+2, r indices jb..jb+2
    ib, jb = min(iz0, iz1), min(j0, j1)
    S0z = np.zeros(3); S1z = np.zeros(3); S0r = np.zeros(3); S1r = np.zeros(3)
    S0z[iz0 - ib:iz0 - ib + 2] = wz0; S1z[iz1 - ib:iz1 - ib + 2] = wz1
    S0r[j0 - jb:j0 - jb + 2] = wr0; S1r[j1 - jb:j1 - jb + 2] = wr1
    Wz = 0.5 * np.outer(S1z - S0z, S0r + S1r)                    # Esirkepov 2D
    Wr = 0.5 * np.outer(S0z + S1z, S1r - S0r)
    Wbar = 0.5 * (np.outer(S0z, S0r) + np.outer(S1z, S1r))
    for m in range(M):
        e0, e1 = np.exp(-1j * m * ph0), np.exp(-1j * m * ph1)
        eb = 0.5 * (e0 + e1)
        for a in range(3):
            for b in range(3):
                i, j = (ib + a) % Nz, jb + b
                rho0[m, i, j] += q * S0z[a] * S0r[b] * e0 / V[j]
                rho1[m, i, j] += q * S1z[a] * S1r[b] * e1 / V[j]
        # z fluxes: Fz(face a+1) = -cumsum_a Wz; face index of node i+1/2 -> i+1 (Jz at integer i)
        Fz = -np.cumsum(Wz, axis=0) * eb                            # through the face above node a
        for a in range(3):
            for b in range(3):
                j = jb + b
                Jz[m, (ib + a + 1) % Nz, j] += q * Fz[a, b] / (V[j] / dz) / dt
        # r fluxes from the axis: Fr(surface above node b) = -cumsum_b Wr (zero below jb: no weight there)
        Fr = -np.cumsum(Wr, axis=1) * eb
        for a in range(3):
            for b in range(3):
                i, j = (ib + a) % Nz, jb + b
                Jr[m, i, j] += q * Fr[a, b] / (2 * np.pi * rh[j] * dz) / dt     # J_r at r_{j+1/2}
        if m > 0:
            for a in range(3):
                for b in range(3):
                    i, j = (ib + a) % Nz, jb + b
                    rj = j * dr
                    Jp[m, i, j] += 1j * rj * q * Wbar[a, b] * (e1 - e0) / (m * dt * V[j])

if __name__ == "__main__":
    shape = (M, Nz, Nr + 2)
    rho0 = np.zeros(shape, complex); rho1 = np.zeros(shape, complex)
    Jz = np.zeros(shape, complex); Jr = np.zeros(shape, complex); Jp = np.zeros(shape, complex)
    for n in range(300):
        while True:
            r = rng.uniform(0, (Nr - 3) * dr); phi = rng.uniform(-np.pi, np.pi)
            p0 = np.array([rng.uniform(0, Nz * dz), r * np.cos(phi), r * np.sin(phi)])
            d = rng.uniform(-1, 1, 3) * np.array([0.95 * dz, 0.6 * dr, 0.6 * dr])
            p1 = p0 + d
            if np.hypot(p1[1], p1[2]) < (Nr - 3) * dr: break
        p1[0] %= Nz * dz; p0[0] %= Nz * dz
        if abs(p1[0] - p0[0]) > dz: p1[0] = p0[0] + d[0]              # keep the unwrapped z for the shape
        deposit(p0, p1, rng.uniform(0.5, 1.5), rho0, rho1, Jz, Jr, Jp)
    # divergence per mode at the rho nodes
    worst = np.zeros(M); scale = np.abs(rho1).max()
    for m in range(M):
        for i in range(Nz):
            for j in range(Nr - 2):
                if j == 0:
                    if m > 0: continue                                  # axis: no Gauss point for m >= 1
                    div = (Jz[m, (i + 1) % Nz, 0] - Jz[m, i, 0]) / dz + 4 * Jr[m, i, 0] / dr
                else:
                    rj = j * dr
                    div = (Jz[m, (i + 1) % Nz, j] - Jz[m, i, j]) / dz + (rh[j] * Jr[m, i, j] - rh[j - 1] * Jr[m, i, j - 1]) / (rj * dr) \
                          + 1j * m / rj * Jp[m, i, j]
                res = (rho1[m, i, j] - rho0[m, i, j]) / dt + div
                worst[m] = max(worst[m], abs(res) * dt / scale)
    print("continuity per mode, max |d rho/dt + div J| dt / max|rho|:", ["m=%d %.1e" % (m, w) for m, w in enumerate(worst)])
