"""2D pictures of the boosted-frame drift boxes (gamma 5, 10; RIP-CC, 128 x 16 x 16, lab lambda/32, n0 = 1e-3 n_c):
E_y in the x-y plane (z = mid) at the last output (step 7200, about 1.2 x the residence time):
no filter; 2 binomial passes on the same colour scale; 2 passes on its own scale; (k_x, k_y) power of E_y without filter."""
import glob, os, sys, numpy as np, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__))


def last(run):
    fs = sorted(glob.glob(os.path.join(D, run, "openpmd", "data_*.h5")))
    with h5py.File(fs[-1], "r") as h:
        it = int(list(h["data"])[0]); m = h["data/%d/meshes" % it]
        ey = np.asarray(m["E"]["y"], dtype=float)
    return it, ey


fig, ax = plt.subplots(2, 4, figsize=(19, 6.6), constrained_layout=True)
for r, g in enumerate((5, 10)):
    it0, e0 = last("g%d_f0_map" % g); it2, e2 = last("g%d_f2_map" % g)
    nz, ny, nx = e0.shape
    s0, s2 = e0[nz // 2], e2[nz // 2]
    vm0, vm2 = np.abs(s0).max(), np.abs(s2).max()
    ext = [0, nx, 0, ny]
    panels = ((s0, vm0, "no filter"), (s2, vm0, "2 passes, same scale"), (s2, vm2, "2 passes, own scale"))
    for c, (s, vm, lab) in enumerate(panels):
        im = ax[r, c].imshow(s, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm, extent=ext)
        ax[r, c].set_title(r"$\gamma = %d$, %s (max $|E_y|$ = %.2g)" % (g, lab, np.abs(s).max()), fontsize=10)
        ax[r, c].set_xlabel(r"$x/h_x$ (plasma flows along $-x$)"); ax[r, c].set_ylabel(r"$y/h_y$")
        fig.colorbar(im, ax=ax[r, c], shrink=0.9)
    P = np.fft.fftshift((np.abs(np.fft.fftn(e0)) ** 2).sum(0))
    kx = np.fft.fftshift(np.fft.fftfreq(nx)) * 2; ky = np.fft.fftshift(np.fft.fftfreq(ny)) * 2
    im = ax[r, 3].pcolormesh(kx, ky, np.log10(P / P.max() + 1e-12), cmap="viridis", vmin=-6, vmax=0, shading="auto")
    ax[r, 3].set_title(r"$\gamma = %d$, no filter: $\log_{10}|E_y(k_x,k_y)|^2$ (sum over $k_z$)" % g, fontsize=10)
    ax[r, 3].set_xlabel(r"$k_x h_x/\pi$"); ax[r, 3].set_ylabel(r"$k_y h_y/\pi$"); fig.colorbar(im, ax=ax[r, 3])
fig.suptitle(r"3D drift boxes in a boosted frame: RIP-CC, $128\times16\times16$, lab $\lambda_0/32$, $n_0 = 10^{-3}n_c$; "
             r"$E_y$ (code units) in the plane $z$ = mid at step 7200 ($\approx1.2\times$ residence time)")
out = os.path.join(D, "maps_g5_g10.png")
fig.savefig(out, dpi=110); print(out)
