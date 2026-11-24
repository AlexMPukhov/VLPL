#!/usr/bin/env python3
"""Plasma and laser numbers for a VLPL deck, and starting values of the grid that follow references/physics_rules.md.

  python3 vlpl_numbers.py --density 1e18 --a0 2                       # matched spot, c tau = lambda_p / 3
  python3 vlpl_numbers.py --density 1e18 --a0 2 --waist 15 --tau 30 --length 0.3
  python3 vlpl_numbers.py --density 1e18 --a0 2 --length 2 --boost 5  # the same in a boosted frame
  python3 vlpl_numbers.py --density 5e19 --a0 3 --geometry 3d --cells 32

  --density   electron density in cm^-3          --lambda0  laser wavelength in um (default 0.8)
  --a0        normalized laser amplitude         --waist    spot w0 in um (1/e of the field)
  --tau       FWHM of the intensity in fs        --length   propagation length in mm (default 1)
  --cells     cells per laser wavelength along the axis (default 16: first look; 32: quantitative)
  --geometry  rz (default) or 3d                 --modes    azimuthal modes of an RZ run (default 2)
  --boost     Lorentz factor of a boosted frame  --periods  plasma periods kept behind the pulse (default 1.5)

Needs only the python standard library. The grid values are starting values: check the deck with
`vlpl deck --check`, make a smoke run, and halve the cells once for any number that matters.
"""
import argparse
import math

C_UM_FS = 0.299792458


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--density", type=float, required=True)
    ap.add_argument("--a0", type=float, default=1.0)
    ap.add_argument("--lambda0", type=float, default=0.8)
    ap.add_argument("--waist", type=float)
    ap.add_argument("--tau", type=float)
    ap.add_argument("--length", type=float, default=1.0)
    ap.add_argument("--cells", type=int, default=16)
    ap.add_argument("--geometry", choices=["rz", "3d"], default="rz")
    ap.add_argument("--modes", type=int, default=2)
    ap.add_argument("--boost", type=float, default=0.0)
    ap.add_argument("--periods", type=float, default=1.5)
    a = ap.parse_args()

    lam = a.lambda0
    nc = 1.1148539e21 / lam**2                      # critical density, cm^-3
    n = a.density / nc                              # in n_c: the unit of the deck
    lp = lam / math.sqrt(n)                         # plasma wavelength, um
    skin = lp / (2 * math.pi)
    w_matched = 2 * math.sqrt(a.a0) * skin          # k_p w = 2 sqrt(a0)
    w0 = a.waist if a.waist else w_matched
    tau = a.tau if a.tau else lp / 3 / C_UM_FS
    zr = math.pi * w0**2 / lam
    power = 0.5 * math.pi * (w0 * 1e-4) ** 2 * a.a0**2 * 1.37e18 / lam**2 / 1e9     # GW, linear polarization
    pc = 17.4 / n                                   # GW
    e0 = 96.16 * math.sqrt(a.density / 1e18)        # cold wave-breaking field, GV/m
    gw = 1 / math.sqrt(n)

    print("plasma:  n = %.4g cm^-3 = %.4g n_c   (n_c = %.4g cm^-3 at %.3g um)" % (a.density, n, nc, lam))
    print("plasma wavelength = %.4g um, c/omega_p = %.4g um, omega_p/omega_0 = %.4g, gamma_wake ~ %.3g"
          % (lp, skin, math.sqrt(n), gw))
    print("laser:   a0 = %g, waist = %.4g um%s, duration (intensity FWHM) = %.4g fs%s"
          % (a.a0, w0, "" if a.waist else " (matched, k_p w = 2 sqrt(a0))", tau, "" if a.tau else " (c tau = lambda_p / 3)"))
    print("         Rayleigh length = %.4g um, power = %.4g GW = %.3g P_c (P_c = %.4g GW)" % (zr, power, power / pc, pc))
    print("wake:    E0 = m c omega_p / e = %.4g GV/m = %.4g in code units (m c omega_0 / e)" % (e0, math.sqrt(n)))

    # ---- grid along the axis -------------------------------------------------------------------------------------
    dx = lam / a.cells
    delay = 2 * tau                                 # default of laser.delay: the peak enters then
    head = C_UM_FS * delay                          # the pulse reaches this far ahead of its peak
    box = head * 2 + a.periods * lp                 # pulse (twice its half length) + the plasma periods behind it
    nx = int(math.ceil(box / dx / 16.0)) * 16
    box = nx * dx
    # the window must start after the pulse is in (>= 2 delay) and before light that entered at t = 0 reaches the
    # right edge (c window.start < box length): as late as that allows, rounded down to 5 fs
    wstart = math.floor((box - 4 * dx) / C_UM_FS / 5.0) * 5.0
    wstart = max(wstart, 2 * delay)
    x_peak = C_UM_FS * (wstart - delay)             # the peak when the window starts
    ramp = max(round(2 * lp, -1), 20.0)
    xmin = round(box + 2.0, 1)                      # the plasma begins ahead of the box
    length_um = a.length * 1e3
    t_end = math.ceil((wstart + (xmin + ramp + length_um - x_peak) / C_UM_FS) / 10.0) * 10.0
    steps = int(round(t_end * C_UM_FS / dx))
    wpdt = 2 * math.pi / a.cells * math.sqrt(n)

    # ---- transverse grid -----------------------------------------------------------------------------------------
    w_end = w0 * math.sqrt(1 + (length_um / zr) ** 2)
    h_phys = min(w0 / 20, skin / 5)
    ax = "z" if a.geometry == "rz" else "x"
    print()
    print("grid along %s (c dt = d%s):  d%s = lambda0 / %d = %.4g um, n%s = %d (window %.4g um: the pulse and %.3g plasma periods)"
          % (ax, ax, ax, a.cells, dx, ax, nx, box, (box - 2 * head) / lp))
    print("    omega_p dt = %.3g  (plasma frequency %.2g %% fast; limit 2)" % (wpdt, 100 * (2 * math.asin(wpdt / 2) / wpdt - 1)))
    print("    laser.delay = %.4g fs (default 2 x duration), window.start = %.4g fs, the plasma from %s = %.4g um on,"
          % (delay, wstart, ax, xmin))
    print("    up-ramp %.4g um, laser.focus = %.4g um (end of the ramp)" % (ramp, xmin + ramp))
    print("    grid.t_end = %.6g fs for %.4g mm of plasma: %d steps" % (t_end, a.length, steps))
    print("    smoke run: grid.t_end = %.4g fs (%d steps: the plasma has entered; it must end by itself to print 'done:')"
          % (wstart + 150, int((wstart + 150) * C_UM_FS / dx)))

    def cells(h):
        return float("%.2g" % h)

    if a.geometry == "rz":
        dr = cells(h_phys)
        explicit = dr >= 4 * dx and a.modes <= 2
        if not explicit and a.modes <= 2 and h_phys >= 3 * dx:
            dr, explicit = cells(4 * dx), True
        guided, free = 3 * w0, 3 * w_end
        print("radial grid:  dr = %.4g um (waist / 20 and (c/omega_p) / 5 allow %.3g um)" % (dr, h_phys))
        for name, r in (("a guided pulse (3 w0)", guided), ("an unguided pulse (3 w at the end, w = %.3g um)" % w_end, free)):
            print("    box for %s: species rmax = %.4g um, grid.nr = %d (%d + 32 absorber cells)"
                  % (name, r, int(math.ceil(r / dr)) + 32, int(math.ceil(r / dr))))
        if length_um > zr:
            print("    the run is %.2g Rayleigh lengths long: take the larger box unless guiding is established" % (length_um / zr))
        print("solver:  numerics.solver = %s  (%s)" % ("rip_cc" if explicit else "irip",
              "explicit: dr >= 4 dz with %d modes; --check verifies the exact condition" % a.modes if explicit
              else "implicit: dr < 4 dz or more than 2 modes"))
        print("particles: ppc_z = 1, ppc_r = 2, ppc_theta = %d (>= 2 (2 modes - 1)), numerics.shape = 2" % (2 * (2 * a.modes - 1)))
        ncell = nx * int(math.ceil(guided / dr))
        npart = ncell * 2 * 2 * (2 * a.modes - 1) * 2      # two species
    else:
        dy = cells(h_phys)
        cour = 2 * dx**2 / dy**2 + 0.25 * wpdt**2
        explicit = cour < 0.9
        if not explicit and 2 * dx**2 / (2 * dx) ** 2 + 0.25 * wpdt**2 < 0.9 and h_phys >= 1.5 * dx:
            dy = cells(max(1.6 * dx, dy))
            cour = 2 * dx**2 / dy**2 + 0.25 * wpdt**2
            explicit = cour < 0.9
        ny = int(math.ceil(6 * w0 / dy / 16.0)) * 16
        print("transverse grid:  dy = dz = %.4g um, ny = nz = %d (%.4g um = 6 w0 + ...; add pml or widen for an unguided pulse)"
              % (dy, ny, ny * dy))
        print("    explicit condition dx^2 (1/dy^2 + 1/dz^2) + (omega_p dt)^2 / 4 = %.3g (< 1 needed)" % cour)
        print("solver:  numerics.solver = %s" % ("rip_cc" if explicit else "irip  (ny, nz powers of two; MPI ranks along x only)"))
        print("particles: ppc_x = ppc_y = ppc_z = 1, numerics.shape = 2")
        ncell = nx * ny * ny
        npart = ncell * 2
    print("size:    %.3g cells in the window, about %.3g macro-particles, %d steps: %.3g particle steps"
          % (ncell, npart, steps, float(npart) * steps))
    print("         (time per step: from the 'done: N steps, T s' line of a smoke run, not from this number)")

    if a.boost > 1:
        g = a.boost
        b = math.sqrt(1 - 1 / g**2)
        s = g * (1 + b)
        dxl = lam / max(a.cells, 32)
        print()
        print("boosted frame, gamma = %g (beta = %.5f, gamma (1 + beta) = %.4g; gamma_wake ~ %.3g%s):"
              % (g, b, s, gw, "" if g < gw else ": TOO LARGE, the wake would not move forward"))
        print("    lab d%s = lambda0 / %d = %.4g um  ->  boosted cell and c dt' = %.4g um" % (ax, max(a.cells, 32), dxl, s * dxl))
        nlab = int(t_end * C_UM_FS / dxl)
        print("    steps: about %d (the lab run at this resolution: %d, divided by (gamma (1 + beta))^2 = %.3g) plus a few\n"
              "    times n%s = %d for the pulse to enter and the last lab time to leave the box"
              % (int(nlab / s**2), nlab, s**2, ax, int(nx * max(a.cells, 32) / a.cells)))
        print("    solver: irip (the transverse cell is below c dt'), numerics.shape = 2, numerics.filter_passes = %d"
              % (2 if box / skin > 100 else 1))
        print("    antenna (laser.injection = antenna): field within 1 %% only beyond d_1%% = 1.1 (gamma (1 + beta))^2 d%s = %.4g um"
              % (ax, 1.1 * s**2 * dxl))
        print("    a beam against the boost is %.4g times shorter in boosted cells: raise its ppc along the axis" % s**2)


if __name__ == "__main__":
    main()
