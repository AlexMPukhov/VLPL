# O5 (docs/specs/overdense.md): oblique phase error of the closures over the CHF path, from their dispersion relations.
# "O5 full" adds to the paraxial factor of the spec a transverse factor 1 + Lam (D^2/6 + dr^2/12) on the closure (a
# transverse Laplacian on Ex_bar, Bx_bar), which cancels the leading time-step and transverse-grid errors.
# Run: python3 research/o5/dispersion.py (numpy, scipy)
import numpy as np
# Oblique dispersion of the closures at fixed omega and k_perp: phase error of kz over the CHF path (3 um = 3.75 lambda0).
# a = w D/2, b = kz D/2, q = D^2 Lam/4, Lam = (2/dr sin(kp dr/2))^2 (discrete transverse eigenvalue)
#   RIP-CC   sin^2 a = sin^2 b + q cos^2 b
#   iRIP     tan^2 a = tan^2 b + q
#   O5 par.  tan^2 a = tan^2 b + q (1 + 4/3 sin^2 b)               (overdense.md: Ex_bar with (1 - dx^2/3 delta_x^2))
#   O5 wide  tan^2 a = tan^2 b + q (1 + 4/3 sin^2 b + 2/3 q)      (+ a transverse term (D^2/6) L_perp on Ex_bar)
#   floor    a^2 = b^2 + q  (exact time stepping, the discrete transverse operator only: the dr part)
from scipy.optimize import brentq
L = 3.0 / 0.8
def kz_of(rel, a, q, D):
    f = lambda b: rel(a, b, q)
    return 2 / D * brentq(f, 1e-12, np.pi / 2 - 1e-9) if f(1e-12) * f(np.pi / 2 - 1e-9) < 0 else np.nan
rels = {
    "RIP-CC": lambda a, b, q: np.sin(a) ** 2 - np.sin(b) ** 2 - q * np.cos(b) ** 2,
    "iRIP": lambda a, b, q: np.tan(a) ** 2 - np.tan(b) ** 2 - q,
    "O5 par": lambda a, b, q: np.tan(a) ** 2 - np.tan(b) ** 2 - q * (1 + 4 / 3 * np.sin(b) ** 2),
    "O5 wide": lambda a, b, q: np.tan(a) ** 2 - np.tan(b) ** 2 - q * (1 + 4 / 3 * np.sin(b) ** 2 + 2 / 3 * q),
    "O5 full": lambda a, b, q: np.tan(a) ** 2 - np.tan(b) ** 2 - q * (1 + 4 / 3 * np.sin(b) ** 2 + 2 / 3 * q + q * RD2 / 3),
    "dr floor": lambda a, b, q: a * a - b * b - q,
}
for R, rr in ((128, 40), (128, 80), (256, 80)):
    D, dr = 1.0 / R, 1.0 / rr
    globals()["RD2"] = (dr / D) ** 2
    print("dz = lambda0/%d, dr = lambda0/%d: phase error (rad) over 3 um" % (R, rr))
    print("   N theta " + "".join("%10s" % k for k in rels))
    for N in (7, 15):
        w = 2 * np.pi * N
        for th in (0.2, 0.4, 0.6):
            kz0, kp = w * np.cos(th), w * np.sin(th)
            lam = (2 / dr * np.sin(kp * dr / 2)) ** 2
            row = [(kz_of(r, w * D / 2, D * D * lam / 4, D) - kz0) * L for r in rels.values()]
            print("  %2d  %.1f " % (N, th) + "".join("%+10.3f" % e for e in row))
