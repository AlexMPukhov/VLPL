# RZ laser wakefield with many azimuthal modes

`rip_rz.deck`: the parameters of the laser-wakefield example of FBPIC (a0 = 4, w0 = 5 um, tau = 16 fs, n_e = 4e18 cm^-3,
a linear ramp, moving window) in VLPL-iRZ, with the number of azimuthal modes and particles in theta from the command
line:

    build/vlpl bench/fbpic_nm_scan/rip_rz.deck grid.modes=10 "species electrons.ppc_theta=40" "species ions.ppc_theta=40"

Used by the multi-GPU scaling runs (`bench/jupiter/`, case `rz_nm10`) and, without the laser, by the unit test R-B7
(many modes, thermal plasma: no mode grows, Gauss law per mode at round-off).
