# iRIP against RIP-CC in the CHF focus (2026-11-06)

Results: bench/chf/README.md, section "Why iRIP's focus is 17 % below RIP-CC's". Paths point to $WORK/chfgap
on the CPU cluster; the binary is the main build (`make -j BUILD=...`), copied there as `vlpl`. Python with numpy:
`. $WORK/smilei/smilei_env.sh`.

- `r1_make.py`, `r1_ana.py`: Fresnel reflection off a 40 n_c step (0 deg; 30 deg s and p) for RIP-CC, iRIP and iRIP+O5 at
  lambda0/64 and /128. The decks follow tests/test_xflux.cpp `fresnel_deck`; r = A_left(plasma) / A_right(vacuum).
- `cmp_maps.py`: the band energy through every plane z and the axial fluence and peak maps of two CHF runs.
- `r3_make.py`, `r3b_ana.py`, `r3c_ana.py`: the vacuum focus of one harmonic order (lambda0 = 0.8/7 um, NA 0.5, 3 um from
  the open left edge, 2 modes) on the CHF grid (g1) and refined x2, x4 (g2, g4; r4: dr / 4 only). Orders 5 and 10 are the same
  decks with lambda0 = 0.8/N um and waist = lambda0 / (pi 0.5) (directories n5_*, n10_*). r3b: peak, position and widths;
  r3c: the radially integrated fluence and its rms radius along z.
- The CHF runs with open z edges: the RZ /128 iRIP deck of the series ($WORK/chf/irip/rr3.2/t128/chf.deck) with `boundary_z = open`, no
  `z_absorber_width` and no `tfsf_x`, `solver = rip_cc | irip | irip` + `closure = o5`.
