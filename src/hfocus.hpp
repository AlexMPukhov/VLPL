// Track O, CHF (docs/specs/harmonic_focus.md): the focus of the reflected harmonics inside the box. Every step the
// electric field at each node of a region goes through a band-pass in time (Butterworth, bilinear transform: a
// high-pass at n_min omega0 and a low-pass at n_max omega0, cascaded second-order sections, one state per node and
// component); per node the peak of |E_band|^2 over the run, its time and the fluence int |E_band|^2 dt are kept.
// RZ: the field at phi = 0 and phi = pi/2 (the polarisation plane and the plane across it for a laser polarised
// along x), reconstructed from the modes before the filter. Written to dir/harmonic_focus/ at the end of the run
// (and every `every` steps); tools/hfocus.py assembles the maps and gives the focus position, peak and spot size.
#pragma once
#include <memory>
#include <string>
#include "pipeline.hpp"

struct HFocusConfig {
    std::string name;                    // block harmonic_focus <name> (empty: the block harmonic_focus)
    double n_min = 4.5, n_max = 30;      // band edges, harmonic order omega / omega0
    int order = 8;                       // poles of each edge (even, <= 16)
    double lo[3] = {-1e300, -1e300, -1e300}, hi[3] = {1e300, 1e300, 1e300};   // region (code units; RZ: z, r)
    long every = 0;                      // also write every n steps (0: at the end only)
};

// The cascade of second-order sections (direct form II transposed): y = b0 x + s1, s1 = b1 x - a1 y + s2,
// s2 = b2 x - a2 y.
struct HFFilter {
    int ns = 0;
    double b0[16], b1[16], b2[16], a1[16], a2[16];
};
// The band-pass for the time step dt (code units, omega0 = 1); |H(omega)| of the cascade in gain (tests).
HFFilter hfocus_filter(double n_min, double n_max, int order, double dt);
double hfocus_gain(const HFFilter& f, double omega, double dt);

// one operator per block; output in dir/harmonic_focus (or dir/harmonic_focus_<name>)
std::unique_ptr<Operator> make_hfocus_operator(const HFocusConfig& c);
