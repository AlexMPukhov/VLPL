// O2 (docs/specs/xflux.md): the field leaving through the open x edges. The transport splits the
// transverse fields into the components moving along -x and +x; at the left edge the one moving along
// -x is the reflected field (exactly, also with the laser entering there), at the right edge the one
// moving along +x the transmitted field. Every step: the energy each carries out (dir/xflux.txt);
// every `xflux_every` steps: the components themselves on the edge plane (dir/xflux/, float64 per rank).
#pragma once
#include <memory>
#include "pipeline.hpp"

struct XFluxConfig {
    bool on = false;               // output.xflux
    long every = 0;                // output.xflux_every: field records every n steps (0: energies only)
};

std::unique_ptr<Operator> make_xflux_operator();
