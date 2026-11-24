// Moving window along +x (docs/specs/moving_window.md, decision 22). With c dt = h_x a window at the
// speed of light shifts by exactly one cell per step: the fields move one cell towards -x in storage,
// Domain::wx counts the shifts (particles keep their absolute positions), new plasma is loaded into
// the cell that enters at the right edge, and the plasma behind the window leaves through the open
// left edge with exact currents (decision 21). Operator `moving_window` of stage Fields, after rip_cc.
#pragma once
#include <memory>
#include "comm.hpp"
#include "grid.hpp"
#include "pipeline.hpp"

struct WindowConfig {
    bool enabled = false;
    double start = 0;              // code units (1/omega0): first shift once t - start >= h_x / v
    double v = 1;                  // window velocity in units of c, 0 < v <= 1 (at most one shift per step)
};

// f(i) = f(i + 1) over the owned cells of each field; the last cell takes the right guard (the right
// neighbour's cell 0, zero beyond an open edge). tmp: a field of the same layout used as scratch. The
// y, z guards move with the data; the x guards need an exchange_x afterwards. left: the guard layer
// i = -1 takes cell 0 (0 at an open left edge).
void shift_fields_x(const Domain& d, std::vector<Field*> f, Field& tmp, bool left = false);
std::unique_ptr<Operator> make_window_operator();
// Shifts of the window after the step that ends at t (window cell size h).
long window_shifts(const WindowConfig& W, double t, double h);
