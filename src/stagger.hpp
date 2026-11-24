// Staggering descriptors (decision 11, docs/specs/architecture.md section 3). A field solver declares
// where its components live; gather, Esirkepov deposition and the charge density are generated from
// the descriptor, so no other file writes stagger offsets.
//
// Rules the generated kernels rely on (checked by the continuity test in tests/tests.cpp):
//   - J_a sits at rho + e_a/2 along axis a (mod 1 cell), at rho's offset along the other axes: then
//     Esirkepov deposition on the charge grid is exactly the continuity equation of the solver.
//   - All components are at integer time levels (a solver with B at n+1/2 adds a time offset later).
#pragma once

struct Loc { double o[3]; };              // offset in cells per grid axis

// RIP-CC (src/fields.hpp): all fields at integer x, transverse pairs (Ey, Bz), (Ez, By) co-located.
struct RipCC {
    static constexpr Loc E[3] = {{{0, 0, 0}}, {{0, 0.5, 0}}, {{0, 0, 0.5}}};
    static constexpr Loc B[3] = {{{0, 0.5, 0.5}}, {{0, 0, 0.5}}, {{0, 0.5, 0}}};
    static constexpr Loc rho = {{0.5, 0, 0}};
};

// Yee (docs/specs/yee.md): E_a at the edges, B_a at the faces, rho at the nodes; J_a at E_a. B is kept
// at integer time levels too (two half steps of Faraday's law per step), so the gather is the same.
struct Yee {
    static constexpr Loc E[3] = {{{0.5, 0, 0}}, {{0, 0.5, 0}}, {{0, 0, 0.5}}};
    static constexpr Loc B[3] = {{{0, 0.5, 0.5}}, {{0.5, 0, 0.5}}, {{0.5, 0.5, 0}}};
    static constexpr Loc rho = {{0, 0, 0}};
};

// Position of J_a, derived from rho.
template <class St> constexpr Loc current_loc(int a) {
    Loc l = St::rho;
    l.o[a] += 0.5;
    if (l.o[a] >= 1) l.o[a] -= 1;
    return l;
}
// Index shift of J_a along axis a: the current between charge nodes c and c+1 is stored at J index
// c + current_shift (0 or 1).
template <class St> constexpr int current_shift(int a) { return St::rho.o[a] + 0.5 >= 1 ? 1 : 0; }
