// Helpers shared by the test files (tests/tests.cpp defines them).
#pragma once
#include <string>
#include <vector>
#include "comm.hpp"
#include "grid.hpp"

extern int failures;
void check(const std::string& name, bool ok, const std::string& info);
std::string fmt(const char* f, double a, double b = 0);

// Host mirror of a device field: read with H(f), modify, write back with put().
struct H {
    const Field* f; std::vector<double> v;
    explicit H(const Field& fl) : f(&fl), v(fl.host()) {}
    double& operator()(int i, int j, int k) { return v[f->idx(i, j, k)]; }
    void put(Field& fl) { fl.upload(v); }
};

void test_window(Comm& comm);              // tests/test_window.cpp
void test_profile(Comm& comm);             // tests/test_profile.cpp
void test_beam(Comm& comm);                // tests/test_beam.cpp
void test_diag(Comm& comm);                // tests/test_diag.cpp
void test_pml(Comm& comm);                 // tests/test_pml.cpp
void test_laser(Comm& comm);               // tests/test_laser.cpp
void test_openpmd(Comm& comm);             // tests/test_openpmd.cpp
void test_lasy(Comm& comm);                // tests/test_lasy.cpp
void test_yee(Comm& comm);                 // tests/test_yee.cpp
void test_irip(Comm& comm);                // tests/test_irip.cpp
void test_rz(Comm& comm);                  // tests/test_rz.cpp
void test_rz_grid(Comm& comm);             // tests/test_rz_grid.cpp
void test_xflux(Comm& comm);               // tests/test_xflux.cpp
void test_fresnel(Comm& comm);             // tests/test_xflux.cpp (oblique Fresnel)
void test_tfsf(Comm& comm);                // tests/test_tfsf.cpp
void test_x_absorber(Comm& comm);          // tests/test_tfsf.cpp (O3)
void test_tfsf_modes(Comm& comm);          // tests/test_tfsf.cpp (exact plane waves at the TFSF plane)
void test_side(Comm& comm);                // tests/test_side.cpp (side injection)
void test_side_ff(Comm& comm);             // tests/test_side.cpp (side injection, flying focus)
void test_side_plasma(Comm& comm);         // tests/test_side.cpp (side injection in a background plasma)
void test_rz_tfsf(Comm& comm);             // tests/test_rz_tfsf.cpp (RZ TFSF plane, z layers, injected energy)
void test_rz_shape(Comm& comm);            // tests/test_rz_shape.cpp (RZ transverse particle shape, decision 79)
void test_rz_axis(Comm& comm);             // tests/test_rz_axis.cpp (RZ axis cells in the modes m >= 1, decision 106)
void test_hfocus(Comm& comm);              // tests/test_hfocus.cpp (CHF diagnostic)
void test_solid(Comm& comm);               // tests/test_solid.cpp (O4 solid-density targets)
void test_cart2d(Comm& comm);              // tests/test_cart2d.cpp (2D Cartesian geometry)
void test_mover(Comm& comm);               // tests/test_mover.cpp (O6 implicit mover)
void test_rz_metal(Comm& comm);           // tests/test_rz_metal.cpp (RZ embedded metal, self-field)
void test_rz_metal_phi(Comm& comm);       // tests/test_rz_metal_phi.cpp (RZ metal depending on phi, decision 103)
void test_structure(Comm& comm);           // tests/test_structure.cpp (Cart3D metal, quadrupoles, beam files, wake diagnostics)
void test_rz_structure(Comm& comm);        // tests/test_rz_structure.cpp (RZ quadrupoles, beam files, wake probe, slices; decision 105)
void test_rz_dielectric(Comm& comm);        // tests/test_rz_dielectric.cpp (RZ dielectric blocks, RIP-CC, decision 100)
void test_dielectric3d(Comm& comm);         // tests/test_dielectric3d.cpp (Cart3D / 2D dielectric, RIP-CC, decision 100)
void test_medium(Comm& comm);              // tests/test_medium.cpp (O7 linear media)
void test_medium2(Comm& comm);             // tests/test_medium2.cpp (O7: deck, particles, RZ, parabolic mirror)
void test_bound(Comm& comm);               // tests/test_bound.cpp (O7 model = bound: Maxwell-Bloch atoms, decision 85)
void test_ionization(Comm& comm);         // tests/test_ionization.cpp (field ionization, decision 86)
void test_boost(Comm& comm);               // tests/test_boost.cpp (boosted frame, decision 87)
void test_boost_antenna(Comm& comm);       // tests/test_boost_antenna.cpp (lab-fixed laser antenna, decision 89)
void test_overlap(Comm& comm);             // tests/test_overlap.cpp (multi-GPU migration, guard exchange, split push; decision 92)
void test_qed(Comm& comm);                 // tests/test_qed.cpp (spin-resolved QED, decision 93)
void test_collisions(Comm& comm);          // tests/test_collisions.cpp (binary collisions, decision 94)
void test_pairs(Comm& comm);               // tests/test_pairs.cpp (Breit-Wheeler pairs, decision 96)
