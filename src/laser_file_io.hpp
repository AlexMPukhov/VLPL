// Internal: the LASY reader shared by the Cartesian file laser (laser_file.cpp) and the RZ one (laser_file_rz.cpp).
#pragma once
#include <complex>
#include <string>
#include <vector>
#include "laser_file.hpp"

namespace lasy {
using cd = std::complex<double>;

// The file in code units on a regular grid (u, Y, Z), Y = LASY x, Z = LASY y: e[((iz * ny) + iy) * nu + iu]
struct Grid {
    int nu = 0, ny = 1, nz = 1;
    double u0 = 0, du = 1, y0 = 0, dy = 1, z0 = 0, dz = 1;
    bool uy = false, uz = false;       // uniform along y / z (an axis of length 1)
    std::vector<cd> e;
    cd& at(int iu, int iy, int iz) { return e[(size_t(iz) * ny + iy) * nu + iu]; }
};

// A thetaMode file as azimuthal modes: env(u, r, theta) = sum_{k = -K .. K} a_k(u, r) e^{-i k theta}, theta from LASY's x
// towards y; nodes r_j = r0 + j dr; a[((k + K) * nu + l) * nr + j]
struct Modes {
    int nu = 0, nr = 0, K = 0;
    double u0 = 0, du = 1, r0 = 0, dr = 1;
    std::vector<cd> a;
    cd& at(int k, int l, int j) { return a[(size_t(k + K) * nu + l) * nr + j]; }
    const cd& at(int k, int l, int j) const { return a[(size_t(k + K) * nu + l) * nr + j]; }
};

// Reads the envelope in code units (E in m c omega0 / e, a vector potential converted to E). With modes != nullptr a
// thetaMode file is returned in *modes and the Grid stays empty (no resampling on a plane); a cartesian file always comes
// back as the Grid.
Grid read_file(const LaserFileConfig& c, double lambda0, double& omegaL, cd& px, cd& py, std::vector<std::string>& info,
               Modes* modes = nullptr);
void fft(cd* a, size_t n, long stride, int sign);   // a_q <- sum_l a_l exp(sign 2 pi i q l / N), N a power of two
size_t pow2(size_t n);
std::string fmt(const char* f, double a, double b = 0, double c = 0);
}  // namespace lasy
