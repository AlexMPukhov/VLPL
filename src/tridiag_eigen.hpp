// Symmetric tridiagonal eigenproblem (host, setup only): iRIP's transverse eigenbases (src/irip_pml.cpp,
// src/rz_irip.cpp).
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

// Eigenvalues d and orthonormal eigenvectors (columns of z, row-major n x n) of the symmetric tridiagonal
// matrix with diagonal d and off-diagonal e (e[i] couples i and i+1): QL with implicit shifts.
inline void tridiag_eigen(int n, std::vector<double>& d, std::vector<double> e, std::vector<double>& z) {
    z.assign(size_t(n) * n, 0.0);
    for (int i = 0; i < n; ++i) z[size_t(i) * n + i] = 1;
    e.push_back(0.0);
    for (int l = 0; l < n; ++l) {
        for (int iter = 0;; ++iter) {
            int m = l;
            for (; m < n - 1; ++m) {
                const double dd = std::abs(d[size_t(m)]) + std::abs(d[size_t(m + 1)]);
                if (std::abs(e[size_t(m)]) + dd == dd) break;
            }
            if (m == l) break;
            if (iter > 200) throw std::runtime_error("tridiag_eigen: no convergence");
            double g = (d[size_t(l + 1)] - d[size_t(l)]) / (2 * e[size_t(l)]);
            double r = std::hypot(g, 1.0);
            g = d[size_t(m)] - d[size_t(l)] + e[size_t(l)] / (g + (g >= 0 ? r : -r));
            double s = 1, c = 1, p = 0;
            int i = m - 1;
            for (; i >= l; --i) {
                double f = s * e[size_t(i)];
                const double b = c * e[size_t(i)];
                r = std::hypot(f, g);
                e[size_t(i + 1)] = r;
                if (r == 0) { d[size_t(i + 1)] -= p; e[size_t(m)] = 0; break; }
                s = f / r; c = g / r;
                g = d[size_t(i + 1)] - p;
                r = (d[size_t(i)] - g) * s + 2 * c * b;
                p = s * r;
                d[size_t(i + 1)] = g + p;
                g = c * r - b;
                for (int k = 0; k < n; ++k) {
                    f = z[size_t(k) * n + i + 1];
                    z[size_t(k) * n + i + 1] = s * z[size_t(k) * n + i] + c * f;
                    z[size_t(k) * n + i] = c * z[size_t(k) * n + i] - s * f;
                }
            }
            if (r == 0 && i >= l) continue;
            d[size_t(l)] -= p; e[size_t(l)] = g; e[size_t(m)] = 0;
        }
    }
}
