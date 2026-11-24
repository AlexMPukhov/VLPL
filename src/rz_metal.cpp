// Embedded perfect conductors in RZ (block `rings`, docs/specs/rz_metal.md): snapping the rings onto the grid and the
// removal of particles inside the metal. The field rules are in FieldsRZ::pass (src/rz_fields.cpp).
#include "rz_metal.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "rz.hpp"
#include "species.hpp"

namespace {
int nearest_node(const RadialGrid& g, double r) {
    int best = 0;
    for (int j = 1; j <= g.nr; ++j)
        if (std::abs(g.r[size_t(j)] - r) < std::abs(g.r[size_t(best)] - r)) best = j;
    return best;
}
}  // namespace

std::shared_ptr<MetalRZ> rz_metal_setup(const RingsConfig& c, int nz, double dz, bool periodic, const RadialGrid& g) {
    if (!(c.r_out > c.r_in) || c.r_in < 0) throw std::runtime_error("rings: 0 <= r_in < r_out");
    if (!(c.z_end > c.z_start)) throw std::runtime_error("rings: z_start < z_end");
    if (!(c.period > 0) || c.gap < 0 || c.gap >= c.period) throw std::runtime_error("rings: period > 0 and 0 <= gap < period");
    auto m = std::make_shared<MetalRZ>();
    m->cfg = c;
    m->J = nearest_node(g, c.r_in);
    m->K = c.r_out >= g.R() ? g.nr : nearest_node(g, c.r_out);
    if (m->K <= m->J) throw std::runtime_error("rings: r_out - r_in is less than one radial cell after snapping to the nodes");
    m->N = periodic ? nz : 0;
    const long iend = periodic ? std::min<long>(std::lround(c.z_end / dz), nz - 1) : std::lround(c.z_end / dz);
    for (long k = 0;; ++k) {
        const double za = c.z_start + double(k) * c.period;
        if (za >= c.z_end - 1e-12 * c.period) break;
        const double zb = c.gap == 0 ? c.z_end : std::min(za + c.period - c.gap, c.z_end);
        const long a = std::lround(za / dz), b = std::min(std::lround(zb / dz), iend);
        if (b <= a) throw std::runtime_error("rings: a ring thinner than one z cell after snapping (dz too coarse)");
        if (!m->ia.empty() && a <= m->ib.back() + 1) { m->ib.back() = std::max(m->ib.back(), b); }   // no vacuum column between
        else { m->ia.push_back(a); m->ib.push_back(b); }
        if (c.gap == 0) break;
    }
    // periodic z: inside [0, nz); a ring touching both ends would join across the boundary (allowed only as one full tube)
    const bool full = m->ia.size() == 1 && m->ia[0] == 0 && m->ib[0] == nz - 1;
    if (periodic && (m->ia.front() < 0 || m->ib.back() > nz - 1 || (!full && m->ia.front() == 0 && m->ib.back() == nz - 1)))
        throw std::runtime_error("rings with periodic z: the rings must lie inside [0, nz) dz and not touch both ends");
    if (c.jacket > 0) {                // the jacket: a solid tube r_out < r < r_out + jacket over the z range of the stack
        if (m->K >= g.nr) throw std::runtime_error("rings: jacket > 0 needs r_out inside the grid");
        m->KJ = c.r_out + c.jacket >= g.R() ? g.nr : nearest_node(g, c.r_out + c.jacket);
        if (m->KJ <= m->K) throw std::runtime_error("rings: the jacket is thinner than one radial cell after snapping to the nodes");
        m->ta = std::lround(c.z_start / dz);
        m->tb = periodic ? std::min<long>(std::lround(c.z_end / dz), nz - 1) : std::lround(c.z_end / dz);
    }
    m->dia.upload(m->ia); m->dib.upload(m->ib);
    return m;
}

std::string MetalRZ::describe(double dz, const RadialGrid& g) const {
    char s[400];
    const long t0 = ib[0] - ia[0], g0 = ia.size() > 1 ? ia[1] - ib[0] : 0;
    std::snprintf(s, sizeof s,
                  "  rings: %zu, radial nodes %d..%d (r = %.6g .. %.6g), first ring cells %ld..%ld (metal %ld cells = %.6g, gap %ld "
                  "cells = %.6g), last face at z = %.6g (1/k0)\n",
                  ia.size(), J, K, g.r[size_t(J)], g.r[size_t(K)], ia[0], ib[0], t0, t0 * dz, g0, g0 * dz, ib.back() * dz);
    std::string out = s;
    if (tb >= ta) {
        std::snprintf(s, sizeof s, "  rings: jacket, radial nodes %d..%d (r = %.6g .. %.6g), cells %ld..%ld (open end at z = %.6g)\n",
                      K, KJ, g.r[size_t(K)], g.r[size_t(KJ)], ta, tb, tb * dz);
        out += s;
    }
    return out;
}

void rz_metal_remove(Species& sp, const FieldsRZ& f, double rwall, int order) {
    const MetalView mv = f.metal ? f.metal->view() : MetalView{};
    if (mv.n == 0 || sp.size() == 0) return;
    const RadialView g = f.grid().view();
    const double dz = f.dom().h[0], mg = 0.5 * (order - 1), out = 2 * rwall;
    const bool all = f.map_all || f.c1_map || f.modes() == 1;
    double* X = sp.ptr<double>(Species::X);
    double* Y = sp.ptr<double>(Species::Y);
    double* Z = sp.ptr<double>(Species::Z);
    par_for(long(sp.size()), [=] HD(long p) {
        // in the radial map of the deposit (decision 53): shape 1 touches the nodes floor(xi), floor(xi) + 1, shape 2 the
        // nodes round(xi) - 1 .. round(xi) + 1
        // (the modes m >= 1 deposit in the plain xi <= the mapped one, decision 106: the lower edge is tested with it)
        const double rp = sqrt(Y[p] * Y[p] + Z[p] * Z[p]), xi = rz_xi(g, rp, order), xp = all ? xi : rz_xi(g, rp, 0);
        const double zc = X[p] / dz;
        const long lo = long(floor(zc - mg)), hi = long(ceil(zc + mg));
        auto inside = [&](int J, int K, bool tube) {   // the shape entirely in the closed rectangle (rings or jacket)
            if (order == 1 ? (xp < J || xi > K) : (xp <= J + mg || xi >= K - mg)) return false;
            for (long I = lo; I <= hi; ++I)
                if (!(tube ? mv.tube(I) : mv.in(I))) return false;
            return true;
        };
        if (!inside(mv.J, mv.K, false) && !(mv.tb >= mv.ta && inside(mv.K, mv.KJ, true))) return;
        Y[p] = out; Z[p] = 0;                      // migrate() absorbs it like the wall
    });
}
