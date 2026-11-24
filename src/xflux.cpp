#include "xflux.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include "fsutil.hpp"
#include "rz.hpp"
#include "sim.hpp"

// ---------------------------------------------------------------- kernels (free functions)
// Cart3D, the plane i0: out[q], out[n + q] (q = j + ny k) = the outbound T of the (Ey, Bz) and (Ez, By)
// pairs; s = +1 at the left edge (Ey - Bz, Ez + By: along -x), -1 at the right edge (Ey + Bz, Ez - By).
// Returns sum over the plane of T_y^2 + T_z^2; *in = the same for the inbound components in the cell beyond the plane
// (i0 - s), which move into i0 during the next step (at an edge: the guard, e.g. a laser entering there).
double xflux_plane_3d(const FV Ey, const FV Ez, const FV By, const FV Bz, int i0, int ny, int nz, double s, double* out,
                      double* in) {
    const long n = long(ny) * nz;
    par_for(n, [=] HD(long q) {
        const int j = int(q % ny), k = int(q / ny);
        out[q] = Ey(i0, j, k) - s * Bz(i0, j, k);
        out[n + q] = Ez(i0, j, k) + s * By(i0, j, k);
    });
    const int ib = i0 - int(s);
    *in = par_sum(n, [=] HD(long q) {
        const int j = int(q % ny), k = int(q / ny);
        const double a = Ey(ib, j, k) + s * Bz(ib, j, k), b = Ez(ib, j, k) - s * By(ib, j, k);
        return a * a + b * b;
    });
    return par_sum(n, [=] HD(long q) { return out[q] * out[q] + out[n + q] * out[n + q]; });
}

// RZ, the plane i0 (z index), per radial index j and mode slot k (q = j + nr k): out[q] = E_r -+ B_phi,
// out[n + q] = E_phi +- B_r (upper sign: left edge, s = +1). Returns the energy weight sum
// w_m (B_j T_r^2 + A_j T_phi^2) with w_0 = 1, w_m = 2 (as FieldsRZ::energy: E_r, B_phi at the half nodes,
// E_phi, B_r at the nodes).
double xflux_plane_rz(const FV Er, const FV Ep, const FV Br, const FV Bp, const double* A, const double* B,
                      int i0, int nr, int S, double s, double* out, double* in) {
    const long n = long(nr) * S;
    par_for(n, [=] HD(long q) {
        const int j = int(q % nr), k = int(q / nr);
        out[q] = Er(i0, j, k) - s * Bp(i0, j, k);
        out[n + q] = Ep(i0, j, k) + s * Br(i0, j, k);
    });
    const int ib = i0 - int(s);
    *in = par_sum(n, [=] HD(long q) {
        const int j = int(q % nr), k = int(q / nr);
        const double wm = k == 0 ? 1.0 : 2.0;
        const double a = Er(ib, j, k) + s * Bp(ib, j, k), b = Ep(ib, j, k) - s * Br(ib, j, k);
        return wm * (B[j] * a * a + A[j] * b * b);
    });
    return par_sum(n, [=] HD(long q) {
        const int j = int(q % nr), k = int(q / nr);
        const double wm = k == 0 ? 1.0 : 2.0;
        return wm * (B[j] * out[q] * out[q] + A[j] * out[n + q] * out[n + q]);
    });
}

// Cart3D: (1/2) sum (E^2 + B^2) dV over the owned cells with the local x index in [i0, i1] (the cells between the
// measuring planes, without the x absorbers), as Fields::energy
double xflux_energy_between(const FV a, const FV b, const FV c, const FV e, const FV f, const FV g, int i0, int i1, int nx,
                            int ny, long n, double vol) {
    return 0.5 * vol * par_sum(n, [=] HD(long q) {
        const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
        if (i < i0 || i > i1) return 0.0;
        auto sq = [&](const FV& v) { return v(i, j, k) * v(i, j, k); };
        return sq(a) + sq(b) + sq(c) + sq(e) + sq(f) + sq(g);
    });
}

// RZ: FieldsRZ::energy over the owned z cells with the local index in [i0, i1] (a = Ez, c = Ephi, f = Br at the nodes,
// b = Er, e = Bz, g = Bphi at the half nodes)
double xflux_energy_between_rz(const FV a, const FV b, const FV c, const FV e, const FV f, const FV g, const double* A,
                               const double* B, int i0, int i1, int nz, int nr, long n, double dz) {
    return 0.5 * dz * par_sum(n, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), k = int(q / (long(nz) * nr));
        if (i < i0 || i > i1) return 0.0;
        const double wm = k == 0 ? 1.0 : 2.0, vn = 2 * M_PI * A[j], vh = 2 * M_PI * B[j];
        auto sq = [&](const FV& v) { return v(i, j, k) * v(i, j, k); };
        return wm * (vn * (sq(a) + sq(c) + sq(f)) + vh * (sq(b) + sq(e) + sq(g)));
    });
}

namespace {
class XFlux : public Operator {
public:
    const char* name() const override { return "xflux"; }
    Stage stage() const override { return Stage::Fields; }    // after the field step, before a window shift

    void setup(Simulation& s) override {
        if (s.cfg.window.enabled)
            throw std::runtime_error("output.xflux: not with a moving window (a wave along -x crosses two cells per step "
                                     "relative to the grid; docs/specs/xflux.md)");
        const Domain& d = s.dom;
        // the measuring cells: the edge cells, or with x (RZ: z) absorbers (O3) the first cells inside them
        const int L = s.cfg.rz ? s.cfg.x_abs_cells : s.fields->xabs.cells;
        L_ = L;
        const int g0[2] = {L, d.N[0] - 1 - L};
        for (int k = 0; k < 2; ++k) {
            i0_[k] = g0[k] - d.off[0];
            edge_[k] = i0_[k] >= 0 && i0_[k] < d.n[0];
        }
        np_ = s.cfg.rz ? long(d.n[1]) * d.n[2] : long(d.n[1]) * d.n[2];
        buf_.resize(size_t(2 * np_));
        host_.resize(size_t(2 * np_));
        // energy per step = c * sum T^2 / 4 * (cell volume): Cart3D h_x h_y h_z; RZ dz 2 pi (A_j or B_j)
        unit_ = s.cfg.rz ? 0.25 * d.h[0] * 2 * M_PI : 0.25 * d.cell_volume();
        // The laser energy injected through the open left edge: the inbound component the inflow hooks wrote into the guard
        // layer, measured by one more hook right after them (before the transport moves it into cell 0; after the step the
        // guard no longer holds it). It is the inbound energy of the left plane; with an RZ TFSF plane the hook sits on the
        // incident grid instead, and its count is only the E_inj column (the laser energy, for the balance).
        if (d.off[0] == 0 && !d.periodic[0]) {
            hbuf_.resize(size_t(2 * np_));
            double* acc = &inj_;
            double* hb = hbuf_.data();
            const double unit = unit_;
            const int n1 = d.n[1], n2 = d.n[2];
            if (s.cfg.rz) {
                FieldsRZ* G = s.rz_tfsf() ? &s.rz_tfsf()->incident() : (L == 0 ? s.rz : nullptr);
                if (G)
                    G->inflow.push_back([acc, hb, unit, n1, n2](FieldsRZ& F, double) {
                        const RadialView g = F.grid().view();
                        double in = 0;
                        xflux_plane_rz(F.er.view(), F.ep.view(), F.br.view(), F.bp.view(), g.A, g.B, 0, n1, n2, 1.0, hb, &in);
                        *acc += in * unit;
                    });
            } else if (L == 0) {
                s.hooks.inflow.push_back([acc, hb, unit, n1, n2](Fields& F, double) {
                    double in = 0;
                    xflux_plane_3d(F.ey.view(), F.ez.view(), F.by.view(), F.bz.view(), 0, n1, n2, 1.0, hb, &in);
                    *acc += in * unit;
                });
            }
            edge_inj_ = L == 0 && !(s.cfg.rz && s.rz_tfsf());
        }
        if (s.comm.rank == 0) {
            make_dirs(s.cfg.dir);
            bool fresh;
            log_ = open_log(s.cfg.dir + "/xflux.txt", s.restart_step, fresh);
            if (fresh) std::fprintf(log_, "# step  t  P_left  P_right  E_left  E_right  Enet_left  Enet_right  E_abs_r  W_in  E_part_out  E_inj   (P, E: the energy of "
                                          "the outbound component through the left / right measuring plane per unit time and its sum "
                                          "since the start; Enet: outbound - inbound, the Poynting flux, exact at any angle; the "
                                          "planes are the x edges or the inner faces of the x absorbers; E_abs_r: the energy removed by the radial absorber of RZ; W_in: the field energy between the planes; E_part_out: the kinetic energy of the particles that left the box; W_in + W_kin + Enet_left + Enet_right + E_abs_r + E_part_out is the energy balance; E_inj: the laser energy injected through the open left edge, which Enet_left counts as inbound, or with an RZ TFSF plane into its incident grid)\n");
            if (s.restart_step >= 0)
                std::fprintf(stderr, "warning: output.xflux: E_left, E_right count from the restart (step %ld)\n", s.restart_step);
        }
        if (s.cfg.xflux.every > 0 && (edge_[0] || edge_[1])) {
            const std::string dir = s.cfg.dir + "/xflux";
            make_dirs(dir);
            const std::string r = std::to_string(s.comm.rank);
            const char* mode = s.restart_step >= 0 ? "a" : "w";
            for (int e = 0; e < 2; ++e)
                if (edge_[e]) {
                    const std::string fn = dir + (e == 0 ? "/left_r" : "/right_r") + r + ".bin";
                    bin_[e] = std::fopen(fn.c_str(), (std::string(mode) + "b").c_str());
                    if (!bin_[e]) throw std::runtime_error("cannot open " + fn);
                }
            FILE* m = std::fopen((dir + "/meta_r" + r + ".txt").c_str(), "w");
            if (!m) throw std::runtime_error("cannot open " + dir + "/meta_r" + r + ".txt");
            std::fprintf(m, "geometry %s\nevery %ld\ndt %.17g\nh %.17g %.17g %.17g\n", s.cfg.rz ? "rz" : "cart3d",
                         s.cfg.xflux.every, d.dt(), d.h[0], d.h[1], d.h[2]);
            std::fprintf(m, "n %d %d %d\noff %d %d %d\nN %d %d %d\n", d.n[0], d.n[1], d.n[2], d.off[0], d.off[1], d.off[2],
                         s.cfg.N[0], s.cfg.N[1], s.cfg.N[2]);
            std::fprintf(m, "left %d\nright %d\nlambda0 %.17g\nplanes %d %d\n", int(edge_[0]), int(edge_[1]), s.cfg.lambda0,
                         g0[0], g0[1]);
            std::fprintf(m, "# a record: float64 [2][%s] = the outbound components (left: Ey - Bz, Ez + By / E_r - B_phi, "
                            "E_phi + B_r; right: Ey + Bz, Ez - By / E_r + B_phi, E_phi - B_r) after the field step of step "
                            "every * (record + 1)\n", s.cfg.rz ? "slots][nr" : "nz][ny");
            if (s.cfg.rz) {                // radii and the energy weights A_j (nodes), B_j (half nodes) of FieldsRZ::energy
                const RadialGrid& g = s.rz->grid();
                const std::vector<double>* v[4] = {&g.r, &g.rh, &g.A, &g.B};
                const char* nm[4] = {"r_nodes", "r_half", "area_nodes", "area_half"};
                for (int q = 0; q < 4; ++q) {
                    std::fprintf(m, "%s", nm[q]);
                    for (int j = 0; j < d.n[1]; ++j) std::fprintf(m, " %.17g", (*v[q])[size_t(j)]);
                    std::fprintf(m, "\n");
                }
            }
            std::fclose(m);
        }
        measure(s, false, prev_, prevn_);
    }

    // The outbound component in the measuring cell at t_n leaves it during the step n -> n+1 (and the inbound one of
    // the cell beyond enters): the energy of step n+1 is the one measured after step n (prev_), measured first in setup.
    void apply(Simulation& s) override {
        const bool rec = s.cfg.xflux.every > 0 && (s.istep + 1) % s.cfg.xflux.every == 0;
        double e[2], en[2];
        measure(s, rec, e, en);
        const double out[2] = {prev_[0], prev_[1]};
        for (int k = 0; k < 2; ++k) {
            sum_[k] += prev_[k]; net_[k] += prevn_[k];
            prev_[k] = e[k]; prevn_[k] = en[k];
        }
        // the laser injected through the open left edge during this step (it is in the box now): inbound at the left plane
        double injs = edge_inj_ ? inj_ - inj_used_ : 0.0;
        inj_used_ = inj_;
        s.comm.allreduce_sum(&injs, 1);
        net_[0] -= injs;
        const long step = s.istep + 1;
        double v2[4] = {s.absorbed_r, 0, 0, inj_};
        for (const Species& sp : s.species) v2[2] += sp.lost_energy;
        if (step % s.cfg.every == 0) {
            const Domain& dd = s.dom;
            if (s.cfg.rz) {
                const FieldsRZ& F = *s.rz;
                const RadialView g = F.grid().view();
                v2[1] = xflux_energy_between_rz(F.ez.view(), F.er.view(), F.ep.view(), F.bz.view(), F.br.view(), F.bp.view(), g.A,
                                                g.B, L_ - dd.off[0], dd.N[0] - 1 - L_ - dd.off[0], dd.n[0], dd.n[1], dd.owned(),
                                                dd.h[0]);
            } else {
                const Fields& F = *s.fields;
                const Domain& d = s.dom;
                const int L = F.xabs.cells;
                v2[1] = xflux_energy_between(F.ex.view(), F.ey.view(), F.ez.view(), F.bx.view(), F.by.view(), F.bz.view(),
                                             L - d.off[0], d.N[0] - 1 - L - d.off[0], d.n[0], d.n[1], d.owned(), d.cell_volume());
            }
            s.comm.allreduce_sum(v2, 4);
        }
        const double absorbed = v2[0], win = v2[1], lost = v2[2], inj = v2[3];
        if (log_ && step % s.cfg.every == 0) {
            const double dt = s.dom.dt();
            std::fprintf(log_, "%ld %.6e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", step, step * dt,
                         out[0] / dt, out[1] / dt, sum_[0], sum_[1], net_[0], net_[1], absorbed, win, lost, inj);
            std::fflush(log_);
        }
    }

    ~XFlux() override {
        if (log_) std::fclose(log_);
        for (FILE* f : bin_) if (f) std::fclose(f);
    }

private:
    // e[k]: the energy of the outbound component in the measuring cell k (0 left, 1 right), en[k] = e[k] minus that of
    // the inbound component entering it, over all ranks; with rec the components go to the record file of the plane
    void measure(Simulation& s, bool rec, double e[2], double en[2]) {
        const Domain& d = s.dom;
        double v[4] = {0, 0, 0, 0};
        for (int k = 0; k < 2; ++k) {
            if (!edge_[k]) continue;
            const double sg = k == 0 ? 1.0 : -1.0;
            double in = 0;
            if (s.cfg.rz) {
                FieldsRZ& F = *s.rz;
                const RadialView g = F.grid().view();
                v[k] = xflux_plane_rz(F.er.view(), F.ep.view(), F.br.view(), F.bp.view(), g.A, g.B, i0_[k], d.n[1], d.n[2], sg,
                                      buf_.data(), &in);
            } else {
                Fields& F = *s.fields;
                v[k] = xflux_plane_3d(F.ey.view(), F.ez.view(), F.by.view(), F.bz.view(), i0_[k], d.n[1], d.n[2], sg,
                                      buf_.data(), &in);
            }
            v[k] *= unit_;
            // the open left edge with the laser: its injection is counted in the step it entered (inj, apply), not here
            v[2 + k] = k == 0 && edge_inj_ ? v[k] : v[k] - in * unit_;
            if (rec && bin_[k]) {
                buf_.to_host(host_.data(), 0, size_t(2 * np_));
                std::fwrite(host_.data(), sizeof(double), host_.size(), bin_[k]);
                std::fflush(bin_[k]);
            }
        }
        s.comm.allreduce_sum(v, 4);
        e[0] = v[0]; e[1] = v[1]; en[0] = v[2]; en[1] = v[3];
    }

    bool edge_[2] = {false, false};                // this rank owns the measuring cell
    int i0_[2] = {0, 0};                           // its local index
    int L_ = 0;                                    // cells of the x (RZ: z) absorbers
    bool edge_inj_ = false;                        // the left plane is the open edge with the injected laser (this rank)
    double inj_ = 0, inj_used_ = 0;                // laser energy injected (the hook), and the part counted in Enet_left
    Buf<double> hbuf_;                             // scratch of the injection hook
    long np_ = 0;
    double unit_ = 0, sum_[2] = {0, 0}, prev_[2] = {0, 0}, net_[2] = {0, 0}, prevn_[2] = {0, 0};
    Buf<double> buf_;
    std::vector<double> host_;
    FILE* log_ = nullptr;
    FILE* bin_[2] = {nullptr, nullptr};
};
}  // namespace

std::unique_ptr<Operator> make_xflux_operator() { return std::make_unique<XFlux>(); }
