// Laser from a LASY / openPMD envelope file (docs/specs/laser_file.md, decision 35), injected through
// the open left x boundary like the built-in lasers (laser.hpp).
//
// File (LASY, https://github.com/LASY-org/lasy): the mesh `laserEnvelope` of one iteration, the complex
// envelope of the transverse electric field at a plane, versus time t and (x, y) or (r, theta):
//   E_{x,y} = Re[ p_{x,y} env e^{-i omega_L t} ],  attributes angularFrequency, polarization (p_x, p_y),
//   envelopeField = electric_field [V/m] or normalized_vector_potential;
//   geometry cartesian (axes t, y, x) or thetaMode (components m = 0, cos, sin; axes t, r).
// VLPL propagates along x: LASY's x is our y, LASY's y our z (p_x -> E_y, p_y -> E_z), centred at
// (y_center, z_center). An axis of length 1 in a cartesian file makes the pulse uniform along it.
//
// Time: the file plane is x = file_x (absolute). The file's time t_center passes it at the simulation
// time delay + file_x. At the entrance plane x = 0 the field at time t is the file's envelope at the
// retarded time u = t - delay + t_center, propagated from x = file_x to 0 (exact for every plane wave in
// vacuum, phase e^{i (k_x - omega)(0 - file_x)} at fixed u; FFT in u, y, z).
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "comm.hpp"
#include "fields.hpp"
#include "grid.hpp"

struct LaserFileConfig {
    std::string name, path, mesh = "laserEnvelope";
    long iteration = -1;           // < 0: the last
    double delay = -1;             // simulation time (1/omega0) at which t_center passes the file plane - file_x;
                                   // < 0: the file's first time enters at t = 0
    double t_center = 1e300;       // file time put at delay (1/omega0, the file's time base); 1e300: intensity centroid
    double file_x = 0;             // x of the file's plane (1/k0, absolute); 0: the entrance plane
    double yc = 0, zc = 0;         // where the file's transverse origin sits (1/k0)
    double a0 = 0;                 // > 0: rescale so that the peak |env| = a0 omega_L
    bool flat = false;             // geometry = cartesian2d: a slab file (LASY y of length 1) only
    bool antenna = false;          // a boosted run: emitted by the lab-fixed antenna (boost_antenna.cpp)
};

// The envelope at the entrance plane on a regular (u, y, z) grid, in device memory; evaluated by the
// boundary hook. Built on the ranks at the open left x boundary only (the others hold nothing).
struct FileLaserView {             // trivially copyable, used inside kernels
    const double* env = nullptr;   // (re, im) at [((iz * ny) + iy) * nu + iu]
    int nu = 0, ny = 0, nz = 0;
    double u0 = 0, du = 1, y0 = 0, dy = 1, z0 = 0, dz = 1;   // u0, y0, z0: the first node (absolute y, z)
    double omega = 1, u_shift = 0;   // carrier; u = t - delay + t_center = t + u_shift
    double pyr = 1, pyi = 0, pzr = 0, pzi = 0;   // polarization (E_y, E_z), complex
    bool uniform_y = false, uniform_z = false;
    // E_y (comp 0) or E_z (comp 1) at time t and (y, z) of the entrance plane
    HD double field(int comp, double t, double y, double z) const {
        const double u = t + u_shift;
        double fu = (u - u0) / du, fy = uniform_y ? 0 : (y - y0) / dy, fz = uniform_z ? 0 : (z - z0) / dz;
        if (!(fu >= 0 && fu <= nu - 1 && fy >= 0 && fy <= ny - 1 && fz >= 0 && fz <= nz - 1)) return 0;
        int a = int(fu), b = int(fy), c = int(fz);
        a = a > nu - 2 ? (nu > 1 ? nu - 2 : 0) : a;
        b = b > ny - 2 ? (ny > 1 ? ny - 2 : 0) : b;
        c = c > nz - 2 ? (nz > 1 ? nz - 2 : 0) : c;
        fu -= a; fy -= b; fz -= c;
        double re = 0, im = 0;
        for (int q = 0; q < 8; ++q) {
            const int da = q & 1, db = (q >> 1) & 1, dc = q >> 2;
            if ((da && nu == 1) || (db && ny == 1) || (dc && nz == 1)) continue;
            const double w = (da ? fu : 1 - fu) * (db ? fy : 1 - fy) * (dc ? fz : 1 - fz);
            const long i = 2 * ((long(c + dc) * ny + (b + db)) * nu + (a + da));
            re += w * env[i]; im += w * env[i + 1];
        }
        // Re[p env e^{-i omega u}]
        const double cr = cos(omega * u), ci = -sin(omega * u);
        const double er = re * cr - im * ci, ei = re * ci + im * cr;
        return comp == 0 ? pyr * er - pyi * ei : pzr * er - pzi * ei;
    }
};

class FileLaser {
public:
    // Reads the file and prepares the envelope at the entrance plane for the cells [y_lo, y_hi) x
    // [z_lo, z_hi) of this rank (absolute coordinates); info: lines for the run summary.
    FileLaser(const LaserFileConfig& c, double lambda0, double y_lo, double y_hi, double z_lo, double z_hi);
    FileLaserView view() const { FileLaserView v = v_; v.env = env_.data(); return v; }      // device memory (kernels)
    FileLaserView host_view() const { FileLaserView v = v_; v.env = host_.data(); return v; }  // host memory
    std::vector<std::string> info;
    double t_first = 0, t_last = 0;  // simulation times at which the file's time span passes x = 0
    double edge = 0;                 // largest |env| on the edge of the grid at x = 0, relative to the peak
private:
    FileLaserView v_;
    Buf<double> env_;
    std::vector<double> host_;       // the same envelope on the host (tests, diagnostics)
};

// RZ (geometry = rz, laser_file_rz.cpp): the file as azimuthal modes at the entrance plane z = 0 (the lab plane of the boosted
// antenna), env(u, r, phi) = sum_{k = -K .. K} a_k(u, r) e^{-i k phi} with phi of the particles (LASY's x is the particles' x,
// as the built-in polarization y), on the nodes r_j = j dr; a thetaMode file is used mode by mode, a cartesian one resampled on
// polar nodes around (y_center, z_center) = its origin; file_x != 0: exact propagation per mode (Hankel transform of order |k|
// and FFT in u). Only the modes k < the simulation's modes are kept.
struct RzFileLaserView {           // trivially copyable, used inside kernels
    const double* env = nullptr;   // (re, im) at [((k + K) * nr + j) * nu + l]
    int nu = 0, nr = 0, K = 0;
    double u0 = 0, du = 1, dr = 1;
    double omega = 1, u_shift = 0;   // carrier; u = t + u_shift
    double pxr = 1, pxi = 0, pyr = 0, pyi = 0;   // polarization (E_x, E_y of the particles), complex
    // E_x and E_y (particle axes) at time t, radius r, angle phi (c = cos phi, s = sin phi)
    HD void field(double t, double r, double c, double s, double& ex, double& ey) const {
        ex = ey = 0;
        const double u = t + u_shift;
        double fu = (u - u0) / du, fr = r / dr;
        if (!(fu >= 0 && fu <= nu - 1 && fr >= 0 && fr <= nr - 1)) return;
        int a = int(fu), b = int(fr);
        a = a > nu - 2 ? nu - 2 : a;
        b = b > nr - 2 ? nr - 2 : b;
        fu -= a; fr -= b;
        double re = 0, im = 0, er = 1, ei = 0;   // (er + i ei) = e^{-i k phi} = (c - i s)^k
        for (int k = 0; k <= K; ++k) {
            for (int sg = (k ? -1 : 1); sg <= 1; sg += 2) {   // k > 0: the modes -k (conjugate factor) and +k
                const int kk = sg * k;
                const double fr_ = er, fi = sg > 0 ? ei : -ei;
                double vr = 0, vi = 0;
                for (int q = 0; q < 4; ++q) {
                    const int da = q & 1, db = q >> 1;
                    const double w = (da ? fu : 1 - fu) * (db ? fr : 1 - fr);
                    const long i = 2 * ((long(kk + K) * nr + (b + db)) * nu + (a + da));
                    vr += w * env[i]; vi += w * env[i + 1];
                }
                re += vr * fr_ - vi * fi; im += vr * fi + vi * fr_;
            }
            const double nr_ = er * c + ei * s, ni_ = ei * c - er * s;   // times (c - i s)
            er = nr_; ei = ni_;
        }
        // Re[p env e^{-i omega u}]
        const double cr = cos(omega * u), ci = -sin(omega * u);
        const double vr = re * cr - im * ci, vi = re * ci + im * cr;
        ex = pxr * vr - pxi * vi;
        ey = pyr * vr - pyi * vi;
    }
};

class RzFileLaser {
public:
    // r_need: the largest radius the run evaluates (the radial grid's edge)
    RzFileLaser(const LaserFileConfig& c, double lambda0, int modes, double r_need);
    RzFileLaserView view() const { RzFileLaserView v = v_; v.env = env_.data(); return v; }
    RzFileLaserView host_view() const { RzFileLaserView v = v_; v.env = host_.data(); return v; }
    std::vector<std::string> info;
    double t_first = 0, t_last = 0;  // simulation (lab) times at which the file's time span passes z = 0
    double edge = 0;                 // largest |a_k| on the edge of the grid at z = 0, relative to the peak
    double peak = 0;                 // peak |env| (code units)
private:
    RzFileLaserView v_;
    Buf<double> env_;
    std::vector<double> host_;
};
