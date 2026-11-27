// Beams from openPMD particle files (species key `file`, docs/specs/structure3d.md, decision 102; port of GEM-PIC-RZ
// src/beam_import.cpp). Reads /data/<iteration>/<particlesPath>/<species>/: position + positionOffset, momentum,
// weighting (datasets or constant records, times unitSI; momentum macroWeighted / weightingPower), charge and mass
// (electrons or positrons only). HDF5 builds only.
//
// Placement in the run (the file's z is the propagation axis, the code's x):
//   X = x_center + (z - <z>_Q)     (the charge centroid of the file at x_center: the time of the file does not matter)
//   Y = y_center + x + y_offset,  Z = z_center + y + z_offset   (y_center, z_center: the axis, default the box centre)
//   (u_x, u_y, u_z) = (u_z, u_x, u_y) of the file, u = p / (m_e c)
// Optional: `charge` rescales the weights to a total charge, `macroparticles` resamples systematically (probability ~
// charge, equal weights) with an optional Gaussian jitter `resample_smoothing` x the rms of every coordinate.
#pragma once
#include <string>
#include <vector>

// A bunch without a file (block `bunch <name>`, docs/specs/deck_standard.md): Gaussian in all three coordinates, at
// its waist at t = 0, uncorrelated: transverse momenta of rms eps_n / sigma, gamma = gamma0 (1 + de N(0, 1)), equal
// weights. Electrons or positrons.
struct BunchShape {
    bool on = false;
    double sign = -1;                    // -1 electrons, +1 positrons
    double charge = 0;                   // total charge [C] (> 0)
    double gamma = 0, de = 0;            // mean Lorentz factor, relative rms energy spread
    double sigma[3] = {0, 0, 0};         // rms sizes along the run's x (propagation), y, z (code units)
    double eps[2] = {0, 0};              // normalized rms emittances in y and z (code units)
    long n = 100000;                     // macroparticles
};

struct BeamFileConfig {
    BunchShape gen;                      // on: the particles are generated (BunchSource), `path` only names the source
    std::string path;                    // empty: no file
    std::string species;                 // file_species ("" = the only species with particles)
    long iteration = -1;                 // file_iteration (-1: the last)
    double x_center = 0;                 // code units, absolute (lab) x of the charge centroid
    double y_center = 0, z_center = 0;   // the file's transverse origin (code units)
    double y_offset = 0, z_offset = 0;   // extra transverse shift (code units)
    double charge = 0;                   // > 0: rescale to this total charge [C]
    long macroparticles = 0;             // > 0: resample to this number
    double smoothing = 0;                // resample_smoothing
};

struct BeamParticles {                   // in code units, the axes of the run; w in the code's weight (n_c / k0^3)
    std::vector<double> x, y, z, ux, uy, uz, w;
    double sign = -1;                    // charge of the species read (-1 electrons, +1 positrons)
    std::string format, species;
    long iteration = -1;
    size_t size() const { return x.size(); }
};

// The whole beam of the file at once (tests, small files), placed and converted (lambda0: the run's wavelength [m], sets 1/k0 and n_c). Throws
// std::runtime_error naming the file. seed: the resampling's random numbers (identical on every rank).
BeamParticles import_beam(const BeamFileConfig& c, double lambda0, unsigned long seed);

// The same beam in portions (the run: no rank holds the whole beam). The file is read CHUNK = 2^18 particles at a
// time; the particles and their order are those of import_beam to the last bit.
class BeamFileSource {
public:
    BeamFileSource(const BeamFileConfig& c, double lambda0, unsigned long seed);
    ~BeamFileSource();
    BeamFileSource(const BeamFileSource&) = delete;
    BeamFileSource& operator=(const BeamFileSource&) = delete;
    long left() const;
    BeamParticles next(long count);      // the next min(count, left()) particles
    double gamma_mean() const;           // weighted mean Lorentz factor of the beam
    double sign = -1;
    std::string format, species;
    long iteration = -1;
private:
    struct Impl;
    Impl* p_ = nullptr;
};

// The particles of a `bunch` block, in portions: every rank draws the same sequence (seed) and keeps what lies in
// its part of the box, so no rank holds the whole bunch. Placement as for a file: the centroid at x_center, the
// transverse origin at (y_center + y_offset, z_center + z_offset).
class BunchSource {
public:
    BunchSource(const BeamFileConfig& c, double lambda0, unsigned long seed);
    ~BunchSource();
    long left() const { return left_; }
    BeamParticles next(long count);      // the next min(count, left()) particles
private:
    struct Rng;
    Rng* rng_;
    BeamFileConfig c_;
    double w_ = 0;                       // weight of a macroparticle (code units)
    long left_ = 0;
};

// Moments of a beam in portions (one pass; sums about the given centre and gamma, which for a bunch block are its
// nominal values, so the small spreads do not drown in the round-off of the large means)
class MomentSums {
public:
    MomentSums(const double centre[3], double gamma0) : g0_(gamma0) { for (int a = 0; a < 3; ++a) c_[a] = centre[a]; }
    void add(const BeamParticles& b);
    struct FileMoments result(double lambda0) const;
    double mean_ux() const { return W_ > 0 ? ux_ / W_ : 0.0; }
private:
    double c_[3], g0_;
    double W_ = 0, g1_ = 0, g2_ = 0, r1_[3] = {0, 0, 0}, r2_[3] = {0, 0, 0}, u1_[2] = {0, 0}, u2_[2] = {0, 0}, ru_[2] = {0, 0}, ux_ = 0;
    long n_ = 0;
};

// Moments of a beam (code units in, SI out): Q [C] (|charge|), the mean kinetic energy [MeV], rms dE/E, the charge
// centroid and rms sizes [m] along the run's x, y, z, the normalized rms emittances in y and z [m]
struct FileMoments { double Q = 0, ekin_mev = 0, de_rel = 0, mean[3] = {0, 0, 0}, sigma[3] = {0, 0, 0}, eps_n[2] = {0, 0}; long n = 0; };
FileMoments file_moments(const BeamParticles& b, double lambda0);
