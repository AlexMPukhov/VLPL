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

struct BeamFileConfig {
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

// The beam of the file, placed and converted (lambda0: the run's wavelength [m], sets 1/k0 and n_c). Throws
// std::runtime_error naming the file. seed: the resampling's random numbers (identical on every rank).
BeamParticles import_beam(const BeamFileConfig& c, double lambda0, unsigned long seed);

// Moments of a beam (code units in, SI out): Q [C] (|charge|), the mean kinetic energy [MeV], rms dE/E, the charge
// centroid and rms sizes [m] along the run's x, y, z, the normalized rms emittances in y and z [m]
struct FileMoments { double Q = 0, ekin_mev = 0, de_rel = 0, mean[3] = {0, 0, 0}, sigma[3] = {0, 0, 0}, eps_n[2] = {0, 0}; long n = 0; };
FileMoments file_moments(const BeamParticles& b, double lambda0);
