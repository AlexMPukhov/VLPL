// Beam diagnostics (docs/specs/beam_diagnostics.md): for the particles of one species with
// gamma >= gamma_min, every `every` steps: charge, energy moments, rms sizes, normalised emittances,
// the energy spectrum dQ/dE and optionally the particles themselves. Deck block `diagnostic <name>`,
// operator `beam_diag` of stage Diagnostics. All sums run on the device; centred second moments
// (two passes).
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "comm.hpp"
#include "deck.hpp"
#include "pipeline.hpp"
#include "species.hpp"

struct BeamDiagConfig {
    std::string name, species;
    long every = 10;               // steps
    double gamma_min = 1;          // select gamma >= gamma_min
    int bins = 200;                // spectrum bins on [0, e_max] (0: no spectrum)
    double e_max_mev = 1000;
    bool particles = false;        // write the selected particles
};
std::vector<BeamDiagConfig> beam_diags_from_deck(const Deck& d, long default_every);

// Moments of the selected particles, code units (lengths 1/k0, u = gamma v / c), summed over all ranks.
struct BeamMoments {
    double count = 0, wsum = 0;    // macro-particles, sum of weights (n_c / k0^3)
    double gmean = 0, gsig = 0, gmax = 0;                 // weighted mean and rms of gamma, maximum
    double xmean = 0, sx = 0, sy = 0, sz = 0;             // mean x, rms sizes
    double eps_y = 0, eps_z = 0;                          // normalised rms emittances
};
BeamMoments beam_moments(const Species& s, double gamma_min, Comm& comm);
// Sum of weights per bin of the kinetic energy (gamma - 1) m (units m_e c^2) on [0, ek_max), all ranks.
std::vector<double> beam_spectrum(const Species& s, double gamma_min, int bins, double ek_max, Comm& comm);
// The selected particles of this rank in index order: 7 values (x y z ux uy uz w) per particle.
std::vector<double> beam_particles(const Species& s, double gamma_min);

std::unique_ptr<Operator> make_beam_diag_operator(const BeamDiagConfig& c);
