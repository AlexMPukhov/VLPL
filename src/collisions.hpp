// Binary Coulomb collisions (Phase 3; docs/specs/collisions.md, decision 94): relativistic Monte Carlo after Perez et al.,
// Phys. Plasmas 19, 083104 (2012), as in Smilei 5.1, with Nanbu's scattering-angle distribution. Device functions in
// collision_kernels.hpp.
//
// Every `every` steps, after the migration (all particles in the cells of their rank): per cell the macro-particles of the
// two species of a block are shuffled and paired (intra-species: the shuffled list in pairs, an odd one twice; inter-species:
// every particle of the larger group with one of the smaller, which repeat), and every pair scatters in its centre-of-mass
// frame by an angle drawn from Nanbu's distribution for the parameter s of Perez (relativistic Rutherford with the Coulomb
// logarithm, the low-temperature cap). Unequal weights: the particle of the larger weight is deflected with the probability
// w_small / w_large (Nanbu and Yonemura). s is normalized as in Smilei 5.1: dt every N max(w) / V per pair, with the weight
// correction of the repeated partners.
//
// Cells: Cart3D / 2D the grid cells (V = hx hy hz, 2D: the slab thickness dz); RZ the annuli between the radial nodes,
// V = pi (r_{j+1}^2 - r_j^2) dz (all azimuthal angles of a cell together). Particles in the entry columns of a moving window
// (x >= xfreeze) do not collide.
//
// Random numbers: the shuffle is the order of a hash of (seed, step, block, particle id) within the cell, the angles and the
// weight rule are hashes of (seed, step, block, absolute cell, pair): the same for any number of ranks or threads and any
// particle order (the ids are a hash of the load position, as for field ionization; new electrons of field ionization get
// theirs from the ion id and the level).
#pragma once
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

class Deck;
class Simulation;
class Operator;
struct Config;

struct CollisionConfig {
    std::string name;                  // the block name (collisions [<name>])
    std::string sp1, sp2;              // species; sp2 == sp1: collisions within one species
    double coulomb_log = 0;            // > 0: fixed; 0: automatic (Perez: the Debye length of the cell, b_min classical/quantum)
    long every = 1;                    // collide every `every` steps (with every dt)
    int angle = 0;                     // 0: nanbu, 1: smilei5 (the fit of Smilei 5.1)
    bool cap = true;                   // low_t_cap: Perez's low-temperature limit of s (mean free path >= interatomic distance)
    bool hquantum = false;             // quantum_bmin = h: the quantum b_min h / (2 p*) of Smilei 5.1 instead of hbar / (2 p*)
    // impact ionization (impact.hpp): one species electrons (q = -1, m = 1), the other ions with an element and a charge state
    bool ionizing = false;
    std::string electrons;             // species of the new electrons (default: the electron species of the block)
};

// blocks collisions [<name>]; an ionizing block gives its ion species a charge state per particle (ion.states)
std::vector<CollisionConfig> collisions_from_deck(const Deck& d, Config& c);
void collisions_summary(FILE* f, const Config& c);
// Registers the attribute id on the species that collide (before they are loaded).
void collisions_register(Simulation& s);
// The operators: ids of new particles (stage Inject) and the collisions (stage Migrate, after the migration and the sort);
// adds them to the pipeline (nothing without collisions blocks).
void collisions_add_operators(Simulation& s);
// One collision step now (all blocks, whatever the step; tests). Needs collisions_add_operators and the pipeline setup.
void collisions_apply(Simulation& s);
// dir/collisions.txt at the cadence of diag.txt (collective): per block the pairs, the mean s and Coulomb log, and the
// change of the kinetic energy by the collisions (0 to round-off with equal weights).
void collisions_diagnostics(Simulation& s, bool root);
// events: impact ionizations (count), e_ion: their ionization energy sum w_i I (m c^2 n_c / k0^3)
struct CollisionTotals { double pairs = 0, s = 0, lnL = 0, dE = 0, dP[3] = {0, 0, 0}, events = 0, e_ion = 0; };
CollisionTotals collisions_totals(Simulation& s, int block);   // sums over the ranks since the start (collective)
