// openPMD 1.1.0 files over HDF5 (docs/specs/openpmd.md, decision 34): field and particle output with
// the ED-PIC extension, and checkpoints that restart a run bit-exactly. One file per iteration
// (fileBased); the values are stored in code units with their unitSI, so a checkpoint holds the
// state exactly. All ranks call these functions together; the blocks of the ranks stream through
// rank 0, which alone writes (serial HDF5; at most one block or particle chunk in its memory). On a
// restart every rank reads its own part of the file.
#pragma once
#include <string>
#include <vector>

class Simulation;

struct OpenPmdConfig {
    long every = 0;                                   // steps, 0: no output
    std::vector<std::string> fields{"E", "B", "rho"}; // E B J rho rho_<species>
    std::vector<std::string> species;                 // species with particle output ("all": every one)
    bool single = true;                               // float32 values (checkpoints: always float64)
};

struct CheckpointConfig {
    long every = 0;                                   // steps, 0: none
    int keep = 2;                                     // the newest kept, older ones deleted (0: all kept)
    double stop_hours = 0;                            // wall time [h] after which a checkpoint is written and
                                                      // the run ends (a batch time limit), 0: never
};

bool hdf5_available();                                // built with HDF5 (RIP_HDF5)
// dir/openpmd/data_<step>.h5 of the current step.
void write_openpmd(Simulation& s);
// dir/checkpoints/checkpoint_<step>.h5 of the current step (written under a temporary name, then
// renamed: an interrupted write never looks like a checkpoint); returns the path.
std::string write_checkpoint(Simulation& s);
// Restores the state of a checkpoint into a constructed Simulation (called by Simulation::init).
// Same decomposition: bit-exact continuation; another decomposition: the particles are redistributed.
void read_checkpoint(Simulation& s, const std::string& path);
// The state of operator `op` in the checkpoint this run started from (Simulation::restart_path; group rippic/op_<op>,
// Operator::save, decision 116): the listed keys (a key not in the checkpoint comes back empty); false if the
// checkpoint holds no state of op (written before decision 116, or without that operator) or the run did not restart.
// Not collective: each rank reads the keys it lists.
struct OpState;
bool read_op_state(const Simulation& s, const std::string& op, const std::vector<std::string>& keys, OpState& st);
// The newest checkpoint in dir/checkpoints, "" if there is none.
std::string latest_checkpoint(const std::string& dir);
