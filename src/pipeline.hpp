// Step pipeline of operators (docs/specs/architecture.md section 6). One step n -> n+1 runs the stages
// in this fixed order; within a stage the operators run in the order they were added. Each physics
// module is one operator with its own deck block. Operators only launch kernels through free
// functions (no kernel lambdas in member functions, no this capture).
//
// Push and move+deposit are one fused operator (stage Push) unless an ionization or QED operator
// needs u^{n+1/2} before the move (decision 15); stage MoveDeposit is then empty.
#pragma once
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class Simulation;

enum class Stage { Inject, Ionize, Push, Qed, MoveDeposit, Fields, Collisions, Migrate, Boundaries, Diagnostics, Count };
const char* stage_name(Stage s);

// The state of an operator that a restart needs beyond fields and particles (decision 116): named float64 arrays, in a
// checkpoint the group rippic/op_<name> (openpmd.cpp). Saving is collective; rank 0's arrays are written (the operator
// gathers what the ranks hold). An operator reads its state back in setup with read_op_state (openpmd.hpp).
struct OpState {
    std::vector<std::pair<std::string, std::vector<double>>> arrays;
    void add(const std::string& key, std::vector<double> v) { arrays.emplace_back(key, std::move(v)); }
    const std::vector<double>* get(const std::string& key) const {   // null: not in the state
        for (const auto& a : arrays)
            if (a.first == key) return &a.second;
        return nullptr;
    }
};

class Operator {
public:
    virtual ~Operator() = default;
    virtual const char* name() const = 0;
    virtual Stage stage() const = 0;
    virtual void setup(Simulation&) {}        // read the deck block, require attributes, add hooks
    virtual void apply(Simulation&) = 0;
    virtual void finish(Simulation&) {}       // at the end of a run that was not stopped (collective; e.g. flush outputs)
    virtual bool save(Simulation&, OpState&) { return false; }   // at a checkpoint (collective): true if it has state
};

class Pipeline {
public:
    void add(std::unique_ptr<Operator> op);
    void setup(Simulation& sim);              // calls every operator's setup once, in run order
    void run(Simulation& sim);                // one step
    void finish(Simulation& sim);             // every operator's finish, at the end of the run
    std::vector<std::pair<std::string, OpState>> save(Simulation& sim);   // the operators with state (checkpoints; collective)
    std::vector<std::string> names() const;   // in run order
    void describe(FILE* f) const;             // "  pipeline: push_deposit [push] -> ..."
    void report(FILE* f, double total) const; // wall time per operator (synchronised on the GPU)
private:
    std::vector<std::unique_ptr<Operator>> ops_[int(Stage::Count)];
    std::vector<double> sec_[int(Stage::Count)];
};
