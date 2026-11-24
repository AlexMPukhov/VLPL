#include "pipeline.hpp"
#include <chrono>
#include "portable.hpp"
#include "prof.hpp"
#include "sim.hpp"

const char* stage_name(Stage s) {
    static const char* n[] = {"inject", "ionize", "push", "qed", "move_deposit", "fields", "collisions", "migrate",
                              "boundaries", "diagnostics"};
    return n[int(s)];
}

void Pipeline::add(std::unique_ptr<Operator> op) {
    sec_[int(op->stage())].push_back(0.0);
    ops_[int(op->stage())].push_back(std::move(op));
}

void Pipeline::setup(Simulation& sim) {
    for (auto& st : ops_)
        for (auto& op : st) op->setup(sim);
}

void Pipeline::finish(Simulation& sim) {
    for (auto& st : ops_)
        for (auto& op : st) op->finish(sim);
}

std::vector<std::pair<std::string, OpState>> Pipeline::save(Simulation& sim) {
    std::vector<std::pair<std::string, OpState>> v;
    for (auto& st : ops_)
        for (auto& op : st) {
            OpState o;
            if (op->save(sim, o)) v.emplace_back(op->name(), std::move(o));
        }
    return v;
}

void Pipeline::run(Simulation& sim) {
    prof::step(sim.istep);
    for (int s = 0; s < int(Stage::Count); ++s)
        for (size_t i = 0; i < ops_[s].size(); ++i) {
            prof::barrier(sim.comm);                   // VLPL_PROFILE_SYNC only: the load imbalance before the operator
            const auto t0 = std::chrono::steady_clock::now();
            {
                prof::Scope sc(prof::id(ops_[s][i]->name()));
                ops_[s][i]->apply(sim);
                device_sync();
            }
            sec_[s][i] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
}

void Pipeline::report(FILE* f, double total) const {
    std::fprintf(f, "  time per operator:");
    double sum = 0;
    for (int s = 0; s < int(Stage::Count); ++s)
        for (size_t i = 0; i < ops_[s].size(); ++i) {
            std::fprintf(f, " %s %.2f s (%.0f %%)", ops_[s][i]->name(), sec_[s][i], 100 * sec_[s][i] / total);
            sum += sec_[s][i];
        }
    std::fprintf(f, ", other %.2f s\n", total - sum);
}

std::vector<std::string> Pipeline::names() const {
    std::vector<std::string> v;
    for (auto& st : ops_)
        for (auto& op : st) v.push_back(op->name());
    return v;
}

void Pipeline::describe(FILE* f) const {
    std::fprintf(f, "  pipeline:");
    bool first = true;
    for (auto& st : ops_)
        for (auto& op : st) {
            std::fprintf(f, "%s %s [%s]", first ? "" : " ->", op->name(), stage_name(op->stage()));
            first = false;
        }
    std::fprintf(f, "\n");
}
