#include "prof.hpp"
#include <cstdlib>
#include <cstring>
#include "comm.hpp"
#include "env.hpp"

namespace prof {
State& state() {
    static State* s = [] {
        State* p = new State;
        const char* on = vlpl_getenv("VLPL_PROFILE");
        p->on = on && std::atoi(on) != 0;
        const char* sy = vlpl_getenv("VLPL_PROFILE_SYNC");
        p->sync = p->on && sy && std::atoi(sy) != 0;
        const char* fr = vlpl_getenv("VLPL_PROFILE_FROM");
        p->from = fr ? std::atol(fr) : 0;
        return p;
    }();
    return *s;
}

int id(const char* name) {
    State& s = state();
    for (size_t q = 0; q < s.e.size(); ++q)
        if (s.e[q].name == name) return int(q);
    s.e.push_back(Entry{name, 0.0, 0});
    return int(s.e.size()) - 1;
}

void step(long istep) {
    State& s = state();
    s.counting = s.on && istep >= s.from;
    if (s.counting) ++s.steps;
}

void barrier(Comm& comm) {
    State& s = state();
    if (!s.counting || !s.sync) return;
    static const int w = id("wait for ranks");
    Scope sc(w);
    comm.barrier();
}

// Rank 0's names in its order; every rank adds its time of each (0 if it never ran that scope).
void report(FILE* f, Comm& comm) {
    State& s = state();
    if (!s.on) return;
    std::string names;
    for (const Entry& e : s.e) names += e.name + '\n';
    comm.bcast(names);
    std::vector<std::string> nm;
    for (size_t a = 0, b; (b = names.find('\n', a)) != std::string::npos; a = b + 1) nm.push_back(names.substr(a, b - a));
    const int n = int(nm.size());
    std::vector<double> mine(size_t(n) + 1, 0.0);
    for (int q = 0; q < n; ++q)
        for (const Entry& e : s.e)
            if (e.name == nm[size_t(q)]) mine[size_t(q)] = e.sec;
    mine[size_t(n)] = double(s.steps);
    std::vector<double> all(size_t(comm.size) * (n + 1));
    comm.allgather(mine.data(), n + 1, all.data());
    if (comm.rank != 0 || !f) return;
    const double steps = std::max(1.0, all[size_t(n)]);
    std::fprintf(f, "  profile (%.0f steps from step %ld%s), ms per step, max / mean over %d ranks:\n", steps, s.from,
                 s.sync ? ", synchronised scopes" : "", comm.size);
    for (int q = 0; q < n; ++q) {
        double mx = 0, sum = 0;
        for (int r = 0; r < comm.size; ++r) {
            const double v = all[size_t(r) * (n + 1) + q];
            mx = std::max(mx, v); sum += v;
        }
        std::fprintf(f, "    %-28s %9.3f %9.3f\n", nm[size_t(q)].c_str(), 1e3 * mx / steps, 1e3 * sum / comm.size / steps);
    }
}
}  // namespace prof
