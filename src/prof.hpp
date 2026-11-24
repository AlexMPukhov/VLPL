// Step profiler (environment, off by default; docs/GPU_TESTING.md "Multi-GPU overlap"):
//   VLPL_PROFILE=1         wall time of named scopes (the operators and their parts: guard exchanges, migration
//                          phases), ms per step over the counted steps, max and mean over the ranks, printed at the end;
//   VLPL_PROFILE_FROM=<n>  count only steps >= n (e.g. after a moving window has filled the box with plasma);
//   VLPL_PROFILE_SYNC=1    synchronise the device at both ends of every scope and put a barrier in front of every
//                          operator (its time is the "wait for ranks" line: load imbalance). Exact attribution of GPU
//                          time, but it removes the overlap of the exchanges with the push: use it only to see where the
//                          time goes, and time the run without it.
// Without VLPL_PROFILE a scope costs one branch. In the GPU build every scope is also an NVTX range (nsys timelines).
#pragma once
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>
#include "portable.hpp"
#if RIP_GPU && defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define VLPL_NVTX 1
#endif
#endif

class Comm;

namespace prof {
struct Entry { std::string name; double sec = 0; long calls = 0; };
struct State {
    bool on = false, sync = false, counting = false;
    long from = 0, steps = 0;
    std::vector<Entry> e;
};
State& state();
int id(const char* name);                  // registers the name once (first use)
void step(long istep);                     // at the start of a step: counting from VLPL_PROFILE_FROM on
void barrier(Comm& comm);                  // VLPL_PROFILE_SYNC: the "wait for ranks" scope before an operator
void report(FILE* f, Comm& comm);          // collective; rank 0 prints

class Scope {
public:
    explicit Scope(int id) : id_(id) {
#ifdef VLPL_NVTX
        nvtxRangePushA(state().e[size_t(id)].name.c_str());
#endif
        State& s = state();
        if (!s.counting) return;
        if (s.sync) device_sync();
        t0_ = std::chrono::steady_clock::now();
        act_ = true;
    }
    ~Scope() {
        if (act_) {
            State& s = state();
            if (s.sync) device_sync();
            Entry& en = s.e[size_t(id_)];
            en.sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
            ++en.calls;
        }
#ifdef VLPL_NVTX
        nvtxRangePop();
#endif
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
private:
    int id_;
    bool act_ = false;
    std::chrono::steady_clock::time_point t0_;
};
}  // namespace prof

#define VLPL_PROF_CAT2(a, b) a##b
#define VLPL_PROF_CAT(a, b) VLPL_PROF_CAT2(a, b)
// PROF_SCOPE("migrate.list"): times the rest of the enclosing block under that name
#define PROF_SCOPE(name)                                                     \
    static const int VLPL_PROF_CAT(prof_id_, __LINE__) = prof::id(name);     \
    prof::Scope VLPL_PROF_CAT(prof_scope_, __LINE__)(VLPL_PROF_CAT(prof_id_, __LINE__))
