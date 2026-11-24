// Environment variables of the code: VLPL_<X>; the names from when the code was formerly RIP-PIC, RIP_<X>, are still accepted
// as deprecated aliases (decision 88). The warning comes once per name and only from rank 0 of a launcher that tells the rank
// (Slurm, OpenMPI, MPICH / Intel MPI), so that JUPITER jobs with many ranks print it once.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>

inline const char* vlpl_getenv(const char* name)
{
    if (const char* v = std::getenv(name)) return v;
    if (std::strncmp(name, "VLPL_", 5) != 0) return nullptr;
    const std::string old = std::string("RIP_") + (name + 5);
    const char* v = std::getenv(old.c_str());
    if (v) {
        static std::mutex m;
        static std::set<std::string> warned;
        std::lock_guard<std::mutex> lock(m);
        if (warned.insert(old).second) {
            bool rank0 = true;
            for (const char* r : {"OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID"})
                if (const char* s = std::getenv(r)) { rank0 = std::atoi(s) == 0; break; }
            if (rank0) std::fprintf(stderr, "warning: %s is deprecated (code renamed RIP-PIC -> VLPL), use %s\n", old.c_str(), name);
        }
    }
    return v;
}
