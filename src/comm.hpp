// MPI layer: 3D Cartesian domain decomposition, guard-cell exchange of fields, reductions.
// Without -DRIP_MPI it is a single-rank stub; periodic axes then exchange with itself.
// GPU build: one GPU per rank (local rank modulo the GPUs of the node); the guard layers of all fields
// of one call are packed on the device by one kernel, both directions of an axis are in flight at once,
// and they go through pinned host memory, or straight from device memory with a CUDA-aware MPI
// (environment VLPL_CUDA_AWARE=1), or device to device when the neighbour is this rank (one kernel for both directions).
// Every exchange runs in a high-priority stream ordered after the caller's stream (decision 92): the push of the interior
// particles goes on meanwhile in another stream (sim.cpp PushDeposit).
#pragma once
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>
#include "grid.hpp"
#ifdef RIP_MPI
#include <mpi.h>
#endif

class Comm {
public:
    int rank = 0, size = 1;
    int dims[3] = {1, 1, 1}, coords[3] = {0, 0, 0};
    int nbr[3][2] = {{-1, -1}, {-1, -1}, {-1, -1}};   // left/right neighbour per axis, -1 = none
    int device = -1, local_rank = 0;                    // GPU of this rank (GPU build)
    bool cuda_aware = false;                            // MPI on device pointers (GPU build, VLPL_CUDA_AWARE=1)

    void init(int* argc, char*** argv);
    void finalize();
    // Builds the Cartesian topology (want[a] = 0: let MPI choose) and fills the local part of d.
    void decompose(Domain& d, const int want[3]);

    // Guard cells: copy owned edge layers into the neighbours' guards (open edges: guards set to 0)
    void exchange(const Domain& d, std::vector<Field*> f);
    // Only the guards along x (the slabs span the guarded y, z range): after the moving window's shift,
    // whose y, z guards move with the data (window.cpp)
    void exchange_x(const Domain& d, std::vector<Field*> f);
    // Guard cells: add the guards into the neighbours' owned edge layers, then zero the guards
    void add_guards(const Domain& d, std::vector<Field*> f);

    // send to dest and receive from src (either may be -1); sizes in bytes
    void sendrecv(const void* sbuf, size_t sbytes, int dest, void* rbuf, size_t rbytes, int src, int tag);
    // Host byte buffers of any size along axis a, both directions at once: send[0] goes to the left
    // neighbour and recv[0] comes from the right one, send[1] to the right and recv[1] from the left
    // (recv[dir] stays empty without a neighbour). One message per neighbour, its size from MPI_Mprobe; tags tag + 2, tag + 3.
    void exchange_bytes(int a, const std::vector<unsigned char> send[2], std::vector<unsigned char> recv[2], int tag);

    // Time spent in the guard-cell exchanges (exchange, exchange_x, add_guards) and in exchange_bytes,
    // and the part of it waiting for MPI; report prints the maximum over the ranks (collective).
    double t_halo = 0, t_halo_mpi = 0, t_bytes = 0;
    long n_halo = 0;
    void report(FILE* f);
    void allreduce_sum(double* v, int n);
    // every rank's n values, in rank order, into recv (size * n values)
    void allgather(const double* send, int n, double* recv);
    double allreduce_max(double v);
    void bcast(std::string& s);                         // rank 0's string to every rank
    void barrier();
private:
    void halo(const Domain& d, std::vector<Field*>& f, int axis, bool add);
#ifdef RIP_MPI
    MPI_Comm cart_ = MPI_COMM_NULL;
#endif
    Buf<double> dsb_, drb_;                 // device pack buffers, two slabs (one per direction) each
    double* hsb_ = nullptr;                 // pinned host staging (GPU build), 2 x hcap_ doubles each
    double* hrb_ = nullptr;
    size_t hcap_ = 0;
    void stage(size_t cnt);                 // host staging for two slabs of cnt doubles
};
