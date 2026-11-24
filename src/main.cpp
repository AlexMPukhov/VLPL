#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "sim.hpp"
#include "prof.hpp"
#include "fsutil.hpp"

int main(int argc, char** argv) {
    Comm comm;
    comm.init(&argc, &argv);
    int rc = 0;
    try {
        if (argc < 2)
            throw std::runtime_error("usage: vlpl deck [--check] [--restart file|latest|auto] [block.key=value ...]");
        Deck d = Deck::read(argv[1]);
        bool check = false;
        std::string restart;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--check")) check = true;
            else if (!std::strcmp(argv[i], "--restart")) {
                if (++i == argc) throw std::runtime_error("--restart needs a checkpoint file, latest or auto");
                restart = argv[i];
            } else d.override_from(argv[i]);
        }
        Config c = Config::from_deck(d);
        if (comm.rank == 0) c.summary(stdout, comm.size);
        if (comm.rank == 0) medium_summary(stdout, c);
        if (comm.rank == 0) ionization_summary(stdout, c);
        if (comm.rank == 0) qed_summary(stdout, c);
        if (comm.rank == 0) collisions_summary(stdout, c);
        if (comm.rank == 0) boost_summary(stdout, c);
        // latest: the newest checkpoint in output.dir/checkpoints (an error if there is none); auto: the
        // same, or a fresh start without one (for a chain of batch jobs)
        if (restart == "latest" || restart == "auto") {
            const std::string p = latest_checkpoint(c.dir);
            if (p.empty() && restart == "latest") throw std::runtime_error("--restart latest: no checkpoint in " + c.dir + "/checkpoints");
            restart = p;
        }
        if (!check) {
            const auto t0 = std::chrono::steady_clock::now();
            if (comm.rank == 0) make_dirs(c.dir);
            comm.barrier();
            Simulation sim(c, comm);
            sim.init(restart);
            bool fresh = true;
            FILE* f = comm.rank == 0 ? open_log(c.dir + "/diag.txt", sim.restart_step, fresh) : nullptr;
            fresh = comm.allreduce_max(comm.rank == 0 && fresh ? 1.0 : 0.0) > 0;   // diagnostics() is collective
            if (comm.rank == 0) {
                if (!restart.empty()) std::printf("  restart from %s at step %ld\n", restart.c_str(), sim.istep);
                std::printf("  decomposition %d x %d x %d%s\n", comm.dims[0], comm.dims[1], comm.dims[2],
                            RIP_GPU ? ", one GPU per rank" : "");
                sim.pipeline.describe(stdout);
                std::printf("  deposit: %s\n", sim.tiled_deposit ? "tiled (coloured tiles, no atomics)"
                                             : sim.box_deposit ? "box (one block per tile, scratch box)"
                                                               : "global (atomics)");
                if (sim.peak_wpdt > 0)
                    std::printf("  plasma: omega_p dt = %.3f at the peak density (c / omega_p = %.3g dx)\n", sim.peak_wpdt,
                                sim.dom.dt() / (sim.peak_wpdt * sim.dom.h[0]));
            }
            if (restart.empty()) {                     // step 0 (a restart continues after its step)
                sim.diagnostics(f, true);
                if (c.slice_every > 0) sim.write_slices();
                if (c.openpmd.every > 0) write_openpmd(sim);
            } else if (fresh) sim.diagnostics(f, true);   // diag.txt was missing: header and the restart step
            const CheckpointConfig& ck = c.checkpoint;
            auto t_loop = std::chrono::steady_clock::now();
            const long step0 = sim.istep;
            bool stopped = false;
            while (!sim.finished()) {
                sim.step();
                if (sim.istep % c.every == 0) sim.diagnostics(f, false);
                if (c.slice_every > 0 && sim.istep % c.slice_every == 0) sim.write_slices();
                if (c.openpmd.every > 0 && sim.istep % c.openpmd.every == 0) write_openpmd(sim);
                const bool chk = ck.every > 0 && sim.istep % ck.every == 0;
                if (chk) write_checkpoint(sim);
                if (ck.stop_hours > 0 && !sim.finished()) {   // the same decision on every rank
                    const double h = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 3600;
                    if (comm.allreduce_max(h) >= ck.stop_hours) {
                        if (!chk) write_checkpoint(sim);
                        if (comm.rank == 0)
                            std::printf("stopped after %.2f h at step %ld (checkpoint.stop_after_hours); continue with "
                                        "--restart latest\n", h, sim.istep);
                        stopped = true;
                        break;
                    }
                }
            }
            if (!stopped && sim.istep % c.every) sim.diagnostics(f, false);
            if (!stopped) sim.pipeline.finish(sim);      // e.g. the lab-frame snapshots of a boosted run
            absorber_report(sim, stdout);                // collective: the particles removed in the absorbing layers
            if (comm.rank == 0) {
                std::fclose(f);
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_loop).count();
                std::printf("done: %ld steps, %.1f s\n", sim.istep - step0, sec);
                sim.pipeline.report(stdout, sec);
            }
            comm.report(stdout);                         // collective: every rank, rank 0 prints
            prof::report(stdout, comm);                  // VLPL_PROFILE (prof.hpp), collective
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error (rank %d): %s\n", comm.rank, e.what());
        rc = 2;
    }
    comm.finalize();
    return rc;
}
