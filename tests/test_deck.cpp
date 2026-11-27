// The common input standard of the VLPL family in VLPL (docs/specs/deck_standard.md): statements separated by ';',
// blocks on one line, a key given twice, densities with units, the common key names. Reference: the same deck in
// the spelling that VLPL always read (the start-up summary must be the same text), and n_c = 1.11485e21 cm^-3 /
// lambda[um]^2 computed here.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include "beam_import.hpp"
#include "common.hpp"
#include "sim.hpp"

namespace {

void write(const std::string& path, const std::string& text) { std::ofstream f(path); f << text; }

std::string summary_of(const std::string& path, bool names) {
    Deck d = Deck::read(path);
    if (names) d.apply_standard_names();
    Config c = Config::from_deck(d);
    const std::string out = path + ".summary";
    FILE* f = std::fopen(out.c_str(), "w");
    c.summary(f, 1);
    std::fclose(f);
    std::ifstream in(out);
    std::stringstream ss;
    ss << in.rdbuf();
    std::remove(out.c_str());
    return ss.str();
}

bool rejected(const std::string& text, const std::string& expect, std::string& msg) {
    write("deck_tmp_bad.deck", text);
    try {
        Deck d = Deck::read("deck_tmp_bad.deck");
        d.apply_standard_names();
        Config::from_deck(d);
    } catch (const std::exception& e) {
        msg = e.what();
        std::remove("deck_tmp_bad.deck");
        return msg.find(expect) != std::string::npos;
    }
    std::remove("deck_tmp_bad.deck");
    msg = "accepted";
    return false;
}

const char* old_deck =
    "grid {\n"
    "  geometry   = rz\n"
    "  modes      = 2\n"
    "  lambda0    = 0.8 um\n"
    "  nz         = 256\n"
    "  nr         = 32\n"
    "  dz         = 0.05 um\n"
    "  dr         = 0.2 um\n"
    "  boundary_z = open\n"
    "  t_end      = 200 fs\n"
    "}\n"
    "numerics {\n"
    "  solver = rip_cc\n"
    "  pusher = higuera_cary\n"
    "}\n"
    "laser {\n"
    "  a0           = 2.5\n"
    "  polarization = y\n"
    "  duration     = 10 fs\n"
    "  waist        = 4 um\n"
    "  focus        = 20 um\n"
    "}\n"
    "window {\n"
    "  start    = 40 fs\n"
    "  velocity = 1\n"
    "}\n"
    "species electrons {\n"
    "  charge    = -1\n"
    "  mass      = 1\n"
    "  density   = 0.01\n"
    "  zmin      = 13 um\n"
    "  ppc_r     = 2\n"
    "  ppc_theta = 8\n"
    "}\n"
    "output {\n"
    "  dir   = out_deck_test\n"
    "  every = 50\n"
    "}\n";

}  // namespace

void test_deck(Comm& comm) {
    if (comm.rank != 0) return;
    write("deck_tmp_old.deck", old_deck);
    const std::string ref = summary_of("deck_tmp_old.deck", false);

    // 1. the same deck with ';', blocks on one line and a comment after a statement
    write("deck_tmp_semi.deck",
          "grid { geometry = rz; modes = 2; lambda0 = 0.8 um   # reference wavelength\n"
          "  nz = 256; nr = 32\n"
          "  dz = 0.05 um; dr = 0.2 um; boundary_z = open; t_end = 200 fs }\n"
          "numerics { solver = rip_cc; pusher = higuera_cary }\n"
          "laser { a0 = 2.5; polarization = y; duration = 10 fs; waist = 4 um; focus = 20 um\n"
          "}\n"
          "window { start = 40 fs; velocity = 1 }\n"
          "species electrons { charge = -1; mass = 1; density = 0.01; zmin = 13 um; ppc_r = 2; ppc_theta = 8 }\n"
          "output {\n  dir = out_deck_test; every = 50\n}\n");
    const std::string semi = summary_of("deck_tmp_semi.deck", true);
    check("deck: statements with ';' and blocks on one line = one key per line (start-up summary)", semi == ref && !ref.empty(),
          fmt("%.0f characters", double(ref.size())));

    // 2. common names and units: m_max, tau_fwhm, z_focus, wavelength, density in cm^-3 and in n_c
    const double nc_cm3 = 1.11485422e21 / (0.8 * 0.8);               // critical density at 0.8 um
    char dens[64];
    std::snprintf(dens, sizeof dens, "%.10e cm^-3", 0.01 * nc_cm3);
    write("deck_tmp_names.deck",
          std::string("grid { geometry = rz; m_max = 1; nz = 256; nr = 32; dz = 0.05 um; dr = 0.2 um; boundary_z = open; t_end = 200 fs }\n"
                      "numerics { solver = rip_cc; pusher = higuera_cary }\n"
                      "laser { a0 = 2.5; polarization = y; wavelength = 0.8 um; tau_fwhm = 10 fs; waist = 4 um; z_focus = 20 um }\n"
                      "window { start = 40 fs; velocity = 1 }\n"
                      "species electrons { charge = -1; mass = 1; density = ") + dens +
              "; zmin = 13 um; ppc_r = 2; ppc_theta = 8 }\n"
              "output { dir = out_deck_test; every = 50 }\n");
    const std::string names = summary_of("deck_tmp_names.deck", true);
    Deck dn = Deck::read("deck_tmp_names.deck");
    dn.apply_standard_names();
    Config cn = Config::from_deck(dn);
    const double dd = std::abs(cn.species[0].density / 0.01 - 1);
    check("deck: common names (m_max, wavelength, tau_fwhm, z_focus) and a density in cm^-3 = the old keys",
          names == ref && cn.modes == 2 && dd < 1e-7,
          fmt("density %.3e cm^-3 -> n / n_c off by %.1e", 0.01 * nc_cm3, dd));
    write("deck_tmp_nc.deck", std::string(old_deck));
    Deck dc = Deck::read("deck_tmp_nc.deck");
    dc.override_from("species electrons.density=0.01 n_c");
    check("deck: a density with the unit n_c", Config::from_deck(dc).species[0].density == 0.01, "");

    // 3. errors
    std::string m;
    int ok = 0;
    std::string what;
    const auto bad = [&](const std::string& text, const std::string& expect) {
        if (rejected(text, expect, m)) ++ok; else what += " [" + expect + ": " + m + "]";
    };
    std::string t = old_deck;
    bad(t + "species ions { charge = 1; mass = 1836; density = 0.01; density = 0.02 }\n", "given twice");
    t = old_deck; t.replace(t.find("modes      = 2"), 14, "modes = 2; m_max = 1");
    bad(t, "two names of one quantity");
    t = old_deck; t.replace(t.find("duration     = 10 fs"), 20, "duration = 10 fs; tau_fwhm = 10 fs");
    bad(t, "two names of one quantity");
    t = old_deck; t.replace(t.find("waist        = 4 um"), 19, "waist = 4 um; wavelength = 1 um");
    bad(t, "differs from grid.lambda0");
    t = old_deck; t.replace(t.find("density   = 0.01"), 16, "density = 0.01 um");
    bad(t, "a density");
    t = old_deck; t.replace(t.find("ppc_theta = 8"), 13, "ppc_theta = 8; ppc_thet = 8");
    bad(t, "unknown key");
    bad("grid { geometry = rz\n", "not closed");
    check("deck: a key twice, two names of one quantity, a second wavelength, a wrong density unit, an unknown key, an open block rejected",
          ok == 7, fmt("%.0f of 7", ok) + what);

    // 4. keys and the kind of a block in any case; the name of a block keeps its case
    {
        std::string u = old_deck;
        const char* swap[][2] = {{"grid {", "Grid {"}, {"geometry   =", "Geometry ="}, {"lambda0    =", "LAMBDA0 ="}, {"laser {", "LASER {"},
                                 {"duration     =", "Duration ="}, {"species electrons {", "Species electrons {"}, {"ppc_theta =", "PPC_Theta ="}};
        for (auto& s : swap) u.replace(u.find(s[0]), std::string(s[0]).size(), s[1]);
        write("deck_tmp_case.deck", u);
        const std::string cs = summary_of("deck_tmp_case.deck", true);
        Deck dk = Deck::read("deck_tmp_case.deck");
        check("deck: keys and block kinds in any case = lower case; the name of a species keeps its case",
              cs == ref && Config::from_deck(dk).species[0].name == "electrons", "");
    }

    // 5. bunch block: a Gaussian bunch without a file. References: the values of the deck (statistical accuracy of
    //    200000 particles) and the two-pass moments of the import (file_moments) for the one-pass sums of the loader.
    {
        std::string u = old_deck;
        u += "bunch Witness { species = electron; charge = 10 pC; energy = 203.8887 MeV; energy_spread = 1e-4\n"
             "  sigma_x = 3 um; sigma_y = 2 um; sigma_z = 4 um; emittance_x = 1 mm*mrad; emittance_y = 0.5 um\n"
             "  z_center = 6 um; x_offset = 0.5 um; macroparticles = 200000 }\n";
        write("deck_tmp_bunch.deck", u);
        Deck db = Deck::read("deck_tmp_bunch.deck");
        db.apply_standard_names();
        Config cb = Config::from_deck(db);
        const SpeciesConfig* sb = nullptr;
        for (const SpeciesConfig& s : cb.species) if (s.name == "Witness") sb = &s;
        bool found = sb != nullptr && sb->file.gen.on;
        if (found) {
            BunchSource S(sb->file, cb.lambda0, 7);
            BeamParticles all;
            const double centre[3] = {sb->file.x_center, sb->file.y_center + sb->file.y_offset, sb->file.z_center + sb->file.z_offset};
            MomentSums sums(centre, sb->file.gen.gamma);
            long portion = 70001;                        // unequal portions: the sequence must not depend on them
            while (S.left() > 0) {
                const BeamParticles B = S.next(portion);
                portion = 50000;
                sums.add(B);
                all.x.insert(all.x.end(), B.x.begin(), B.x.end()); all.y.insert(all.y.end(), B.y.begin(), B.y.end());
                all.z.insert(all.z.end(), B.z.begin(), B.z.end()); all.ux.insert(all.ux.end(), B.ux.begin(), B.ux.end());
                all.uy.insert(all.uy.end(), B.uy.begin(), B.uy.end()); all.uz.insert(all.uz.end(), B.uz.begin(), B.uz.end());
                all.w.insert(all.w.end(), B.w.begin(), B.w.end());
            }
            BunchSource S1(sb->file, cb.lambda0, 7);
            const BeamParticles one = S1.next(200000);
            const bool same = one.size() == all.size() && one.x[150000] == all.x[150000] && one.uz[199999] == all.uz[199999];
            const FileMoments M = file_moments(all, cb.lambda0), Q = sums.result(cb.lambda0);
            const double e_ref = 203.8887, um = 1e-6;
            const double dq = std::abs(M.Q / 10e-12 - 1) /* round-off of the sum of 200000 weights */, de = std::abs(M.ekin_mev / e_ref - 1), ds = std::abs(M.de_rel / 1e-4 - 1);
            const double sz = std::abs(M.sigma[0] / (4 * um) - 1), sx = std::abs(M.sigma[1] / (3 * um) - 1), sy = std::abs(M.sigma[2] / (2 * um) - 1);
            const double ex = std::abs(M.eps_n[0] / (1 * um) - 1), ey = std::abs(M.eps_n[1] / (0.5 * um) - 1);
            const double cz = std::abs(M.mean[0] - 6 * um) / (4 * um), cx = std::abs(M.mean[1] - 0.5 * um) / (3 * um);
            check("deck: bunch block: charge, energy, spread, sizes, emittances, centroid of 200000 generated particles = the deck",
                  M.n == 200000 && dq < 1e-10 && de < 2e-6 && ds < 0.01 && sz < 0.01 && sx < 0.01 && sy < 0.01 && ex < 0.015 && ey < 0.015 &&
                      cz < 0.01 && cx < 0.01 && same,
                  fmt("Q %.1e, E %.1e", dq, de) + fmt(", dE/E %.1e, sigma_z %.1e", ds, sz) + fmt(", sigma_x %.1e, sigma_y %.1e", sx, sy) +
                      fmt(", eps_x %.1e, eps_y %.1e", ex, ey) + fmt(", centroid z %.1e x %.1e sigma", cz, cx) +
                      (same ? "; portions 70001 + 50000 ... = one portion" : "; PORTIONS DIFFER"));
            const double m1 = std::max({std::abs(Q.Q / M.Q - 1), std::abs(Q.ekin_mev / M.ekin_mev - 1), std::abs(Q.de_rel / M.de_rel - 1),
                                        std::abs(Q.sigma[0] / M.sigma[0] - 1), std::abs(Q.sigma[1] / M.sigma[1] - 1),
                                        std::abs(Q.sigma[2] / M.sigma[2] - 1), std::abs(Q.eps_n[0] / M.eps_n[0] - 1),
                                        std::abs(Q.eps_n[1] / M.eps_n[1] - 1), std::abs(Q.mean[0] / M.mean[0] - 1)});
            check("deck: bunch block: the one-pass moments of the loader = the two-pass moments of the file import", m1 < 1e-8,
                  fmt("max relative difference %.1e", m1));
        } else check("deck: bunch block read", false, "no species Witness with a generated bunch");
        std::string m2;
        int okb = 0;
        std::string v = u;
        v.replace(v.find("energy = 203.8887 MeV"), 21, "energy = 203.8887 MeV; gamma = 400");
        okb += rejected(v, "one of them", m2);
        v = u; v.replace(v.find("species = electron"), 18, "species = proton");
        okb += rejected(v, "electron or positron", m2);
        v = u; v.replace(v.find("charge = 10 pC"), 14, "charge = 10");
        okb += rejected(v, "a unit is required", m2);
        check("deck: bunch block: energy with gamma, a proton, a charge without a unit rejected", okb == 3, fmt("%.0f of 3", okb));
    }
    for (const char* f : {"deck_tmp_old.deck", "deck_tmp_semi.deck", "deck_tmp_names.deck", "deck_tmp_nc.deck", "deck_tmp_case.deck",
                          "deck_tmp_bunch.deck"})
        std::remove(f);
}
