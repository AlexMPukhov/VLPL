// Elements known to the code: atomic number, standard atomic mass and, for every charge state k, the ionization energy
// I_k (k -> k + 1, NIST ASD 5.12, H .. Lr) and the orbital quantum number l of the electron removed. The table is the data file
// data/ionization_energies.def (provenance there; docs/specs/field_ionization.md).
#pragma once
#include <string>
#include <vector>

struct Element {
    std::string symbol;            // "H", "He", ... (capitalized)
    int Z = 0;                     // atomic number
    double mass_u = 0;             // standard atomic mass [u]
    std::vector<double> ip_eV;     // ip_eV[k]: ionization energy of k -> k + 1 [eV], size Z
    std::vector<int> l;            // l[k]: orbital quantum number of the electron removed in k -> k + 1
};

const std::vector<Element>& elements();                 // all of the table, by atomic number
const Element* find_element(const std::string& symbol); // case-insensitive; nullptr if unknown
const char* atomic_data_source();                       // provenance (run summary)
