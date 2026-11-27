// Structured ASCII input deck: blocks `name { key = value }`, comments with '#', statements ending at the end of
// the line or at ';', values with physical units, command-line overrides `block.key=value`.
// (Taken over from GEM-PIC-3D; block_names() added.) The form is the common input standard of the VLPL family
// (VLPL, GEM-PIC-RZ, QUARZ): docs/specs/deck_standard.md.
#pragma once
#include <map>
#include <set>
#include <string>
#include <vector>

// Density: a bare number is n / n_c (as always); with a unit (cm^-3, m^-3, n_c) it is converted with grid.lambda0
// Energy: in MeV
enum class Kind { None, Length, Time, Charge, Angle, Density, Energy };

class Deck {
public:
    static Deck read(const std::string& file);
    void set(const std::string& block, const std::string& key, const std::string& value, int line = 0);
    void override_from(const std::string& assignment);        // "block.key=value"
    // The common key names of the VLPL family rewritten into the keys the code reads (grid.m_max -> modes,
    // laser tau_fwhm -> duration, z_focus -> focus, wavelength = grid.lambda0, species charge with a unit ->
    // total_charge); two names of one quantity are an error. Call once after the overrides.
    void apply_standard_names();
    bool has_block(const std::string& b) const { return blocks_.count(b) > 0; }
    bool has(const std::string& b, const std::string& k) const;
    std::vector<std::string> block_names() const;
    std::string str(const std::string& b, const std::string& k, const std::string& def) const;
    std::string str(const std::string& b, const std::string& k) const;          // required
    double num(const std::string& b, const std::string& k, Kind kind) const;    // required, SI units
    double num(const std::string& b, const std::string& k, Kind kind, double def_si) const;
    long integer(const std::string& b, const std::string& k, long def) const;
    bool flag(const std::string& b, const std::string& k, bool def) const;
    void check_unused() const;                                                  // unknown keys are errors
    std::string file = "(deck)";
private:
    std::string where(const std::string& b, const std::string& k) const;
    void rename(const std::string& b, const std::string& neu, const std::string& old);
    void erase(const std::string& b, const std::string& k);
    std::map<std::string, std::map<std::string, std::string>> blocks_;
    std::map<std::string, std::map<std::string, int>> lines_;
    mutable std::set<std::string> used_;
};

double parse_with_unit(const std::string& text, Kind kind, const std::string& where);
// A density: bare number or "<n> n_c" = n / n_c; "<n> cm^-3" / "<n> m^-3" with k0 = 2 pi / lambda0 [1/m]
double parse_density(const std::string& text, double k0, const std::string& where);
// Factor from a unit name to code units for expressions (lengths 1/k0, times 1/omega0, densities n_c,
// angles rad); false if `unit` is no unit name. k0 = 2 pi / lambda0 [1/m].
bool unit_to_code(const std::string& unit, double k0, double& factor);
