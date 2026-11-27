#include "deck.hpp"
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r"), b = s.find_last_not_of(" \t\r");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

static std::string lower(std::string s) {
    for (char& c : s) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return s;
}

// Keys and the kind of a block ("grid", the "species" of "species Electrons") do not depend on the case; the name
// a user gives to a block does (it appears in file names).
static std::string norm_block(const std::string& b) {
    const size_t sp = b.find_first_of(" \t");
    if (sp == std::string::npos) return lower(b);
    const size_t rest = b.find_first_not_of(" \t", sp);
    return lower(b.substr(0, sp)) + " " + (rest == std::string::npos ? std::string() : b.substr(rest));
}

// the statements of a line: split at ';' outside quotes
static std::vector<std::string> statements(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    char quote = 0;
    for (char c : line) {
        if (quote) { if (c == quote) quote = 0; cur += c; }
        else if (c == '"' || c == '\'') { quote = c; cur += c; }
        else if (c == ';') { out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    out.push_back(cur);
    return out;
}

Deck Deck::read(const std::string& file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("cannot open deck " + file);
    Deck d; d.file = file;
    std::string line, block; int ln = 0;
    auto err = [&](const std::string& m) { return std::runtime_error(file + ":" + std::to_string(ln) + ": " + m); };
    while (std::getline(in, line)) {
        ++ln;
        if (auto c = line.find('#'); c != std::string::npos) line = line.substr(0, c);
        for (std::string s : statements(line)) {
            s = trim(s);
            while (!s.empty()) {
                if (block.empty()) {
                    auto br = s.find('{');
                    if (br == std::string::npos) throw err("expected 'block {'");
                    block = norm_block(trim(s.substr(0, br)));
                    if (block.empty()) throw err("a block needs a name before '{'");
                    if (d.blocks_.count(block)) throw err("block '" + block + "' given twice");
                    d.blocks_[block];
                    s = trim(s.substr(br + 1));
                    continue;
                }
                if (s[0] == '}') { block.clear(); s = trim(s.substr(1)); continue; }
                bool close = false;
                if (s.back() == '}') { close = true; s = trim(s.substr(0, s.size() - 1)); }
                auto eq = s.find('=');
                if (eq == std::string::npos) throw err("expected 'key = value'");
                const std::string key = lower(trim(s.substr(0, eq)));
                if (key.empty()) throw err("empty key");
                if (d.has(block, key))
                    throw err("key '" + block + "." + key + "' is given twice (first in line " +
                              std::to_string(d.lines_[block][key]) + ")");
                d.set(block, key, trim(s.substr(eq + 1)), ln);
                if (close) block.clear();
                s.clear();
            }
        }
    }
    if (!block.empty()) throw std::runtime_error(file + ": block '" + block + "' is not closed");
    return d;
}

void Deck::set(const std::string& b, const std::string& k, const std::string& v, int line) {
    blocks_[norm_block(b)][lower(k)] = v; lines_[norm_block(b)][lower(k)] = line;
}

void Deck::override_from(const std::string& a) {
    auto dot = a.find('.'), eq = a.find('=');
    if (dot == std::string::npos || eq == std::string::npos || dot > eq)
        throw std::runtime_error("override '" + a + "' must read block.key=value");
    set(a.substr(0, dot), a.substr(dot + 1, eq - dot - 1), a.substr(eq + 1), -1);
}

void Deck::erase(const std::string& b, const std::string& k) {
    blocks_[b].erase(lower(k));
    lines_[b].erase(lower(k));
}

void Deck::rename(const std::string& b, const std::string& neu, const std::string& old) {
    if (!has(b, neu)) return;
    if (has(b, old))
        throw std::runtime_error(where(b, neu) + ": '" + neu + "' and '" + old + "' are two names of one quantity: give one");
    const int ln = lines_[b][neu];
    set(b, old, blocks_[b][neu], ln);
    erase(b, neu);
}

void Deck::apply_standard_names() {
    // grid.m_max = highest azimuthal mode (GEM-PIC-RZ, QUARZ); VLPL counts the modes: modes = m_max + 1
    if (has("grid", "m_max")) {
        if (has("grid", "modes"))
            throw std::runtime_error(where("grid", "m_max") + ": 'm_max' and 'modes' are two names of one quantity: give one");
        const double m = parse_with_unit(blocks_["grid"]["m_max"], Kind::None, where("grid", "m_max"));
        if (m != std::floor(m) || m < 0) throw std::runtime_error(where("grid", "m_max") + ": a whole number >= 0 expected");
        const int ln = lines_["grid"]["m_max"];
        set("grid", "modes", std::to_string(long(m) + 1), ln);
        erase("grid", "m_max");
    }
    for (const std::string& b : block_names()) {
        if (b == "laser" || b.rfind("laser ", 0) == 0) {
            rename(b, "tau_fwhm", "duration");             // FWHM of the intensity, as `duration`
            rename(b, "z_focus", "focus");
            if (has(b, "wavelength")) {                    // one wavelength in VLPL: grid.lambda0
                if (!has("grid", "lambda0")) set("grid", "lambda0", blocks_[b]["wavelength"], lines_[b]["wavelength"]);
                else {
                    const double a = parse_with_unit(blocks_[b]["wavelength"], Kind::Length, where(b, "wavelength"));
                    const double l = parse_with_unit(blocks_["grid"]["lambda0"], Kind::Length, where("grid", "lambda0"));
                    if (std::abs(a - l) > 1e-12 * l)
                        throw std::runtime_error(where(b, "wavelength") + ": differs from grid.lambda0 (VLPL has one "
                                                 "reference wavelength, that of the laser)");
                }
                erase(b, "wavelength");
            }
        }
        if (b.rfind("species ", 0) == 0 && has(b, "charge")) {   // a charge with a unit is the total charge of a beam
            std::istringstream is(blocks_[b]["charge"]);
            double v; std::string unit;
            if ((is >> v) && (is >> unit) && !unit.empty()) rename(b, "charge", "total_charge");
        }
    }
}

bool Deck::has(const std::string& b, const std::string& k) const {
    auto it = blocks_.find(b);
    return it != blocks_.end() && it->second.count(lower(k));
}

std::vector<std::string> Deck::block_names() const {
    std::vector<std::string> v;
    for (auto& kv : blocks_) v.push_back(kv.first);
    return v;
}

std::string Deck::where(const std::string& b, const std::string& k) const {
    auto it = lines_.find(b);
    int ln = (it != lines_.end() && it->second.count(lower(k))) ? it->second.at(lower(k)) : 0;
    return file + (ln > 0 ? ":" + std::to_string(ln) : std::string(" (command line)")) + ": " + b + "." + k;
}

std::string Deck::str(const std::string& b, const std::string& k, const std::string& def) const {
    if (!has(b, k)) return def;
    used_.insert(b + "." + lower(k));
    return blocks_.at(b).at(lower(k));
}
std::string Deck::str(const std::string& b, const std::string& k) const {
    if (!has(b, k)) throw std::runtime_error(file + ": missing key " + b + "." + k);
    return str(b, k, "");
}
double Deck::num(const std::string& b, const std::string& k, Kind kind) const {
    if (kind == Kind::Density) {
        double k0 = 0;
        if (has("grid", "lambda0"))
            k0 = 2 * M_PI / parse_with_unit(blocks_.at("grid").at("lambda0"), Kind::Length, where("grid", "lambda0"));
        return parse_density(str(b, k), k0, where(b, k));
    }
    return parse_with_unit(str(b, k), kind, where(b, k));
}
double Deck::num(const std::string& b, const std::string& k, Kind kind, double def) const {
    return has(b, k) ? num(b, k, kind) : def;
}
long Deck::integer(const std::string& b, const std::string& k, long def) const {
    if (!has(b, k)) return def;
    double v = num(b, k, Kind::None);
    if (v != std::floor(v)) throw std::runtime_error(where(b, k) + ": integer expected");
    return long(v);
}
bool Deck::flag(const std::string& b, const std::string& k, bool def) const {
    if (!has(b, k)) return def;
    std::string v = str(b, k, "");
    if (v == "true" || v == "on" || v == "yes" || v == "1") return true;
    if (v == "false" || v == "off" || v == "no" || v == "0") return false;
    throw std::runtime_error(where(b, k) + ": true or false expected");
}
void Deck::check_unused() const {
    for (auto& [b, kv] : blocks_)
        for (auto& [k, v] : kv)
            if (!used_.count(b + "." + k)) throw std::runtime_error(where(b, k) + ": unknown key");
}

static const std::map<std::string, std::pair<Kind, double>> units = {
    {"nm", {Kind::Length, 1e-9}}, {"um", {Kind::Length, 1e-6}}, {"\xC2\xB5m", {Kind::Length, 1e-6}},
    {"micron", {Kind::Length, 1e-6}}, {"mm", {Kind::Length, 1e-3}},
    {"cm", {Kind::Length, 1e-2}}, {"m", {Kind::Length, 1.0}},
    {"as", {Kind::Time, 1e-18}}, {"fs", {Kind::Time, 1e-15}}, {"ps", {Kind::Time, 1e-12}}, {"ns", {Kind::Time, 1e-9}},
    {"s", {Kind::Time, 1.0}},
    {"fC", {Kind::Charge, 1e-15}}, {"pC", {Kind::Charge, 1e-12}}, {"nC", {Kind::Charge, 1e-9}}, {"uC", {Kind::Charge, 1e-6}},
    {"rad", {Kind::Angle, 1.0}}, {"mrad", {Kind::Angle, 1e-3}}, {"urad", {Kind::Angle, 1e-6}},
    {"deg", {Kind::Angle, M_PI / 180}},
    {"mm*mrad", {Kind::Length, 1e-6}}, {"mm.mrad", {Kind::Length, 1e-6}},
    {"eV", {Kind::Energy, 1e-6}}, {"keV", {Kind::Energy, 1e-3}}, {"MeV", {Kind::Energy, 1.0}},     // in MeV
    {"GeV", {Kind::Energy, 1e3}}, {"TeV", {Kind::Energy, 1e6}}};

bool unit_to_code(const std::string& unit, double k0, double& f) {
    const double c = 299792458.0, eps0 = 8.8541878128e-12, me = 9.1093837015e-31, e = 1.602176634e-19;
    const double nc = eps0 * me * (c * k0) * (c * k0) / (e * e);        // critical density [1/m^3]
    if (unit == "cm^-3" || unit == "cm-3" || unit == "1/cm3" || unit == "1/cm^3") { f = 1e6 / nc; return true; }
    if (unit == "m^-3" || unit == "m-3" || unit == "1/m3" || unit == "1/m^3") { f = 1.0 / nc; return true; }
    auto it = units.find(unit);
    if (it == units.end() || it->second.first == Kind::Charge || it->second.first == Kind::Energy) return false;
    const double si = it->second.second;
    f = it->second.first == Kind::Length ? si * k0 : it->second.first == Kind::Time ? si * c * k0 : si;
    return true;
}

double parse_density(const std::string& text, double k0, const std::string& where) {
    std::istringstream is(text);
    double v; std::string unit;
    if (!(is >> v)) throw std::runtime_error(where + ": number expected, got '" + text + "'");
    is >> unit;
    if (unit.empty() || unit == "n_c" || unit == "nc") return v;       // in critical densities
    double f = 0;
    const bool dens = unit.find("cm") != std::string::npos || unit.find("m^-3") != std::string::npos ||
                      unit.find("m-3") != std::string::npos || unit.find("/m") != std::string::npos;
    if (!dens || !unit_to_code(unit, k0 > 0 ? k0 : 1.0, f))
        throw std::runtime_error(where + ": a density: a number (n / n_c) or with the unit n_c, cm^-3, m^-3; got '" + unit + "'");
    if (!(k0 > 0)) throw std::runtime_error(where + ": a density with a unit needs grid.lambda0");
    return v * f;
}

double parse_with_unit(const std::string& text, Kind kind, const std::string& where) {
    std::istringstream is(text);
    double v; std::string unit;
    if (!(is >> v)) throw std::runtime_error(where + ": number expected, got '" + text + "'");
    is >> unit;
    if (unit.empty()) {
        if (kind == Kind::None || kind == Kind::Angle) return v;   // a bare angle is in radians
        throw std::runtime_error(where + ": a unit is required");
    }
    if (kind == Kind::None) throw std::runtime_error(where + ": dimensionless value expected");
    auto it = units.find(unit);
    if (it == units.end()) throw std::runtime_error(where + ": unknown unit '" + unit + "'");
    if (it->second.first != kind) throw std::runtime_error(where + ": unit '" + unit + "' is of the wrong kind");
    return v * it->second.second;
}
