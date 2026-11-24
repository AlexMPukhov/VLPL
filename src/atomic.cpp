#include "atomic.hpp"
#include <cctype>
#include <stdexcept>

namespace {
std::string lower(std::string s) {
    for (auto& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
std::vector<Element> build() {
    std::vector<Element> t;
#define ELEMENT(sym, z, mass) t.push_back(Element{#sym, z, mass, {}, {}});
#define LEVEL(sym, k_, ip_, l_)                                                                                     \
    if (t.empty() || t.back().symbol != #sym || int(t.back().ip_eV.size()) != (k_))                             \
        throw std::logic_error("data/ionization_energies.def: level " #sym " " #k_ " out of order");               \
    t.back().ip_eV.push_back(ip_); t.back().l.push_back(l_);
#include "../data/ionization_energies.def"
#undef ELEMENT
#undef LEVEL
    for (const Element& e : t)
        if (int(e.ip_eV.size()) != e.Z) throw std::logic_error("data/ionization_energies.def: " + e.symbol + " incomplete");
    return t;
}
}  // namespace

const std::vector<Element>& elements() {
    static const std::vector<Element> t = build();
    return t;
}

const Element* find_element(const std::string& symbol) {
    const std::string s = lower(symbol);
    for (const Element& e : elements())
        if (lower(e.symbol) == s) return &e;
    return nullptr;
}

const char* atomic_data_source() {
    return "NIST Atomic Spectra Database ver. 5.12 (Kramida, Ralchenko, Reader and NIST ASD Team), ionization energies of "
           "all charge states of H .. Lr retrieved 2026-11-08 (data/ionization_energies.def)";
}
