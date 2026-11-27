// Particle storage: structure of arrays in device memory with a registry of named attributes
// (decision 12, docs/specs/architecture.md section 4).
//
// The core attributes x, y, z, ux, uy, uz, w are F64 in the fixed slots 0..6, so kernels index them by
// constant. Every further attribute is registered by the operator that needs it, before the first
// step:   int s = species.require("chi", Attr::F64, 0.0);
// Append, migration (and later sorting, I/O and checkpoints) loop over the registry, so a new
// attribute needs no change there. Kernels get raw pointers: species.ptr<double>(s).
#pragma once
#include <string>
#include <vector>
#include "comm.hpp"
#include "grid.hpp"

enum class Attr { F64, F32, I64, I32 };
inline int attr_size(Attr t) { return t == Attr::F64 || t == Attr::I64 ? 8 : 4; }

template <class T> struct AttrOf;
template <> struct AttrOf<double> { static constexpr Attr t = Attr::F64; };
template <> struct AttrOf<float> { static constexpr Attr t = Attr::F32; };
template <> struct AttrOf<long long> { static constexpr Attr t = Attr::I64; };
template <> struct AttrOf<int> { static constexpr Attr t = Attr::I32; };

// The attribute arrays of a species for one kernel (pointers valid until the storage grows)
struct AttrTable {
    unsigned char* p[32];
    int es[32];                   // element size in bytes (4 or 8)
    int n = 0, R = 0;             // attributes, bytes of a row
};

struct AttrInfo {
    std::string name;
    Attr type = Attr::F64;
    double init = 0;              // value of new particles that do not carry the attribute
};

class Species {
public:
    enum { X, Y, Z, UX, UY, UZ, W, NC };         // core F64 slots
    static constexpr int MAX_ATTR = 32;
    std::string name;
    double q = -1, m = 1;

    Species();
    // Registers an attribute (idempotent; a type mismatch with an existing one throws). Particles that
    // already exist get the init value. Returns the slot.
    int require(const std::string& name, Attr type, double init = 0);
    int slot(const std::string& name) const;     // -1 if not registered
    int n_attr() const { return int(info_.size()); }
    const AttrInfo& info(int s) const { return info_[s]; }
    int row_bytes() const;                       // bytes of one particle over all attributes

    size_t size() const { return n_; }
    void reserve(size_t cap);
    void resize(size_t n) { reserve(n); n_ = n; tile_start.clear(); }

    template <class T> T* ptr(int s) { check<T>(s); return reinterpret_cast<T*>(a_[s].data()); }
    template <class T> const T* ptr(int s) const { check<T>(s); return reinterpret_cast<const T*>(a_[s].data()); }

    // Core data from the host (x .. w), appended at the end; other attributes get their init value.
    void append(const std::vector<double> h[NC]);
    // m particles at the end whose core slots the caller fills (kernels); the other attributes get their
    // init value. Returns the index of the first.
    size_t extend(size_t m);
    // Packed rows (row_bytes() each, attributes in slot order) from the host, appended at the end.
    void append_rows(const std::vector<unsigned char>& rows);
    void append_rows(const unsigned char* rows, size_t bytes);
    // Device pointers and element sizes of all attributes, for kernels that move whole particles (migration)
    AttrTable table();
    // a[q] = a_old[perm[q]] for every attribute, perm a permutation of 0 .. size()-1 (device).
    void permute(const std::vector<int>& perm);
    void permute_device(const int* perm);        // perm already in device memory (size() entries)

    // Tile ranges of the last sort_by_tile (tiles.hpp); cleared when particles are added or removed.
    std::vector<int> tile_start;

    template <class T> std::vector<T> get_as(int s) const;                  // host copy of one attribute
    template <class T> void set_as(int s, const std::vector<T>& v);         // same length as size()
    std::vector<double> get(int s) const { return get_as<double>(s); }
    void set(int s, const std::vector<double>& v) { set_as<double>(s, v); }

    double kinetic_energy() const;               // sum w m (gamma - 1), local; photons: sum w |u| (m_e c^2)
    bool photon = false;                         // kind = photon (qed.hpp): q = 0, moved ballistically by the qed module
    // charge state per particle (field ionization, ionization.hpp): the slot of the I32 attribute charge_state (-1: none,
    // every particle has the charge q) and the largest charge state (the fixed-point scale of the box deposit)
    int zslot = -1;
    double charge_max = 1;
    double lost_count = 0, lost_charge = 0;      // particles removed in the absorbing layers and their charge q w Z (local; absorber_particles.cpp)
    double lost_energy = 0;                      // w m (gamma - 1) of the particles removed at open edges or the RZ wall (local)
private:
    template <class T> void check(int s) const {
        if (s < 0 || s >= n_attr() || info_[s].type != AttrOf<T>::t) bad_access(s);
    }
    [[noreturn]] void bad_access(int s) const;
    void fill_init(int s, size_t from, size_t to);
    std::vector<AttrInfo> info_;
    std::vector<Buf<unsigned char>> a_;          // a_[s]: capacity cap_ elements of attr_size bytes
    Buf<unsigned char> tmp_;                     // gather scratch of permute (kept: no allocation per sort)
    Buf<int> perm_;                              // device copy of a host permutation
    size_t n_ = 0, cap_ = 0;
};

// Particles leaving the owned box go to the neighbour ranks (or wrap, or leave at an open edge). RZ
// (rwall > 0, d = the RZ field domain): only z (X) is a box axis; particles at r = |(Y, Z)| >= rwall
// are absorbed by the wall.
void migrate(Species& s, const Domain& d, Comm& comm, double rwall = 0);
// All species at once: one message per axis and direction for all of them (the order of each species as one by one)
void migrate(const std::vector<Species*>& sp, const Domain& d, Comm& comm, double rwall = 0);
// Removal of particles on the device (e.g. the particles absorbed by a metal block, metal3d.cpp): removal_flags(n) gives
// a device array of n ints (scratch of the migration; valid until the next removal or migration); set F[p] = 1 for the
// particles to remove and 0 for all others, then remove_flagged closes the holes from the tail (the order of a
// migration) and returns the number removed.
int* removal_flags(long n);
long remove_flagged(Species& s);
