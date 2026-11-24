// Tiles (decision 14, docs/specs/architecture.md section 5): every species stays one structure of
// arrays, kept sorted by tile, a block of T[0] x T[1] x T[2] cells of the owned box. After a sort,
// the particles of tile t are [tile_start[t], tile_start[t+1]) (Species::tile_start); between sorts
// they may stray into the neighbouring tiles (|dx| < 1 cell per step).
//
// The sort is stable (the order inside a tile is the previous order), so a run is reproducible and
// independent of the number of threads. GPU build: keys, CUB's stable radix sort of (key, index)
// pairs, the tile starts (read off the sorted keys, no atomics) and the permutation of all registered
// attributes on the device (only the nt + 1 starts go to the host); CPU build: the keys on the host and a stable counting sort. Both give the
// same permutation (decision 25).
#pragma once
#include "grid.hpp"
#include <vector>
#include "species.hpp"

struct TileGrid {
    int T[3] = {8, 8, 8};          // cells per tile
    int nt[3] = {1, 1, 1};         // tiles per axis of the owned box
    long count() const { return long(nt[0]) * nt[1] * nt[2]; }
};

TileGrid tile_grid(const Domain& d, const int cells[3]);
// Tile of a local cell (i, j, k), x fastest.
inline long tile_of(const TileGrid& g, int i, int j, int k) {
    return (long(k / g.T[2]) * g.nt[1] + j / g.T[1]) * g.nt[0] + i / g.T[0];
}
// Tiles grouped by colour (tx mod 2) + 2 (ty mod 2) + 4 (tz mod 2): colour c is order[start[c] .. start[c+1]).
void tile_colours(const TileGrid& g, std::vector<int>& order, std::vector<int>& start);
// Same-colour tiles write disjoint cells (docs/specs/tile_deposit.md): a particle moves less than one cell
// per step along every axis (dt <= h_a; RIP-CC stability gives it for y, z, but an implicit solver
// allows h_perp < c dt), and T >= 6 along every axis that has three or more tiles.
bool tiled_deposit_safe(const TileGrid& g, const Domain& d);
// Smallest number of tiles of a non-empty colour: at most this many threads work at a time.
int min_tiles_per_colour(const TileGrid& g);
// Sorts s by tile and sets s.tile_start. Particles outside the owned box go to the nearest tile.
void sort_by_tile(Species& s, const Domain& d, const TileGrid& g);
