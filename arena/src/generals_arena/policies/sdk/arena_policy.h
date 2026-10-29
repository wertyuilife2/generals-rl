#pragma once
#include <cstdint>

// ABI 2. All arrays are read-only row-major views, valid only during arena_act.
// Unknown: terrain/structure=255, owner=-3, army=-1. Empty owner=-1,
// neutral city owner=-2. Terrain: plain=0, mountain=1.
// Structure: ordinary=0, city=1, general=2.
struct ArenaMoveResult {
    std::int64_t player, action, valid, source, target, moved;
    std::int64_t captured, eliminated, target_owned; // eliminated=-1 if none
};

struct ArenaObservation {
    int width, height, players, player;
    std::int64_t tick;
    int general_pos;
    const std::uint8_t *terrain, *structure;
    const std::int16_t *owner;
    const std::int64_t *army;
    const std::uint8_t *visible, *alive;
    const std::int64_t *total_army, *total_land, *total_cities;
    // A hidden mountain or city shares one indistinguishable obstacle hint.
    const std::uint8_t *fog_obstacles, *has_kill, *warn_afk;
    // Departure grace progress in [0,1], or -1 when no grace period is active.
    const double *neutralize_in;
    // Previous result is delivered inside the timed decision, never via an
    // untimed policy callback. Null before the first decision.
    const ArenaMoveResult *previous;
};

// Actions: WAIT=0; directions up/right/down/left=0/1/2/3, half=false leaves one.
inline std::int64_t arena_move(int source, int direction, bool half=false) {
    return 1 + std::int64_t(source)*8 + direction*2 + int(half);
}

// Each seat owns one instance. Creation is before play; no map/observation is
// available then. params_json contains only the registered immutable parameters.
extern "C" void *arena_create(int player, std::uint64_t seed, const char *params_json);
extern "C" std::int64_t arena_act(void *instance, const ArenaObservation *observation);
extern "C" void arena_destroy(void *instance);
