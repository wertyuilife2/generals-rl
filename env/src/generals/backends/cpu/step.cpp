// Native single-tick batch API. No policy, Python callbacks, or action history.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <omp.h>

// Benchmark-only hooks compile to nothing in the production library.
#ifndef GENERALS_PROFILE_MARK
#define GENERALS_PROFILE_MARK(phase) ((void)0)
#define GENERALS_PROFILE_BATCH_BEGIN(batch, threads) ((void)0)
#define GENERALS_PROFILE_BATCH_END() ((void)0)
#define GENERALS_PROFILE_WORKER_BEGIN() ((void)0)
#define GENERALS_PROFILE_WORKER_END() ((void)0)
#define GENERALS_PROFILE_LANE_BEGIN(lane) ((void)0)
#define GENERALS_PROFILE_LANE_END(lane) ((void)0)
#endif

using i64 = std::int64_t;
using u64 = std::uint64_t;
using wide = unsigned __int128;
constexpr u64 MAX_ARMY = 0x7fffffffffffffffULL;

struct Batch {
    std::uint8_t *terrain, *structure;
    std::int16_t *owner;
    i64 *army;
    std::uint8_t *alive;
    std::int32_t *general_pos;
    i64 *tick;
    std::uint8_t *terminated;
    std::int32_t *winner;
};
struct Outputs {
    i64 *stats, *moves, *counters;
    std::int32_t *errors;
    std::uint8_t *done;
    i64 *current_stats;
    std::uint16_t *growth_cells;
    std::int32_t *growth_count;
    std::uint8_t *cache_valid;
    std::int32_t *owned_structures;
};
struct Totals {
    wide army[8]{};
    i64 territory[8]{}, cities[8]{};
    std::int32_t structures[8]{};
};

static Totals scan(const std::int16_t *owner, const i64 *army,
                   const std::uint8_t *structure, int n) {
    Totals result;
    for (int cell = 0; cell < n; ++cell) {
        int pid = owner[cell];
        if (pid >= 0) {
            result.army[pid] += u64(army[cell]);
            ++result.territory[pid];
            result.cities[pid] += structure[cell] == 1;
            result.structures[pid] += structure[cell] != 0;
        }
    }
    return result;
}
static void empty(i64 *record) {
    std::fill(record, record + 9, 0);
    record[2] = record[3] = record[6] = -1;
}
static inline void replace(int cell, int pid, i64 count, int kind, Totals &total,
                           std::int16_t *owner, i64 *army, std::uint8_t *structure) {
    int old = owner[cell];
    if (old >= 0) {
        total.army[old] -= u64(army[cell]);
        --total.territory[old];
        total.cities[old] -= structure[cell] == 1;
        total.structures[old] -= structure[cell] != 0;
    }
    if (pid >= 0) {
        total.army[pid] += u64(count);
        ++total.territory[pid];
        total.cities[pid] += kind == 1;
        total.structures[pid] += kind != 0;
    }
    owner[cell] = pid; army[cell] = count; structure[cell] = kind;
}

static void step_lane(Batch s, const Batch initial, Outputs out, const i64 *actions,
                      int lane, int h, int w, int p, bool auto_reset, i64 horizon) {
    int n = h * w;
    const std::size_t base = std::size_t(lane) * n, pb = std::size_t(lane) * p;
    auto *owner = s.owner + base;
    auto *army = s.army + base;
    auto *structure = s.structure + base;
    auto *terrain = s.terrain + base;
    auto *alive = s.alive + pb;
    auto *moves = out.moves + pb * 9;
    auto *stats = out.stats + pb * 3;
    auto *counts = reinterpret_cast<u64 *>(out.counters + std::size_t(lane) * 9);
    auto *cache = out.current_stats + pb * 3;
    auto *growth = out.growth_cells + base;
    int &growth_count = out.growth_count[lane];
    int &error = out.errors[lane];
    if (error || s.terminated[lane] || (!auto_reset && horizon > 0 && s.tick[lane] >= horizon)) {
        for (int pid = 0; pid < p; ++pid) empty(moves + pid * 9);
        return;
    }
    if (u64(s.tick[lane]) == MAX_ARMY) { error = 4; return; }
    Totals total;
    if (out.cache_valid[lane]) {
        for (int pid = 0; pid < p; ++pid) {
            total.army[pid] = u64(cache[pid * 3]);
            total.territory[pid] = cache[pid * 3 + 1];
            total.cities[pid] = cache[pid * 3 + 2];
            total.structures[pid] = out.owned_structures[pb + pid];
        }
    } else {
        total = scan(owner, army, structure, n);
        growth_count = 0;
        for (int cell = 0; cell < n; ++cell)
            if (structure[cell] && owner[cell] >= 0) growth[growth_count++] = cell;
    }
    GENERALS_PROFILE_MARK(0); // Setup and cache read/rebuild.
    const i64 tick = s.tick[lane] + 1;
    if (tick % 2 == 0) {
        bool all = tick % 50 == 0;
        int size = !all ? growth_count : n;
        for (int k = 0; k < size; ++k) {
            int cell = !all ? growth[k] : k;
            if ((!all || owner[cell] >= 0) && u64(army[cell]) == MAX_ARMY) {
                error = 1; return;
            }
        }
        for (int k = 0; k < size; ++k) {
            int cell = !all ? growth[k] : k;
            if (!all || owner[cell] >= 0) ++army[cell];
        }
        for (int pid = 0; pid < p; ++pid)
            total.army[pid] += all ? total.territory[pid] : total.structures[pid];
    }
    GENERALS_PROFILE_MARK(1); // Growth.
    s.tick[lane] = tick;
    for (int pid = 0; pid < p; ++pid) {
        i64 *r = moves + pid * 9;
        empty(r);
        if (!alive[pid] || s.terminated[lane] || error) continue;
        r[0] = 1; ++counts[1];
        i64 action = actions[pb + pid];
        if (!action) { r[1] = 1; r[8] = 1; ++counts[3]; continue; }
        ++counts[2];
        i64 encoded = i64(u64(action) - 1), source = encoded / 8;
        int direction = (encoded % 8) / 2;
        bool half = encoded % 2 == 1, in_bounds = action > 0 && source >= 0 && source < n;
        int target = -1;
        if (in_bounds) {
            int y = int(source) / w, x = int(source) % w;
            int ty = y + (direction == 0 ? -1 : direction == 2 ? 1 : 0);
            int tx = x + (direction == 3 ? -1 : direction == 1 ? 1 : 0);
            in_bounds = ty >= 0 && ty < h && tx >= 0 && tx < w;
            if (in_bounds) target = ty * w + tx;
        }
        r[2] = source; r[3] = target;
        r[8] = !in_bounds ? 3 : owner[source] != pid ? 4 : terrain[target] == 1 ? 5 : 2;
        if (r[8] != 2) { ++counts[4]; continue; }
        r[1] = 1;
        i64 count = army[source];
        i64 moved = half ? count / 2 : std::max<i64>(0, count - 1);
        r[4] = moved;
        int defender = owner[target], kind = structure[target];
        i64 defense = army[target];
        bool friendly = defender == pid;
        if (friendly && u64(moved) > MAX_ARMY - u64(defense)) { error = 2; continue; }
        bool captured = !friendly && moved > defense;
        i64 next = friendly ? defense + moved : captured ? moved - defense : defense - moved;
        // The source never changes owner/structure. Friendly transfers leave
        // every player's totals unchanged, including zero-army transfers.
        army[source] = count - moved;
        if (friendly) {
            army[target] = next;
        } else {
            total.army[pid] -= u64(moved);
            if (captured) {
                if (defender < 0 && kind) growth[growth_count++] = target;

                replace(target, pid, next, kind, total, owner, army, structure);
            } else {
                army[target] = next;
                if (defender >= 0) total.army[defender] -= u64(moved);
            }
        }
        r[5] = captured; r[7] = friendly || captured;
        if (captured && kind == 2) {
            r[6] = defender; alive[defender] = false; structure[target] = 1;
            ++total.cities[pid];
            for (int cell = 0; cell < n; ++cell)
                if (owner[cell] == defender)
                    replace(cell, pid, std::max<i64>(1, army[cell] / 2), structure[cell],
                                    total, owner, army, structure);
            int survivors = 0;
            for (int player = 0; player < p; ++player) survivors += alive[player] != 0;
            if (survivors == 1) { s.terminated[lane] = true; s.winner[lane] = pid; }
        }
        counts[5] += moved > 0; counts[6] += captured;
    }
    GENERALS_PROFILE_MARK(2); // Ordered player actions and elimination.

    for (int pid = 0; pid < p; ++pid) {
        bool overflow = total.army[pid] > MAX_ARMY;
        if (overflow && !error) error = 3;
        stats[pid * 3] = overflow ? 0 : i64(total.army[pid]);
        stats[pid * 3 + 1] = total.territory[pid];
        stats[pid * 3 + 2] = total.cities[pid];
    }
    if (error) { out.cache_valid[lane] = 0; return; }
    ++counts[0];
    bool terminal = s.terminated[lane], truncated = !terminal && horizon > 0 && tick >= horizon;
    out.done[lane * 2] = terminal; out.done[lane * 2 + 1] = truncated;
    counts[7] += terminal; counts[8] += truncated;
    GENERALS_PROFILE_MARK(3); // Statistics, counters and done.
    if (auto_reset && (terminal || truncated)) {
        std::memcpy(structure, initial.structure + base, n);
        std::memcpy(owner, initial.owner + base, n * sizeof(std::int16_t));
        std::memcpy(army, initial.army + base, n * sizeof(i64));
        std::memcpy(alive, initial.alive + pb, p);
        s.tick[lane] = initial.tick[lane];
        s.terminated[lane] = initial.terminated[lane]; s.winner[lane] = initial.winner[lane];
        total = scan(owner, army, structure, n);
        growth_count = 0;
        for (int cell = 0; cell < n; ++cell)
            if (structure[cell] && owner[cell] >= 0) growth[growth_count++] = cell;
    }
    GENERALS_PROFILE_MARK(4); // Automatic reset and cache rebuild.
    for (int pid = 0; pid < p; ++pid) {
        cache[pid * 3] = i64(total.army[pid]);
        cache[pid * 3 + 1] = total.territory[pid]; cache[pid * 3 + 2] = total.cities[pid];
        out.owned_structures[pb + pid] = total.structures[pid];
    }
    out.cache_valid[lane] = 1;
    GENERALS_PROFILE_MARK(5); // Cache writeback.
}

extern "C" void generals_step(Batch *s, const Batch *initial, Outputs *out, const i64 *actions,
                              int batch, int h, int w, int players, int threads,
                              int auto_reset, i64 horizon) {
    GENERALS_PROFILE_BATCH_BEGIN(batch, threads);
    auto advance = [&](int lane) {
        GENERALS_PROFILE_LANE_BEGIN(lane);
        step_lane(*s, *initial, *out, actions, lane, h, w, players, auto_reset, horizon);
        GENERALS_PROFILE_LANE_END(lane);
    };
    // One statically partitioned OpenMP team per public tick.
    #pragma omp parallel num_threads(threads)
    {
        GENERALS_PROFILE_WORKER_BEGIN();
        #pragma omp for schedule(static) nowait
        for (int lane = 0; lane < batch; ++lane) advance(lane);
        GENERALS_PROFILE_WORKER_END();
    }
    GENERALS_PROFILE_BATCH_END();
}
