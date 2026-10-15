// One thread owns an environment; neighboring cells share 16-byte records.
// Incremental totals, owned-land and structure growth indices. Exactly one tick;
// no policy or RNG, all public outputs materialized on every call.
#include <metal_stdlib>
using namespace metal;
constant int N = W * H;
using i64 = long;
using u64 = ulong;

struct Cell {
    i64 army;
    short owner;
    uchar structure;
    uchar terrain;
};
static_assert(sizeof(Cell) == 16, "cell tensor views require 16-byte records");

struct Buffers {
    device Cell *cells;
    device bool *alive;
    device i64 *tick;
    device bool *terminated;
    device int *winner;
    device const Cell *initial_cells;
    device const bool *initial_alive;
    device const i64 *initial_tick;
    device const bool *initial_terminated;
    device const int *initial_winner;
    device const i64 *actions;
    device i64 *counters;
    device i64 *stats;
    device i64 *moves;
    device int *errors;
    device bool *done;
    device i64 *cache;
    device ushort *growth;
    device int *aux;
    device ushort *owned;
};

struct Totals {
    u64 army[P];
    int territory[P], cities[P], structures[P];
    int growth, owned;
};

inline ulong at(int field, uint lane) { return ulong(field) * B + lane; }
inline ulong board(int cell, uint lane) { return ulong(lane) * N + cell; }

inline void rebuild(thread const Buffers &a, thread Totals &t, uint lane) {
    for (int pid = 0; pid < P; ++pid) {
        t.army[pid] = 0;
        t.territory[pid] = t.cities[pid] = t.structures[pid] = 0;
    }
    t.growth = t.owned = 0;
    for (int cell = 0; cell < N; ++cell) {
        ulong idx = board(cell, lane);
        int pid = a.cells[idx].owner;
        if (pid >= 0) {
            a.owned[board(t.owned++, lane)] = cell;
            t.army[pid] += u64(a.cells[idx].army);
            ++t.territory[pid];
            t.cities[pid] += a.cells[idx].structure == 1;
            t.structures[pid] += a.cells[idx].structure != 0;
            if (a.cells[idx].structure) a.growth[at(t.growth++, lane)] = cell;
        }
    }
}

inline void replace_cell(thread const Buffers &a, thread Totals &t, uint lane,
                         int cell, int pid, u64 army, int kind) {
    ulong idx = board(cell, lane);
    int old = a.cells[idx].owner, old_kind = a.cells[idx].structure;
    if (old >= 0) {
        t.army[old] -= u64(a.cells[idx].army);
        --t.territory[old];
        t.cities[old] -= old_kind == 1;
        t.structures[old] -= old_kind != 0;
    }
    if (pid >= 0) {
        t.army[pid] += army;
        ++t.territory[pid];
        t.cities[pid] += kind == 1;
        t.structures[pid] += kind != 0;
    }
    a.cells[idx].army = i64(army); a.cells[idx].owner = pid; a.cells[idx].structure = kind;
}

inline void skipped(thread const Buffers &a, uint lane) {
    for (int k = 0; k < P * 9; ++k) {
        int f = k % 9;
        a.moves[at(k, lane)] = (f == 2 || f == 3 || f == 6) ? -1 : 0;
    }
}

inline void simulate(thread const Buffers &a, uint lane, i64 horizon, int flags) {
    if (lane >= B) return;
    bool auto_reset = flags & 1;
    i64 tick = a.tick[lane];
    bool terminated = a.terminated[lane];
    int winner = a.winner[lane], alive = 0;
    for (int pid = 0; pid < P; ++pid) alive |= int(a.alive[at(pid, lane)]) << pid;
    int valid = (flags & 2) ? 0 : a.aux[at(1, lane)];
    Totals t;
    if (valid & 1) {
        t.growth = a.aux[lane];
        t.owned = a.aux[at(2, lane)];
        for (int pid = 0; pid < P; ++pid) {
            t.army[pid] = u64(a.cache[at(pid * 3, lane)]);
            t.territory[pid] = int(a.cache[at(pid * 3 + 1, lane)]);
            t.cities[pid] = int(a.cache[at(pid * 3 + 2, lane)]);
            t.structures[pid] = a.aux[at(3 + pid, lane)];
        }
    } else rebuild(a, t, lane);
    u64 counters[9];
    for (int k = 0; k < 9; ++k) counters[k] = u64(a.counters[at(k, lane)]);
    if (!(terminated || (!auto_reset && horizon > 0 && tick >= horizon))) {
        i64 actions[P];
        for (int pid = 0; pid < P; ++pid) actions[pid] = a.actions[ulong(lane) * P + pid];
        ++tick;
        if ((tick & 1) == 0) {
            bool all = tick % 50 == 0;
            int count = all ? t.owned : t.growth;
            for (int k = 0; k < count; ++k) {
                int cell = all ? a.owned[board(k, lane)] : a.growth[at(k, lane)];
                ulong idx = board(cell, lane);
                a.cells[idx].army = i64(u64(a.cells[idx].army) + 1);
            }
            for (int pid = 0; pid < P; ++pid) t.army[pid] += all ? t.territory[pid] : t.structures[pid];
        }
        for (int pid = 0; pid < P; ++pid) {
            i64 r[9] = {0, 0, -1, -1, 0, 0, -1, 0, 0};
            if ((alive & (1 << pid)) && !terminated) {
                r[0] = 1; ++counters[1];
                i64 action = actions[pid];
                if (action == 0) { r[1] = 1; r[8] = 1; ++counters[3]; }
                else {
                    ++counters[2];
                    i64 encoded = as_type<i64>(as_type<u64>(action) - 1);
                    i64 source = encoded / 8;
                    int direction = int((encoded % 8) / 2);
                    bool half_move = encoded % 2 == 1;
                    bool inside = action > 0 && source >= 0 && source < N;
                    int target = -1;
                    if (inside) {
                        int y = int(source) / W, x = int(source) % W;
                        int ty = y + (direction == 0 ? -1 : direction == 2 ? 1 : 0);
                        int tx = x + (direction == 3 ? -1 : direction == 1 ? 1 : 0);
                        inside = ty >= 0 && ty < H && tx >= 0 && tx < W;
                        if (inside) target = ty * W + tx;
                    }
                    r[2] = source; r[3] = target;
                    int reason = !inside ? 3 : a.cells[board(int(source), lane)].owner != pid ? 4 :
                                 a.cells[board(target, lane)].terrain == 1 ? 5 : 2;
                    r[8] = reason;
                    if (reason != 2) ++counters[4];
                    else {
                        r[1] = 1;
                        ulong src = board(int(source), lane), dst = board(target, lane);
                        u64 count = u64(a.cells[src].army);
                        u64 moved = half_move ? count / 2 : count > 0 ? count - 1 : 0;
                        r[4] = i64(moved);
                        int defender = a.cells[dst].owner, kind = a.cells[dst].structure;
                        u64 defense = u64(a.cells[dst].army);
                        bool friendly = defender == pid, captured = !friendly && moved > defense;
                        u64 next = friendly ? defense + moved : captured ? moved - defense : defense - moved;
                        a.cells[src].army = i64(count - moved);
                        if (friendly) {
                            a.cells[dst].army = i64(next);
                        } else {
                            t.army[pid] -= moved;
                            if (captured) {
                                if (defender < 0) {
                                    a.owned[board(t.owned++, lane)] = target;
                                    if (kind) a.growth[at(t.growth++, lane)] = target;
                                }
                                replace_cell(a, t, lane, target, pid, next, kind);
                            } else {
                                a.cells[dst].army = i64(next);
                                if (defender >= 0) t.army[defender] -= moved;
                            }
                        }
                        r[5] = captured; r[7] = friendly || captured;
                        if (captured && kind == 2) {
                            r[6] = defender; alive &= ~(1 << defender);
                            a.cells[dst].structure = 1; ++t.cities[pid];
                            for (int k = 0; k < t.owned; ++k) {
                                int cell = a.owned[board(k, lane)];
                                ulong idx = board(cell, lane);
                                if (a.cells[idx].owner == defender) {
                                    u64 inherited = u64(a.cells[idx].army) / 2;
                                    replace_cell(a, t, lane, cell, pid, inherited ? inherited : 1, a.cells[idx].structure);
                                }
                            }
                            if ((alive & (alive - 1)) == 0) { terminated = true; winner = pid; }
                        }
                        counters[5] += moved > 0; counters[6] += captured;
                    }
                }
            }
            for (int k = 0; k < 9; ++k) a.moves[at(pid * 9 + k, lane)] = r[k];
        }
        for (int pid = 0; pid < P; ++pid) {
            a.stats[at(pid * 3, lane)] = i64(t.army[pid]);
            a.stats[at(pid * 3 + 1, lane)] = t.territory[pid];
            a.stats[at(pid * 3 + 2, lane)] = t.cities[pid];
        }
        ++counters[0];
        bool truncated = !terminated && horizon > 0 && tick >= horizon;
        a.done[lane] = terminated; a.done[at(1, lane)] = truncated;
        counters[7] += terminated; counters[8] += truncated;
        if (auto_reset && (terminated || truncated)) {
            for (int cell = 0; cell < N; ++cell) {
                ulong idx = board(cell, lane);
                a.cells[idx].structure = a.initial_cells[idx].structure;
                a.cells[idx].owner = a.initial_cells[idx].owner;
                a.cells[idx].army = a.initial_cells[idx].army;
            }
            rebuild(a, t, lane);
            alive = 0;
            for (int pid = 0; pid < P; ++pid) alive |= int(a.initial_alive[at(pid, lane)]) << pid;
            tick = a.initial_tick[lane]; terminated = a.initial_terminated[lane];
            winner = a.initial_winner[lane];
        }
    } else skipped(a, lane);
    for (int pid = 0; pid < P; ++pid) {
        a.cache[at(pid * 3, lane)] = i64(t.army[pid]);
        a.cache[at(pid * 3 + 1, lane)] = t.territory[pid];
        a.cache[at(pid * 3 + 2, lane)] = t.cities[pid];
        a.aux[at(3 + pid, lane)] = t.structures[pid];
        a.alive[at(pid, lane)] = (alive >> pid) & 1;
    }
    a.aux[lane] = t.growth; a.aux[at(1, lane)] = 1;
    a.aux[at(2, lane)] = t.owned;
    a.tick[lane] = tick; a.terminated[lane] = terminated; a.winner[lane] = winner;
    a.errors[lane] = 0;
    for (int k = 0; k < 9; ++k) a.counters[at(k, lane)] = i64(counters[k]);
}

kernel void step(device uchar *state, device const uchar *initial,
                 device uchar *output, device uchar *cache,
                 device const i64 *actions, constant i64 &horizon,
                 constant i64 &flags, uint lane [[thread_position_in_grid]]) {
    Buffers a = {
        reinterpret_cast<device Cell *>(state + OFF_STATE_ARMY),
        reinterpret_cast<device bool *>(state + OFF_STATE_ALIVE),
        reinterpret_cast<device i64 *>(state + OFF_STATE_TICK),
        reinterpret_cast<device bool *>(state + OFF_STATE_TERMINATED),
        reinterpret_cast<device int *>(state + OFF_STATE_WINNER),
        reinterpret_cast<device const Cell *>(initial + OFF_INITIAL_ARMY),
        reinterpret_cast<device const bool *>(initial + OFF_INITIAL_ALIVE),
        reinterpret_cast<device const i64 *>(initial + OFF_INITIAL_TICK),
        reinterpret_cast<device const bool *>(initial + OFF_INITIAL_TERMINATED),
        reinterpret_cast<device const int *>(initial + OFF_INITIAL_WINNER),
        actions,
        reinterpret_cast<device i64 *>(output + OFF_OUTPUT_COUNTERS),
        reinterpret_cast<device i64 *>(output + OFF_OUTPUT_STATS),
        reinterpret_cast<device i64 *>(output + OFF_OUTPUT_LAST_MOVES),
        reinterpret_cast<device int *>(output + OFF_OUTPUT_ERRORS),
        reinterpret_cast<device bool *>(output + OFF_OUTPUT_DONE),
        reinterpret_cast<device i64 *>(cache + OFF_CACHE_TOTALS),
        reinterpret_cast<device ushort *>(cache + OFF_CACHE_GROWTH),
        reinterpret_cast<device int *>(cache + OFF_CACHE_AUX),
        reinterpret_cast<device ushort *>(cache + OFF_CACHE_OWNED)
    };
    simulate(a, lane, horizon, int(flags));
}
