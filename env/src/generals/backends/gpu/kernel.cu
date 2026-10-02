// Independent CUDA implementation of the shared Generals numeric protocol.
// One cooperative thread group owns a board. Economy is parallel; player slots
// are sequential and totals are incremental. Larger groups own a whole block.
#include <cuda_runtime.h>
#include <cstdint>

using u64 = unsigned long long;
using i64 = long long;
#ifndef GENERALS_PACKED_METADATA
#define GENERALS_PACKED_METADATA 0
#endif
#ifndef GENERALS_GROUP_THREADS
#define GENERALS_GROUP_THREADS 32
#endif
#ifndef GENERALS_ENVS_PER_BLOCK
#define GENERALS_ENVS_PER_BLOCK 1
#endif
#ifndef GENERALS_PLAYERS
#define GENERALS_PLAYERS 8
#endif
constexpr int PLAYERS = GENERALS_PLAYERS;
constexpr int THREADS = GENERALS_GROUP_THREADS;
constexpr int ENVS_PER_BLOCK = GENERALS_ENVS_PER_BLOCK;
constexpr int BLOCK_THREADS = THREADS * ENVS_PER_BLOCK;
static_assert(THREADS == 4 || THREADS == 8 || THREADS == 16 || THREADS == 32 ||
              THREADS == 64 || THREADS == 128 || THREADS == 256 ||
              THREADS == 512 || THREADS == 1024);
// Whole-block barriers are uniform because every thread owns the same lane.
static_assert(THREADS <= 32 || ENVS_PER_BLOCK == 1);
static_assert(BLOCK_THREADS % 32 == 0 && BLOCK_THREADS <= 1024);

constexpr bool PACKED = GENERALS_PACKED_METADATA;

// Keep this POD in the same order as kernel._Arguments (no torch ABI).
struct Arguments {
    const uint8_t *terrain;
    uint8_t *structure;
    int16_t *owner;
    i64 *army;
    bool *alive;
    i64 *tick;
    bool *terminated;
    int32_t *winner;
    const uint8_t *initial_structure;
    const int16_t *initial_owner;
    const i64 *initial_army;
    const bool *initial_alive;
    const i64 *initial_tick;
    const bool *initial_terminated;
    const int32_t *initial_winner;
    const i64 *actions;
    i64 *counters;
    i64 *stats;
    i64 *moves;
    int32_t *errors;
    bool *done;
    int32_t batch;
    i64 horizon;
    int32_t auto_reset;
    i64 *cache_stats;
    uint16_t *growth_cells;
    int32_t *growth_count;
    bool *cache_valid;
    int32_t *owned_structures;
};

// Army values, totals and tick counters must fit int64.
struct Shared {
    u64 total[PLAYERS];
    u64 sums[PLAYERS];
    int territory[PLAYERS], cities[PLAYERS];
    int structures[PLAYERS];  // Actual owned structure count, independent of alive/general invariants.
    u64 counters[9];
    i64 stats[PLAYERS * 3], moves[PLAYERS * 9];
    i64 tick;
    int alive_bits, winner, grow_count;
    bool terminated, done[2], reset_needed;
};

__device__ __forceinline__ void barrier() {
#if GENERALS_GROUP_THREADS > 32
    __syncthreads();
#else
    constexpr unsigned group_mask = 0xffffffffu >> (32 - THREADS);
    unsigned shift = (threadIdx.x % 32 / THREADS) * THREADS;
    __syncwarp(group_mask << shift);
#endif
}

__device__ __forceinline__ void skipped(i64 *record) {
    for (int k = 0; k < 9; ++k) record[k] = (k == 2 || k == 3 || k == 6) ? -1 : 0;
}

__device__ __forceinline__ void replace_cell(
    int cell, int new_owner, u64 new_army, int new_structure,
    u64 *army, int16_t *owner, uint8_t *structure, Shared &s, uint32_t *cell_info
) {
    int old_owner = PACKED ? int(int16_t(cell_info[cell] >> 16)) : owner[cell];
    int old_structure = PACKED ? int((cell_info[cell] >> 8) & 255) : structure[cell];
    if (old_owner >= 0) {
        s.total[old_owner] -= army[cell];
        --s.territory[old_owner];
        s.cities[old_owner] -= old_structure == 1;
        s.structures[old_owner] -= old_structure != 0;
    }
    if (new_owner >= 0) {
        s.total[new_owner] += new_army;
        ++s.territory[new_owner];
        s.cities[new_owner] += new_structure == 1;
        s.structures[new_owner] += new_structure != 0;
    }
    army[cell] = new_army;
    owner[cell] = new_owner;
    structure[cell] = new_structure;
#if GENERALS_PACKED_METADATA
    cell_info[cell] = (cell_info[cell] & 255u) | (uint32_t(new_structure) << 8) | (uint32_t(uint16_t(new_owner)) << 16);
#endif
}

__device__ void initialize_totals(
    const u64 *army, const int16_t *owner, const uint8_t *structure,
    int n, int players, Shared &s
) {
    int tid = threadIdx.x % THREADS;
    for (int pid = tid; pid < players; pid += THREADS) {
        s.sums[pid] = 0;
        s.territory[pid] = s.cities[pid] = 0;
        s.structures[pid] = 0;
    }

    barrier();
    for (int cell = tid; cell < n; cell += THREADS) {
        int pid = owner[cell];
        if (pid >= 0) {
            atomicAdd(s.sums + pid, army[cell]);
            atomicAdd(s.territory + pid, 1);
            if (structure[cell] == 1) atomicAdd(s.cities + pid, 1);
            if (structure[cell] != 0) atomicAdd(s.structures + pid, 1);
        }
    }
    barrier();
    for (int pid = tid; pid < players; pid += THREADS) {
        s.total[pid] = s.sums[pid];
    }
    barrier();
}

__global__ void simulate(Arguments a) {
    int group = threadIdx.x / THREADS, tid = threadIdx.x % THREADS;
    int lane = blockIdx.x * ENVS_PER_BLOCK + group;
    if (lane >= a.batch) return;
    constexpr int height = GENERALS_HEIGHT, width = GENERALS_WIDTH, p = PLAYERS;
    const int n = height * width;
    size_t base = size_t(lane) * n, pb = size_t(lane) * p;
    __shared__ Shared slots[ENVS_PER_BLOCK];
    Shared &s = slots[group];
    u64 *army = reinterpret_cast<u64 *>(a.army + base);
    int16_t *owner = a.owner + base;
    uint8_t *structure = a.structure + base;
    const uint8_t *terrain = a.terrain + base;
    uint16_t *grow_cells = a.growth_cells + base;
    uint32_t *cell_info = PACKED ? reinterpret_cast<uint32_t *>(a.growth_cells + ((size_t(a.batch) * n + 1) & ~size_t(1))) + base : nullptr;

    if (tid == 0) {
        s.tick = a.tick[lane]; s.terminated = a.terminated[lane];
        s.winner = a.winner[lane];

        s.alive_bits = 0; s.grow_count = 0;
        for (int pid = 0; pid < p; ++pid) s.alive_bits |= int(a.alive[pb + pid]) << pid;
        s.done[0] = a.done[lane * 2]; s.done[1] = a.done[lane * 2 + 1];
    }
    for (int k = tid; k < 9; k += THREADS) s.counters[k] = a.counters[size_t(lane) * 9 + k];
    barrier();
    if (s.terminated || (!a.auto_reset && a.horizon > 0 && s.tick >= a.horizon)) {
        for (int k = tid; k < p * 9; k += THREADS) {
            int field = k % 9;
            a.moves[pb * 9 + k] = (field == 2 || field == 3 || field == 6) ? -1 : 0;
        }
        return;
    }
    if (a.cache_valid[lane]) {
        if (tid == 0) s.grow_count = a.growth_count[lane];
        for (int pid = tid; pid < p; pid += THREADS) {
            s.total[pid] = u64(a.cache_stats[pb * 3 + pid * 3]);
            s.territory[pid] = int(a.cache_stats[pb * 3 + pid * 3 + 1]);
            s.cities[pid] = int(a.cache_stats[pb * 3 + pid * 3 + 2]);
            s.structures[pid] = a.owned_structures[pb + pid];
        }
    } else {
        for (int cell = tid; cell < n; cell += THREADS) {
            if constexpr (PACKED) cell_info[cell] = uint32_t(terrain[cell]) | (uint32_t(structure[cell]) << 8) | (uint32_t(uint16_t(owner[cell])) << 16);
            if (structure[cell] != 0 && owner[cell] >= 0)
                grow_cells[atomicAdd(&s.grow_count, 1)] = cell;
        }
        barrier();
        initialize_totals(army, owner, structure, n, p, s);
    }
    barrier();

    // Snapshot control before the leader may advance tick. This barrier is
    // required even within a warp with independent thread scheduling.
    i64 current_tick = s.tick;
    barrier();
    i64 next_tick = current_tick + 1;
    if ((next_tick & 1) == 0) {
        bool all_land = next_tick % 50 == 0;
        int count = all_land ? n : s.grow_count;
        for (int k = tid; k < count; k += THREADS) {
            int cell = all_land ? k : grow_cells[k];
            if (!all_land || owner[cell] >= 0) {
                ++army[cell];
            }
        }
        barrier();
        if (tid == 0) for (int pid = 0; pid < p; ++pid) {
            s.total[pid] += all_land ? s.territory[pid] : s.structures[pid];
        }
    }
    if (tid == 0) {
        s.tick = next_tick;
        for (int pid = 0; pid < p; ++pid) {
            i64 *r = s.moves + pid * 9;
            skipped(r);
            bool active = (s.alive_bits & (1 << pid)) && !s.terminated;
            if (!active) continue;
            r[0] = 1;
            i64 action = a.actions[pb + pid];

            ++s.counters[1];
            if (!action) {
                r[1] = 1; r[8] = 1; ++s.counters[3];
                continue;
            }
            ++s.counters[2];
            // Unsigned subtraction explicitly preserves the protocol's wrap
            // for malformed INT64_MIN actions, without signed C++ overflow.
            i64 encoded = i64(u64(action) - 1);
            i64 source = encoded / 8;
            int direction = (encoded % 8) / 2;
            bool half = encoded % 2 == 1;
            r[2] = source;
            bool in_bounds = action > 0 && source >= 0 && source < n;
            int target = -1;
            if (in_bounds) {
                int sy = int(source) / width, sx = int(source) % width;
                int ty = sy + (direction == 0 ? -1 : direction == 2 ? 1 : 0);
                int tx = sx + (direction == 3 ? -1 : direction == 1 ? 1 : 0);
                in_bounds = ty >= 0 && ty < height && tx >= 0 && tx < width;
                if (in_bounds) target = ty * width + tx;
            }
            r[3] = target;
            int reason;
#if GENERALS_PACKED_METADATA
            reason = !in_bounds ? 3 : int16_t(cell_info[source] >> 16) != pid ? 4 : (cell_info[target] & 255) == 1 ? 5 : 2;
#else
            reason = !in_bounds ? 3 : owner[source] != pid ? 4 : terrain[target] == 1 ? 5 : 2;
#endif
            r[8] = reason;
            if (reason != 2) { ++s.counters[4]; continue; }
            r[1] = 1;
            u64 source_army = army[source];
            u64 moved = half ? source_army / 2 : source_army > 0 ? source_army - 1 : 0;
            r[4] = moved;
#if GENERALS_PACKED_METADATA
            uint32_t target_info = cell_info[target];
            int defender = int16_t(target_info >> 16), target_structure = (target_info >> 8) & 255;
#else
            int defender = owner[target], target_structure = structure[target];
#endif
            u64 defense = army[target];
            bool friendly = defender == pid;
            bool captured = !friendly && moved > defense;
            u64 next_defense = friendly ? defense + moved : captured ? moved - defense : defense - moved;
            army[source] = source_army - moved;
            if (friendly) {
                army[target] = next_defense;
            } else {
                s.total[pid] -= moved;
                if (captured) {
                    if (defender < 0 && target_structure != 0) grow_cells[s.grow_count++] = target;
                    replace_cell(target, pid, next_defense, target_structure,
                                 army, owner, structure, s, cell_info);
                } else {
                    army[target] = next_defense;
                    if (defender >= 0) s.total[defender] -= moved;
                }
            }

            r[5] = captured; r[7] = friendly || captured;
            if (captured && target_structure == 2) {
                r[6] = defender;
                s.alive_bits &= ~(1 << defender);
                structure[target] = 1; ++s.cities[pid];
#if GENERALS_PACKED_METADATA
                cell_info[target] = (cell_info[target] & 0xffff00ffu) | 256u;
#endif
                for (int cell = 0; cell < n; ++cell) {
                    if (owner[cell] == defender) {
                        u64 inherited = army[cell] / 2;
                        replace_cell(cell, pid, inherited ? inherited : 1, structure[cell],
                                      army, owner, structure, s, cell_info);
                    }
                }
                if ((s.alive_bits & (s.alive_bits - 1)) == 0) {
                    s.terminated = true; s.winner = pid;
                }
            }
            s.counters[5] += moved > 0; s.counters[6] += captured;
        }
        for (int pid = 0; pid < p; ++pid) {
            s.stats[pid * 3] = i64(s.total[pid]);
            s.stats[pid * 3 + 1] = s.territory[pid];
            s.stats[pid * 3 + 2] = s.cities[pid];
        }
        ++s.counters[0];
        bool truncated = !s.terminated && a.horizon > 0 && s.tick >= a.horizon;
        s.done[0] = s.terminated; s.done[1] = truncated;
        s.counters[7] += s.terminated; s.counters[8] += truncated;
        s.reset_needed = a.auto_reset && (s.terminated || truncated);
    }
    barrier();
    if (s.reset_needed) {
        // Tensor callers may have edited structures before this launch.
        // Restore the growth index along with the cached initial board.
        if (tid == 0) s.grow_count = 0;
        barrier();
        for (int cell = tid; cell < n; cell += THREADS) {
            army[cell] = a.initial_army[base + cell];
            owner[cell] = a.initial_owner[base + cell];
            structure[cell] = a.initial_structure[base + cell];
#if GENERALS_PACKED_METADATA
            cell_info[cell] = uint32_t(terrain[cell]) | (uint32_t(structure[cell]) << 8) | (uint32_t(uint16_t(owner[cell])) << 16);
#endif
            if (structure[cell] != 0 && owner[cell] >= 0)
                grow_cells[atomicAdd(&s.grow_count, 1)] = cell;
        }
        if (tid == 0) {
            s.alive_bits = 0;
            for (int pid = 0; pid < p; ++pid) s.alive_bits |= int(a.initial_alive[pb + pid]) << pid;
            s.tick = a.initial_tick[lane]; s.terminated = a.initial_terminated[lane];
            s.winner = a.initial_winner[lane];
        }
        barrier();
    }

    // Public statistics double as the cache until reset. Reset invalidates
    // it; the next step rebuilds current statistics from the restored board.
    for (int pid = tid; pid < p; pid += THREADS) {
        a.owned_structures[pb + pid] = s.structures[pid];
    }
    if (tid == 0) {
        a.growth_count[lane] = s.grow_count;
        a.cache_valid[lane] = !s.reset_needed;
    }
    for (int k = tid; k < p * 9; k += THREADS) a.moves[pb * 9 + k] = s.moves[k];
    for (int k = tid; k < p * 3; k += THREADS) {
        a.stats[pb * 3 + k] = s.stats[k];
        // Inference tensors have no version counters, so their public output
        // stays separate from the private cache to tolerate output edits.
        if (a.cache_stats != a.stats) a.cache_stats[pb * 3 + k] = s.stats[k];
    }
    for (int pid = tid; pid < p; pid += THREADS) a.alive[pb + pid] = (s.alive_bits >> pid) & 1;
    if (tid == 0) {
        a.tick[lane] = s.tick; a.terminated[lane] = s.terminated; a.winner[lane] = s.winner;
        a.errors[lane] = 0;
        a.done[lane * 2] = s.done[0]; a.done[lane * 2 + 1] = s.done[1];
    }
    for (int k = tid; k < 9; k += THREADS) a.counters[size_t(lane) * 9 + k] = i64(s.counters[k]);
}

extern "C" int generals_launch(const Arguments *a, void *stream, int *resources) {
    if (resources) {
        cudaFuncAttributes attr;
        cudaError_t status = cudaFuncGetAttributes(&attr, simulate);
        if (status != cudaSuccess) return status;
        int blocks;
        status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, simulate, BLOCK_THREADS, 0);
        resources[0] = attr.numRegs; resources[1] = int(attr.localSizeBytes);
        resources[2] = int(attr.sharedSizeBytes); resources[3] = blocks;
        return status;
    }
    simulate<<<(a->batch + ENVS_PER_BLOCK - 1) / ENVS_PER_BLOCK, BLOCK_THREADS, 0,
               cudaStream_t(stream)>>>(*a);
    return cudaGetLastError();
}

extern "C" const char *generals_error(int status) { return cudaGetErrorString(cudaError_t(status)); }
