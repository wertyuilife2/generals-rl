// Device-resident Metal execution of the same rules.h used by CPU and CUDA.
// A thread owns one board; tensor packing changes addressing, never game rules.
#include <metal_stdlib>
using namespace metal;
using generals::I;
using generals::U;
constant int N = W * H;
inline ulong at(int field, uint lane) { return ulong(field) * B + lane; }

kernel void step(device uchar *state, device uchar *initial,
                 device uchar *output, device uchar *cache_storage,
                 device const I *actions, constant I &horizon,
                 constant I &flags, uint lane [[thread_position_in_grid]]) {
    if (lane >= B) return;
    auto s = state_batch(state), initial_s = initial_batch(initial);
    device I *stats = reinterpret_cast<device I *>(output + OFF_OUTPUT_STATS);
    device I *moves = reinterpret_cast<device I *>(output + OFF_OUTPUT_LAST_MOVES);
    device I *counts = reinterpret_cast<device I *>(output + OFF_OUTPUT_COUNTERS);
    device bool *done = reinterpret_cast<device bool *>(output + OFF_OUTPUT_DONE);
    device int *errors = reinterpret_cast<device int *>(output + OFF_OUTPUT_ERRORS);
    device I *totals = reinterpret_cast<device I *>(cache_storage + OFF_CACHE_TOTALS);
    device int *aux = reinterpret_cast<device int *>(cache_storage + OFF_CACHE_AUX);
    device ushort *growth_data = reinterpret_cast<device ushort *>(cache_storage + OFF_CACHE_GROWTH);
    bool auto_reset = flags & 1;
    if (s.terminated[lane] || (!auto_reset && horizon > 0 && s.tick[lane] >= horizon)) {
        for (int k = 0; k < P * 9; ++k) {
            int f = k % 9;
            moves[at(k, lane)] = (f == 2 || f == 3 || f == 6) ? -1 : 0;
        }
        return;
    }
    generals::Cache c;
    generals::Array<ushort> growth{growth_data + lane, N, N * B, B, 0};
    generals::Rules<> rules(s, lane, P, H, W, c, growth);
    if (!(flags & 2) && aux[at(1, lane)]) {
        c.growth_count = aux[lane];
        for (int pid = 0; pid < P; ++pid) {
            c.total[pid] = U(totals[at(pid * 3, lane)]);
            c.territory[pid] = int(totals[at(pid * 3 + 1, lane)]);
            c.cities[pid] = int(totals[at(pid * 3 + 2, lane)]);
            c.structures[pid] = aux[at(pid + 2, lane)];
        }
    } else {
        c.growth_count = 0;
        for (int pid = 0; pid < P; ++pid)
            c.total[pid] = c.territory[pid] = c.cities[pid] = c.structures[pid] = 0;
        for (int cell = 0; cell < N; ++cell) {
            int pid = rules.owner[cell], kind = rules.structure[cell];
            if (pid < 0) continue;
            c.total[pid] += rules.army[cell];
            ++c.territory[pid];
            c.cities[pid] += kind == 1;
            if (kind) { ++c.structures[pid]; growth[c.growth_count++] = cell; }
        }
    }
    I joint[P], records[P * 9]; U counters[9];
    for (int pid = 0; pid < P; ++pid) joint[pid] = actions[ulong(lane) * P + pid];
    for (int k = 0; k < 9; ++k) counters[k] = U(counts[at(k, lane)]);
    rules.before_growth(joint, records, counters);
    I tick = s.tick[lane];
    if (tick % 2 == 0) {
        for (int k = 0; k < c.growth_count; ++k) ++rules.army[growth[k]];
        for (int pid = 0; pid < P; ++pid) c.total[pid] += c.structures[pid];
    }
    if (tick % 50 == 0) {
        for (int cell = 0; cell < N; ++cell) if (rules.owner[cell] >= 0) ++rules.army[cell];
        for (int pid = 0; pid < P; ++pid) c.total[pid] += c.territory[pid];
    }
    rules.after_growth();
    ++counters[0];
    bool terminal = s.terminated[lane], truncated = !terminal && horizon > 0 && tick >= horizon;
    done[lane] = terminal; done[at(1, lane)] = truncated;
    counters[7] += terminal; counters[8] += truncated;
    for (int pid = 0; pid < P; ++pid) {
        stats[at(pid * 3, lane)] = totals[at(pid * 3, lane)] = I(c.total[pid]);
        stats[at(pid * 3 + 1, lane)] = totals[at(pid * 3 + 1, lane)] = c.territory[pid];
        stats[at(pid * 3 + 2, lane)] = totals[at(pid * 3 + 2, lane)] = c.cities[pid];
        aux[at(pid + 2, lane)] = c.structures[pid];
    }
    aux[lane] = c.growth_count;
    bool reset = auto_reset && (terminal || truncated);
    aux[at(1, lane)] = !reset;
    if (reset) {
        size_t base = size_t(lane) * N;
        for (int cell = 0; cell < N; ++cell) {
            rules.owner[cell] = initial_s.owner[base + cell];
            rules.army[cell] = U(initial_s.army[base + cell]);
            rules.structure[cell] = initial_s.structure[base + cell];
        }
        generals::reset_metadata(s, initial_s, lane, P);
    }
    errors[lane] = 0;
    for (int k = 0; k < P * 9; ++k) moves[at(k, lane)] = records[k];
    for (int k = 0; k < 9; ++k) counts[at(k, lane)] = I(counters[k]);
}
