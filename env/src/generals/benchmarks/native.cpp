// Benchmark-only action sampling and opt-in instrumentation of the REAL CPU step.
// No second implementation of the game rules and no source-string rewriting.
#include <algorithm>
#include <chrono>
#include <vector>
#include <array>
#include <cstdint>
#include <omp.h>

namespace profiling {
using Clock = std::chrono::steady_clock;
static double now() {
    return double(std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count());
}
struct alignas(64) Worker { // Keep instrumentation counters off other workers' cache lines.
    double start = 0, elapsed = 0, previous = 0;
    std::array<double, 7> phases{};
    int samples = 0, phase = 0;
    bool sampled = false;
};
static std::vector<Worker> workers;
static std::vector<std::array<double, 15>> records;
static int index = 0, stride = 32, batch_size = 0;
static double started = 0;
static thread_local Worker *worker = nullptr;
static void begin_batch(int batch, int threads) {
    batch_size = batch;
    for (auto &entry : workers) entry = Worker{};
    started = now();
}
static void begin_worker() {
    worker = &workers[omp_get_thread_num()];
    worker->start = now();
}
static void end_worker() { worker->elapsed = now() - worker->start; }
static void begin_lane(int lane) {
    worker->sampled = lane % stride == index % stride;
    if (worker->sampled) {
        worker->previous = now(); worker->phase = 0; ++worker->samples;
    }
}
static void mark(int phase) {
    if (!worker->sampled) return;
    double at = now();
    worker->phases[phase] += at - worker->previous;
    worker->previous = at; worker->phase = phase + 1;
}
static void end_lane(int lane) {
    if (worker->sampled && worker->phase < 6) mark(6); // Early return / error.
}
static void end_batch() {
    double elapsed = now() - started;
    auto &row = records.at(index++);
    row[0] = elapsed;
    row[3] = elapsed;
    int active = 0;
    double first_start = started + elapsed, last_start = started, last_end = started;
    for (const auto &entry : workers) {
        if (!entry.start) continue;
        ++active;
        first_start = std::min(first_start, entry.start);
        last_start = std::max(last_start, entry.start);
        last_end = std::max(last_end, entry.start + entry.elapsed);
        row[1] = std::max(row[1], entry.elapsed);
        row[2] += entry.elapsed;
        row[3] = std::min(row[3], entry.elapsed);
        row[4] += entry.samples;
        for (int phase = 0; phase < 7; ++phase) row[5 + phase] += entry.phases[phase];
    }
    if (active) row[2] /= active;
    row[12] = first_start - started;
    row[13] = last_start - first_start;
    row[14] = started + elapsed - last_end;
}
}

#define GENERALS_PROFILE_BATCH_BEGIN(batch, threads) profiling::begin_batch(batch, threads)
#define GENERALS_PROFILE_BATCH_END() profiling::end_batch()
#define GENERALS_PROFILE_WORKER_BEGIN() profiling::begin_worker()
#define GENERALS_PROFILE_WORKER_END() profiling::end_worker()
#define GENERALS_PROFILE_LANE_BEGIN(lane) profiling::begin_lane(lane)
#define GENERALS_PROFILE_LANE_END(lane) profiling::end_lane(lane)
#define GENERALS_PROFILE_MARK(phase) profiling::mark(phase)
#include "../backends/cpu/step.cpp"

extern "C" double *generals_profile_begin(int steps, int threads, int stride) {
    profiling::records.assign(steps, {});
    profiling::workers.assign(threads, {});
    profiling::index = 0; profiling::stride = stride;
    return profiling::records[0].data();
}

static std::uint32_t bounded(std::uint32_t &state, std::uint32_t count) {
    const auto threshold = std::uint32_t(-count) % count;
    do { state ^= state << 13; state ^= state >> 17; state ^= state << 5; }
    while (state < threshold);
    return state % count;
}

// All players choose from the same PRE-TICK state, like a batched RL policy.
// Reservoir sampling is uniform over owned cells with >1 army. The policy has
// its own RNG, never touching the environment's RNG. Never timed as simulation.
extern "C" void generals_sample_actions(const Batch *s, std::uint32_t *rng, i64 *actions,
                                        int batch, int h, int w, int players, int threads) {
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int lane = 0; lane < batch; ++lane) {
        auto &random = rng[lane];
        const int n = h * w;
        const std::size_t base = std::size_t(lane) * n, pb = std::size_t(lane) * players;
        int selected[8]{}, count[8]{};
        for (int cell = 0; cell < n; ++cell) {
            int pid = s->owner[base + cell];
            if (pid >= 0 && s->army[base + cell] > 1 && s->alive[pb + pid])
                if (bounded(random, ++count[pid]) == 0) selected[pid] = cell;
        }
        for (int pid = 0; pid < players; ++pid) {
            actions[pb + pid] = 0;
            if (!count[pid] || s->terminated[lane]) continue;
            int cell = selected[pid], y = cell / w, x = cell % w, directions[4], size = 0;
            const int targets[4] = {cell - w, cell + 1, cell + w, cell - 1};
            const bool inside[4] = {y > 0, x + 1 < w, y + 1 < h, x > 0};
            for (int d = 0; d < 4; ++d)
                if (inside[d] && s->terrain[base + targets[d]] != 1) directions[size++] = d;
            if (size) {
                int direction = directions[bounded(random, size)];
                int half = bounded(random, 2);
                actions[pb + pid] = 1 + cell * 8 + direction * 2 + half;
            }
        }
    }
}
