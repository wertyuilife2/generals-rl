// Borrowed-buffer ABI for synchronous native backend/policy execution.
// No Python objects, callbacks, ownership transfers or policy-specific dispatch.
// Owners must keep storage, instances and libraries alive and exclusively leased
// until execution returns. Observation pointers expire after each act call.
#pragma once
#include <cstdint>

constexpr std::uint64_t GENERALS_NATIVE_ABI = 3;
// Current executor reads host buffers. A device backend may stage them and
// provide synchronize(); device-only pointers must never masquerade as host data.
constexpr std::uint64_t GENERALS_HOST_BUFFERS = 1;
constexpr std::uint64_t GENERALS_TIMED_POLICY = 1;
enum GeneralsDevice { GENERALS_CPU=0, GENERALS_CUDA=1, GENERALS_MPS=2 };
// Read-only, row-major state projection. No backend implementation types cross
// this ABI. The owner retains storage for the lease; refresh after mutations.
struct GeneralsStateView {
    const std::uint8_t *terrain, *structure;
    const std::int16_t *owner;
    const std::int64_t *army;
    const bool *alive;
    const std::int32_t *general_pos;
    const std::int64_t *tick;
    const bool *terminated;
    const std::int32_t *winner;
    const std::int64_t *idle_ticks, *afk_turn;
    const std::int32_t *end_reason;
    const std::int64_t *leave_tick;
    const std::int32_t *death_order, *kills;
    const std::int64_t *damage;
    const std::int32_t *spawn_pos;
    const std::uint8_t *city_known, *capture_mask, *warn_afk;
};
struct GeneralsObservation {
    std::uint8_t *terrain, *structure;
    std::int16_t *owner;
    std::int64_t *army;
    bool *visible, *alive, *fog_obstacles, *has_kill, *warn_afk;
    double *neutralize_in;
    std::int64_t *stats;
    std::int32_t *general_pos;
    bool *active;
    std::int64_t *tick;
    bool *terminated;
    std::int32_t *winner_id;
};
struct GeneralsBatchView {
    int count, height, width, players;
    const std::int64_t *player_ids;
    GeneralsObservation observation;
};
struct GeneralsFeedback {
    int count;
    const std::int64_t *moves, *actions;
    const bool *active;
};
struct GeneralsPolicy {
    std::uint64_t abi;
    void *context;
    // -1 = success, otherwise failed lane. The owner supplies error details.
    std::int64_t (*act)(void *, const GeneralsBatchView *, std::int64_t *);
    void (*feedback)(void *, const GeneralsFeedback *);
    // Optional raw serial lane entry. The executor owns its common watchdog;
    // callees report errors in metrics.status and must not arm another timer.
    std::int64_t (*act_one)(void *, const GeneralsBatchView *, int, std::int64_t *);
    std::uint64_t flags;
};
struct GeneralsBackend {
    std::uint64_t abi;
    void *context;
    // Null mask selects every lane; unselected payloads are left untouched and
    // marked inactive. A policy must not act on inactive rows.
    void (*observe)(void *, const std::int64_t *, const bool *, GeneralsObservation *);
    void (*step)(void *, const std::int64_t *);
    const bool *alive, *done;
    const std::int64_t *tick, *moves;
    const std::int32_t *errors;
    void (*depart)(void *, int, const bool *);
    void (*snapshot)(void *, GeneralsStateView *);
    const std::int64_t *stats;
    void (*legal)(void *, int, bool *);
    std::uint64_t capabilities;
    std::int32_t device;
    // Optional completion barrier for queued commands. All exported host views
    // are readable after it returns; synchronous implementations leave it null.
    void (*synchronize)(void *);
};
