// Borrowed-buffer ABI for synchronous native backend/policy execution.
// No Python objects, callbacks, ownership transfers or policy-specific dispatch.
// Owners must keep storage, instances and libraries alive and exclusively leased
// until execution returns. Observation pointers expire after each act call.
#pragma once
#include <cstdint>

constexpr std::uint64_t GENERALS_NATIVE_ABI = 2;
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
    // Optional serial lane entry; the same metrics/watchdog contract for every policy.
    std::int64_t (*act_one)(void *, const GeneralsBatchView *, int, std::int64_t *);

};
struct GeneralsBackend {
    std::uint64_t abi;
    void *context;
    void (*observe)(void *, const std::int64_t *, GeneralsObservation *);
    void (*step)(void *, const std::int64_t *);
    const bool *alive, *done;
    const std::int64_t *tick, *moves;
    const std::int32_t *errors;
    void (*depart)(void *, int, const bool *);
    const void *state;
    const std::int64_t *stats;
    void (*legal)(void *, int, bool *);
};
