// Arena's inert rating reference. Uses the same native policy interface.
#include "arena_policy.h"
struct WaitPolicy {};
extern "C" void *arena_create(int, std::uint64_t, const char *) { return new WaitPolicy; }
extern "C" std::int64_t arena_act(void *, const ArenaObservation *) { return 0; }
extern "C" void arena_destroy(void *policy) { delete static_cast<WaitPolicy *>(policy); }
