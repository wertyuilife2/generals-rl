// Interface example only: a deliberately inert policy, not a baseline.
#include "arena_policy.h"
struct Policy {};
extern "C" void *arena_create(int, std::uint64_t, const char *) { return new Policy; }
extern "C" std::int64_t arena_act(void *, const ArenaObservation *) { return 0; }
extern "C" void arena_destroy(void *p) { delete static_cast<Policy *>(p); }
