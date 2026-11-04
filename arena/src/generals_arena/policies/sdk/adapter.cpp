// Linked with every source file of one policy. Calls are serial, with no threads.
#include "arena_policy.h"
#include "decision.h"
using generals_decision::Metrics;
using generals_decision::Guard;

#define EXPORT extern "C" __attribute__((visibility("default")))
EXPORT void *arena_host_create(int player,std::uint64_t seed,const char *params,Metrics *m) {
    Guard guard(m,2000000000,false);
    void *instance=nullptr; int status=0;
    try { instance=arena_create(player,seed,params); if(!instance) status=2; }
    catch(...) { status=2; }
    guard.done(status); return instance;
}
EXPORT std::int64_t arena_host_act(void *instance,const ArenaObservation *o,Metrics *m) {
    Guard guard(m,10000000,true,o->tick);
    std::int64_t action=0; int status=0;
    try { action=arena_act(instance,o); } catch(...) { status=2; }
    guard.done(status); return action;
}
// Native Arena supplies the watchdog once, outside this adapter.
EXPORT std::int64_t arena_host_decide(void *instance,const ArenaObservation *o,Metrics *m) {
    try {return arena_act(instance,o);} catch(...) {m->status=2;return 0;}
}
EXPORT void arena_host_destroy(void *instance,Metrics *m) {
    Guard guard(m,2000000000,false);
    int status=0;
    try { arena_destroy(instance); } catch(...) { status=2; }
    guard.done(status);
}
