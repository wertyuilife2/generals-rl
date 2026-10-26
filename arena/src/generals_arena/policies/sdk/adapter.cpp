// Linked with every source file of one policy. Calls are serial, with no threads.
#include "arena_policy.h"
#include <csignal>
#include <cstdint>
#include <initializer_list>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

namespace {
struct Metrics {
    std::int64_t calls, total_ns, max_ns, max_tick, timeouts, errors, last_ns, status;
};
static volatile Metrics *active = nullptr;
static std::int64_t started = 0, tick = 0;
static bool decision = false;
static std::int64_t now() {
    timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return std::int64_t(t.tv_sec)*1000000000+t.tv_nsec;
}
static void finish(std::int64_t elapsed, int status) {
    if(!active) return;
    active->last_ns=elapsed; active->status=status;
    if(decision) {
        active->total_ns+=elapsed;
        if(elapsed>active->max_ns) { active->max_ns=elapsed; active->max_tick=tick; }
    }
    if(status==1) ++active->timeouts;
    if(status==2) ++active->errors;
}
static void failed(int signum) {
    // Only async-signal-safe operations; shared diagnostics survive worker exit.
    finish(now()-started, signum==SIGALRM ? 1 : 2);
    _exit(128+signum);
}
class Guard {
    sigset_t blocked, oldmask;
    std::int64_t limit;
public:
    Guard(Metrics *m, std::int64_t ns, bool act, std::int64_t current_tick=0):limit(ns) {
        sigemptyset(&blocked); sigaddset(&blocked,SIGALRM);
        sigprocmask(SIG_BLOCK,&blocked,&oldmask);
        struct sigaction action{};
        action.sa_handler=failed; sigemptyset(&action.sa_mask);
        for(int s : {SIGALRM,SIGSEGV,SIGBUS,SIGFPE,SIGABRT}) sigaction(s,&action,nullptr);
        active=m; decision=act; tick=current_tick; m->status=0;
        if(act) ++m->calls;
        started=now();
        itimerval timer{}; timer.it_value.tv_sec=ns/1000000000;
        timer.it_value.tv_usec=(ns%1000000000)/1000;
        setitimer(ITIMER_REAL,&timer,nullptr);
        sigprocmask(SIG_SETMASK,&oldmask,nullptr);
    }
    void done(int status=0) {
        auto elapsed=now()-started;
        sigprocmask(SIG_BLOCK,&blocked,&oldmask);
        itimerval zero{}; setitimer(ITIMER_REAL,&zero,nullptr);
        finish(elapsed,elapsed>limit ? 1 : status);
        active=nullptr;
        sigset_t pending; sigpending(&pending);
        if(sigismember(&pending,SIGALRM)) { int signal; sigwait(&blocked,&signal); }
        sigprocmask(SIG_SETMASK,&oldmask,nullptr);
    }
};
}

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
EXPORT void arena_host_destroy(void *instance,Metrics *m) {
    Guard guard(m,2000000000,false);
    int status=0;
    try { arena_destroy(instance); } catch(...) { status=2; }
    guard.done(status);
}
