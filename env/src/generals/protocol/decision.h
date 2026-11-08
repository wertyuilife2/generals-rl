// Common per-decision accounting and watchdog for every native Arena policy.
#pragma once
#include <csignal>
#include <cstdint>
#include <initializer_list>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

namespace generals_decision {
struct Metrics {
    std::int64_t calls, total_ns, max_ns, max_tick, timeouts, errors, last_ns, status;
};
static volatile Metrics *active = nullptr;
static std::int64_t started = 0, tick = 0;
static bool decision = false;
static std::int64_t now() {
#ifdef __APPLE__
    // CLOCK_MONOTONIC on Darwin rounds short calls to microseconds. The raw
    // uptime clock retains the hardware resolution needed for tiny policies.
    return std::int64_t(clock_gettime_nsec_np(CLOCK_UPTIME_RAW));
#else
    timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return std::int64_t(t.tv_sec)*1000000000+t.tv_nsec;
#endif
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
static void install() {
    struct sigaction action{};
    action.sa_handler=failed; sigemptyset(&action.sa_mask);
    for(int s : {SIGALRM,SIGSEGV,SIGBUS,SIGFPE,SIGABRT}) sigaction(s,&action,nullptr);
}
class Guard {
    sigset_t blocked, oldmask;
    std::int64_t limit;
public:
    Guard(Metrics *m, std::int64_t ns, bool act, std::int64_t current_tick=0, bool prepared=false):limit(ns) {
        sigemptyset(&blocked); sigaddset(&blocked,SIGALRM);
        sigprocmask(SIG_BLOCK,&blocked,&oldmask);
        if(!prepared) {
            struct sigaction current{};sigaction(SIGALRM,nullptr,&current);
            if(current.sa_handler!=failed)install();
        }
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
