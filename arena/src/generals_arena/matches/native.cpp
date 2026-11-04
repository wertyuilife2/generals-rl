// Synchronous native match loop. Backend owns rules and observations; policies
// own decisions. No Python callbacks, per-tick allocation or pointer rebinding.
#include "generals_native.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "recording.h"
#include "decision.h"

using I=std::int64_t;
using Clock=std::chrono::steady_clock;
struct JournalView {
    I *headers, *records, *committed, *actions;
    I capacity;
};
struct MatchConfig {
    int count, height, width, players;
    GeneralsBackend backend;
    const GeneralsPolicy *policies;
    const I *seats;
    I *joint, *invalid, *eliminations;
    bool *active, *ending;
    JournalView journal;
    I *metrics, *cursor;
    const char *const *topologies;
    int record;

};
struct Status {
    I group=-1, lane=-1, ticks=0;
    double observation=0, policy=0, step=0, bookkeeping=0;
};
struct Storage {
    std::vector<std::uint8_t> terrain,structure;
    std::vector<std::int16_t> owner;
    std::vector<I> army,stats,tick;
    std::vector<std::int32_t> general,winner;
    std::vector<double> neutralize;
    std::unique_ptr<bool[]> visible,alive,fog,kill,warn,active,terminated;
    Storage(std::size_t b,std::size_t n,std::size_t p):
        terrain(b*n),structure(b*n),owner(b*n),army(b*n),stats(b*p*3),tick(b),general(b),winner(b),
        neutralize(b*p),visible(new bool[b*n]{}),alive(new bool[b*p]{}),fog(new bool[b*n]{}),
        kill(new bool[b*p]{}),warn(new bool[b*p]{}),active(new bool[b]{}),terminated(new bool[b]{}) {}
    GeneralsObservation view() {
        return {terrain.data(),structure.data(),owner.data(),army.data(),visible.get(),alive.get(),
            fog.get(),kill.get(),warn.get(),neutralize.data(),stats.data(),general.data(),active.get(),
            tick.data(),terminated.get(),winner.data()};
    }
};
struct Match {
    MatchConfig c;
    std::vector<GeneralsPolicy> policies;
    std::vector<I> seats,actions,moves,measured;
    std::unique_ptr<bool[]> feedback_active;
    Storage storage;
    int remaining;
    std::unique_ptr<bool[]> forfeited,selection;
    std::vector<I> spectators;
    std::vector<record::Trace> traces;
    std::vector<std::pair<I,int>> departures;
    std::string response;
    std::vector<std::string> restored_states,restored_moves;
    record::Lane lane(int l)const{return {c.backend,l,c.height,c.width,c.players};}
    void synchronize(){if(c.backend.synchronize)c.backend.synchronize(c.backend.context);}
    void observe_full(){auto o=storage.view();c.backend.observe(c.backend.context,spectators.data(),nullptr,&o);synchronize();}
    void apply_departures(){for(auto event:departures)if(event.first==c.backend.tick[0])depart(event.second,0);}
    void depart(int seat,int l){
        if(seat<0||seat>=c.players||l<0||l>=c.count)throw std::runtime_error("invalid surrender seat/lane");
        auto at=std::size_t(l)*c.players+seat;
        if(forfeited[at])return;
        forfeited[at]=true;std::fill_n(selection.get(),c.count,false);selection[l]=true;
        c.backend.depart(c.backend.context,seat,selection.get());
        synchronize();
        if(!c.backend.alive[at]&&c.eliminations[at]<0)c.eliminations[at]=c.backend.tick[l];
        if(c.active[l]&&(c.backend.done[l*2]||c.backend.done[l*2+1])){c.active[l]=false;c.ending[l]=true;--remaining;}
        if(c.record==2){observe_full();traces[l].frame(lane(l),storage.view(),forfeited.get(),true);}
    }

    explicit Match(const MatchConfig &config):c(config),policies(c.policies,c.policies+c.players),
        seats(c.seats,c.seats+std::size_t(c.players)*c.count),actions(seats.size()),moves(std::size_t(c.count)*9),measured(c.count),
        feedback_active(new bool[c.count]),storage(c.count,c.height*c.width,c.players),remaining(c.count),
        forfeited(new bool[std::size_t(c.count)*c.players]{}),selection(new bool[c.count]),spectators(c.count,-1) {
        synchronize();
        for(int l=0;l<c.count;++l)traces.emplace_back(lane(l),c.topologies[l],c.record);
        if(c.record==2){observe_full();for(int l=0;l<c.count;++l)traces[l].frame(lane(l),storage.view(),forfeited.get());}
    }

    // Return on a completed lane so Python can serialize/persist its result.
    // Replay callers may also request one tick to hash the completed state.
    int advance(I limit,Status &s,const I *recorded=nullptr,bool warm=false,bool publish=true,bool capture=false) {
        s=Status{};
        generals_decision::install(); // Once per execution lease, never per decision.
        std::fill(c.ending,c.ending+c.count,false);
        auto add=[](double &total,Clock::time_point start) {
            total+=std::chrono::duration<double>(Clock::now()-start).count();
        };
        for(I iter=0;iter<limit && remaining;++iter) {
            auto obs=storage.view();
            for(int g=0;g<c.players;++g) {
                const I *ids=seats.data()+std::size_t(g)*c.count;
                if(recorded&&!warm) {
                    for(int l=0;l<c.count;++l)c.joint[std::size_t(l)*c.players+ids[l]]=
                        actions[std::size_t(g)*c.count+l]=recorded[(std::size_t(iter)*c.count+l)*c.players+ids[l]];
                    continue;
                }
                auto start=Clock::now();
                bool any=false;
                for(int l=0;l<c.count;++l){auto at=std::size_t(l)*c.players+ids[l];
                    selection[l]=c.active[l]&&c.backend.alive[at]&&!forfeited[at];any|=selection[l];}
                if(!any){for(int l=0;l<c.count;++l)c.joint[std::size_t(l)*c.players+ids[l]]=actions[std::size_t(g)*c.count+l]=0;continue;}
                c.backend.observe(c.backend.context,ids,selection.get(),&obs);
                synchronize();
                add(s.observation,start);
                start=Clock::now();
                GeneralsBatchView view{c.count,c.height,c.width,c.players,ids,obs};
                I *output=actions.data()+std::size_t(g)*c.count;
                I failed=-1;
                for(int l=0;l<c.count;++l)obs.active[l]=obs.active[l]&&c.active[l]&&!forfeited[std::size_t(l)*c.players+ids[l]];
                std::fill_n(output,c.count,0);
                if(!recorded||warm) {
                    if(policies[g].act_one) {
                        for(int l=0;l<c.count;++l)if(obs.active[l]) {
                            I *m=c.metrics+(std::size_t(l)*8+ids[l])*8;
                            if(c.cursor)*reinterpret_cast<volatile I *>(c.cursor)=l;
                            auto *header=c.journal.headers?reinterpret_cast<volatile I *>(c.journal.headers)+std::size_t(l)*4:nullptr;
                            if(header){header[1]=ids[l];header[2]=obs.tick[l];header[3]=2;}
                            I saved[8];if(recorded)std::copy_n(m,8,saved);
                            if(policies[g].flags&GENERALS_TIMED_POLICY){
                                generals_decision::Guard guard(reinterpret_cast<generals_decision::Metrics *>(m),10000000,true,obs.tick[l],true);
                                output[l]=policies[g].act_one(policies[g].context,&view,l,m);
                                guard.done(m[7]);
                            }else output[l]=policies[g].act_one(policies[g].context,&view,l,m);
                            if(output[l]<0||output[l]>I(c.height)*c.width*8){++m[5];m[7]=2;}
                            if(m[7]){s.group=g;s.lane=l;add(s.policy,start);return 3;}
                            if(recorded)std::copy_n(saved,8,m);
                            if(header)header[1]=-1;
                        }
                    }else{
                        for(int l=0;l<c.count;++l)measured[l]=c.metrics[(std::size_t(l)*8+ids[l])*8];
                        auto begin=Clock::now();failed=policies[g].act(policies[g].context,&view,output);
                        generals_decision::install(); // A foreign runtime may own signal handlers.
                        I elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-begin).count();
                        int count=0;for(int l=0;l<c.count;++l)count+=obs.active[l];
                        if(!recorded&&count)for(int l=0;l<c.count;++l)if(obs.active[l]){
                            I *m=c.metrics+(std::size_t(l)*8+ids[l])*8;if(m[0]!=measured[l])continue;++m[0];m[1]+=elapsed/count;m[6]=elapsed/count;
                            if(m[6]>m[2]){m[2]=m[6];m[3]=obs.tick[l];}
                        }
                    }
                }
                if(failed>=0) {
                    if(failed>=c.count) throw std::runtime_error("native policy returned invalid failure lane");
                    s.group=g;s.lane=failed;add(s.policy,start);return 1;
                }
                if(recorded)for(int l=0;l<c.count;++l)output[l]=recorded[(std::size_t(iter)*c.count+l)*c.players+ids[l]];
                for(int lane=0;lane<c.count;++lane)
                    c.joint[std::size_t(lane)*c.players+ids[lane]]=output[lane];
                add(s.policy,start);
            }
            auto start=Clock::now();
            c.backend.step(c.backend.context,c.joint);
            synchronize();
            add(s.step,start);
            start=Clock::now();
            for(int lane=0;lane<c.count;++lane) if(c.backend.errors[lane]) {
                s.lane=lane;return 2;
            }
            for(int lane=0;lane<c.count;++lane) if(c.active[lane]) {
                for(int p=0;p<c.players;++p) {
                    const auto at=std::size_t(lane)*c.players+p;
                    const I *row=c.backend.moves+at*9;
                    c.invalid[at]+=row[0] && !row[1];
                    if(!c.backend.alive[at] && c.eliminations[at]<0)
                        c.eliminations[at]=c.backend.tick[lane];
                }
            }
            for(int g=0;g<c.players;++g) if(policies[g].feedback) {
                const I *ids=seats.data()+std::size_t(g)*c.count;
                for(int lane=0;lane<c.count;++lane) {
                    const I *row=c.backend.moves+(std::size_t(lane)*c.players+ids[lane])*9;
                    std::copy_n(row,9,moves.data()+std::size_t(lane)*9);
                    feedback_active[lane]=c.active[lane] && row[0];
                }
                GeneralsFeedback feedback{c.count,moves.data(),actions.data()+std::size_t(g)*c.count,
                                          feedback_active.get()};
                policies[g].feedback(policies[g].context,&feedback);
            }
            bool ended=false;
            for(int lane=0;lane<c.count;++lane) if(c.active[lane]) {
                if(publish&&c.journal.headers) {
                    I tick=c.backend.tick[lane];
                    if(tick<1 || tick>c.journal.capacity) throw std::runtime_error("native journal tick out of range");
                    std::copy_n(c.joint+std::size_t(lane)*c.players,c.players,
                        c.journal.actions+((std::size_t(tick)-1)*c.count+lane)*c.players);
                    std::copy_n(c.journal.records+std::size_t(lane)*64,64,
                                c.journal.committed+std::size_t(lane)*64);
                    // Publish the completed tick only after actions and metrics.
                    std::atomic_thread_fence(std::memory_order_release);
                    reinterpret_cast<volatile I *>(c.journal.headers)[std::size_t(lane)*4]=tick;
                }
                if(c.backend.done[lane*2] || c.backend.done[lane*2+1]) {
                    c.active[lane]=false;c.ending[lane]=true;--remaining;ended=true;
                }
            }
            for(int l=0;l<c.count;++l)if(c.active[l]||c.ending[l])traces[l].tick(lane(l),c.joint+std::size_t(l)*c.players);
            if(c.record==2){observe_full();for(int l=0;l<c.count;++l)if(c.active[l]||c.ending[l])traces[l].frame(lane(l),storage.view(),forfeited.get());}
            if(c.count==1)apply_departures();
            if(capture){restored_states.push_back(lane(0).state());restored_moves.push_back(record::matrix(c.backend.moves,c.players,9));}
            ++s.ticks;add(s.bookkeeping,start);
            if(ended) break;
        }
        return 0;
    }
};
static thread_local std::string error;
extern "C" const char *arena_loop_error() { return error.c_str(); }
extern "C" void *arena_loop_create(const MatchConfig *c) {
    try {
        if(!c || c->count<1 || c->height<1 || c->width<1 || I(c->height)*c->width>4096 ||
           c->players<2 || c->players>8 || c->backend.abi!=GENERALS_NATIVE_ABI ||
           !c->backend.context || !c->backend.observe || !c->backend.step ||
           !c->backend.alive || !c->backend.done || !c->backend.tick || !c->backend.moves || !c->backend.errors ||
           !c->backend.depart || !c->backend.snapshot || !c->backend.stats || !c->metrics || !c->topologies || !c->policies || !c->seats || !c->joint || !c->invalid || !c->eliminations || !c->active || !c->ending ||
           !(c->backend.capabilities&GENERALS_HOST_BUFFERS) || c->record<0 || c->record>2)
            throw std::runtime_error("invalid native match binding");
        for(int g=0;g<c->players;++g)
            if(c->policies[g].abi!=GENERALS_NATIVE_ABI || !c->policies[g].act)
                throw std::runtime_error("unsupported native policy binding");
        for(int lane=0;lane<c->count;++lane) {
            unsigned used=0;
            for(int g=0;g<c->players;++g) {
                I seat=c->seats[std::size_t(g)*c->count+lane];
                if(seat<0 || seat>=c->players || (used & (1U<<seat)))
                    throw std::runtime_error("native seats must be a permutation per lane");
                used|=1U<<seat;
            }
        }
        if(c->journal.headers && (!c->journal.records || !c->journal.committed ||
                                  !c->journal.actions || c->journal.capacity<1))
            throw std::runtime_error("invalid native journal binding");
        return new Match(*c);
    } catch(const std::exception &e) {error=e.what();return nullptr;}
}
extern "C" int arena_loop_advance(Match *match,I ticks,Status *status) {
    try {
        if(!match || !status || ticks<1) throw std::runtime_error("invalid native advance");
        return match->advance(ticks,*status);
    } catch(const std::exception &e) {error=e.what();return -1;}
}
extern "C" void arena_loop_destroy(Match *match) { delete match; }

extern "C" int arena_loop_depart(Match *m,int lane,int seat) {
    try{m->depart(seat,lane);return 0;}catch(const std::exception &e){error=e.what();return -1;}
}
extern "C" int arena_loop_events(Match *m,const I *events,int count) {
    try{if(m->c.count!=1)throw std::runtime_error("event restoration requires one match");
        for(int i=0;i<count;++i){I tick=events[i*2],seat=events[i*2+1];
            if(tick<0||seat<0||seat>=m->c.players)throw std::runtime_error("invalid departure event");
            m->departures.emplace_back(tick,int(seat));}
        m->apply_departures();return 0;
    }catch(const std::exception &e){error=e.what();return -1;}
}
extern "C" int arena_loop_restore(Match *m,const I *actions,I ticks,int warm,int capture,Status *status) {
    try{if(m->c.count!=1||ticks<0)throw std::runtime_error("invalid restore dimensions");
        if(capture){m->restored_states.push_back(m->lane(0).state());m->restored_moves.push_back("null");}
        int result=m->advance(ticks,*status,actions,warm,false,capture);
        if(!result&&status->ticks!=ticks)throw std::runtime_error("recorded actions continue after termination");
        return result;
    }catch(const std::exception &e){error=e.what();return -1;}
}
extern "C" const char *arena_loop_query(Match *m,int kind,int l,int seat) {
    try{
        if(l<0||l>=m->c.count)throw std::runtime_error("invalid lane");
        auto lane=m->lane(l);auto &trace=m->traces[l];
        switch(kind){
        case 0:m->response=lane.result(m->c.invalid+std::size_t(l)*m->c.players,m->c.eliminations+std::size_t(l)*m->c.players,trace.initial_digest,trace.topology);break;
        case 1:m->response=trace.json(lane);break;
        case 2:m->response=lane.state();break;
        case 3:{if(seat<-1||seat>=m->c.players)throw std::runtime_error("invalid observer seat");
            std::fill(m->spectators.begin(),m->spectators.end(),seat);auto o=m->storage.view();
            m->c.backend.observe(m->c.backend.context,m->spectators.data(),nullptr,&o);m->synchronize();
            std::fill(m->spectators.begin(),m->spectators.end(),-1);
            m->response=m->lane(l).board(o,seat,m->forfeited.get());
            if(seat>=0){std::unique_ptr<bool[]> legal(new bool[std::size_t(m->c.count)*lane.n*8]);
                m->c.backend.legal(m->c.backend.context,seat,legal.get());
                m->synchronize();
                m->response.pop_back();m->response+=",\"legal_actions\":"+record::array(legal.get()+std::size_t(l)*lane.n*8,lane.n*8)+"}";}
            break;}
        case 4:m->response="{\"states\":"+record::joined(m->restored_states)+",\"moves\":"+record::joined(m->restored_moves)+"}";break;
        default:throw std::runtime_error("invalid query");}
        return m->response.c_str();
    }catch(const std::exception &e){error=e.what();return nullptr;}
}
