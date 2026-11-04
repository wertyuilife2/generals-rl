// Native single-tick batch API. No policy, Python callbacks, or action history.
#include <algorithm>
#include <cstring>
#include <omp.h>
#include "../rules.h"
#include "../../protocol/generals_native.h"
#ifndef GENERALS_PROFILE_MARK
#define GENERALS_PROFILE_MARK(phase) ((void)0)
#define GENERALS_PROFILE_BATCH_BEGIN(batch, threads) ((void)0)
#define GENERALS_PROFILE_BATCH_END() ((void)0)
#define GENERALS_PROFILE_WORKER_BEGIN() ((void)0)
#define GENERALS_PROFILE_WORKER_END() ((void)0)
#define GENERALS_PROFILE_LANE_BEGIN(lane) ((void)0)
#define GENERALS_PROFILE_LANE_END(lane) ((void)0)
#endif
using i64=generals::I;
using u64=generals::U;
using generals::Batch;
struct Outputs {
    i64 *stats,*moves,*counters;
    int32_t *errors;
    bool *done;
    i64 *current_stats;
    uint16_t *growth_cells;
    int32_t *growth_count;
    bool *cache_valid;
    int32_t *owned_structures;
};
static void initialize(generals::Rules<> &rules){
    auto &c=rules.cache;
    for(int k=0;k<rules.p;++k)c.total[k]=c.territory[k]=c.cities[k]=c.structures[k]=0;
    c.growth_count=0;
    for(int cell=0;cell<rules.n;++cell){int pid=rules.owner[cell];if(pid<0)continue;
        if(rules.army[cell]>u64(INT64_MAX)-c.total[pid]){rules.error=3;return;}
        c.total[pid]+=rules.army[cell];++c.territory[pid];c.cities[pid]+=rules.structure[cell]==1;
        if(rules.structure[cell]){++c.structures[pid];rules.growth[c.growth_count++]=cell;}}
}
static void step_lane(Batch s,Batch initial,Outputs out,const i64 *actions,
                      int lane,int h,int w,int p,bool auto_reset,i64 horizon){
    const int n=h*w;const size_t base=size_t(lane)*n,pb=size_t(lane)*p;
    auto *moves=out.moves+pb*9,*stats=out.stats+pb*3,*cached=out.current_stats+pb*3;
    auto *counts=reinterpret_cast<u64*>(out.counters+size_t(lane)*9);
    if(out.errors[lane]||s.terminated[lane]||(!auto_reset&&horizon>0&&s.tick[lane]>=horizon)){
        for(int k=0;k<p;++k)generals::empty(moves+k*9);return;}
    generals::Cache c{};
    generals::Rules<> rules(s,lane,p,h,w,c,out.growth_cells+base);
    if(out.cache_valid[lane]){
        c.growth_count=out.growth_count[lane];
        for(int k=0;k<p;++k){c.total[k]=cached[k*3];c.territory[k]=cached[k*3+1];
            c.cities[k]=cached[k*3+2];c.structures[k]=out.owned_structures[pb+k];}
    }else initialize(rules);
    if(rules.error){out.errors[lane]=rules.error;return;}
    if(s.tick[lane]==INT64_MAX){out.errors[lane]=4;return;}
    GENERALS_PROFILE_MARK(0);
    rules.before_growth(actions+pb,moves,counts);
    if(rules.error){out.errors[lane]=rules.error;--s.tick[lane];return;}
    GENERALS_PROFILE_MARK(2);
    i64 tick=s.tick[lane];
    for(int k=0;k<p;++k){
        u64 gain=(tick%2==0?c.structures[k]:0)+(tick%50==0?c.territory[k]:0);
        if(c.total[k]>u64(INT64_MAX)-gain){
            out.errors[lane]=3;
            for(int cell=0;cell<n;++cell)if(rules.owner[cell]==k){
                u64 add=(tick%2==0&&rules.structure[cell]?1:0)+(tick%50==0?1:0);
                if(rules.army[cell]>u64(INT64_MAX)-add){out.errors[lane]=1;break;}}
            --s.tick[lane];return;
        }
    }
    if(tick%2==0){
        for(int k=0;k<c.growth_count;++k)++rules.army[rules.growth[k]];
        for(int k=0;k<p;++k)c.total[k]+=c.structures[k];
    }
    if(tick%50==0){
        for(int cell=0;cell<n;++cell)if(rules.owner[cell]>=0)++rules.army[cell];
        for(int k=0;k<p;++k)c.total[k]+=c.territory[k];
    }
    GENERALS_PROFILE_MARK(1);
    rules.after_growth();
    for(int k=0;k<p;++k){stats[k*3]=i64(c.total[k]);stats[k*3+1]=c.territory[k];stats[k*3+2]=c.cities[k];}
    ++counts[0];bool terminal=s.terminated[lane],truncated=!terminal&&horizon>0&&tick>=horizon;
    out.done[lane*2]=terminal;out.done[lane*2+1]=truncated;counts[7]+=terminal;counts[8]+=truncated;
    GENERALS_PROFILE_MARK(3);
    if(auto_reset&&(terminal||truncated)){
        std::memcpy(s.structure+base,initial.structure+base,n);
        std::memcpy(s.owner+base,initial.owner+base,n*sizeof(int16_t));
        std::memcpy(s.army+base,initial.army+base,n*sizeof(i64));
        generals::reset_metadata(s,initial,lane,p);initialize(rules);
    }
    GENERALS_PROFILE_MARK(4);
    for(int k=0;k<p;++k){cached[k*3]=i64(c.total[k]);cached[k*3+1]=c.territory[k];cached[k*3+2]=c.cities[k];
        out.owned_structures[pb+k]=c.structures[k];}
    out.growth_count[lane]=c.growth_count;out.cache_valid[lane]=true;
    GENERALS_PROFILE_MARK(5);
}
extern "C" void generals_step(Batch *s,const Batch *initial,Outputs *out,const i64 *actions,
                              int batch,int h,int w,int players,int threads,int auto_reset,i64 horizon){
    GENERALS_PROFILE_BATCH_BEGIN(batch,threads);
    #pragma omp parallel num_threads(threads) if(threads>1)
    {
        GENERALS_PROFILE_WORKER_BEGIN();
        #pragma omp for schedule(static)
        for(int lane=0;lane<batch;++lane){
            GENERALS_PROFILE_LANE_BEGIN(lane);
            step_lane(*s,*initial,*out,actions,lane,h,w,players,auto_reset,horizon);
            GENERALS_PROFILE_LANE_END(lane);
        }
        GENERALS_PROFILE_WORKER_END();
    }
    GENERALS_PROFILE_BATCH_END();
}

// Boundary commands use the same lifecycle and cache updates as ordinary ticks.
extern "C" void generals_depart(Batch *s, Outputs *out, int batch, int h, int w, int p,
                                int pid, const bool *mask, int neutralize, i64 horizon) {
    for(int lane=0;lane<batch;++lane) {
        if((mask&&!mask[lane]) || (horizon>0&&s->tick[lane]>=horizon))continue;
        generals::Cache cache{};
        generals::Rules<> rules(*s,lane,p,h,w,cache,out->growth_cells+size_t(lane)*h*w);
        initialize(rules);
        if(neutralize)rules.neutralize(pid);else rules.surrender(pid);
        out->cache_valid[lane]=false;
        out->done[lane*2]=out->done[lane*2]||s->terminated[lane];
        if(neutralize)for(int k=0;k<p;++k){auto *v=out->stats+(size_t(lane)*p+k)*3;
            v[0]=cache.total[k];v[1]=cache.territory[k];v[2]=cache.cities[k];}
    }
}

// Public observation construction belongs to env, never to a policy adapter.
// Outputs are caller-owned snapshots. This loop uses no extra worker threads.
using ObservationOutputs = GeneralsObservation;
static void observe_batch(const Batch *source, const i64 *stats, const bool *done,
                                  const i64 *ids, ObservationOutputs *out,
                                  int batch, int h, int w, int players, const bool *selected) {
    const auto &s=*source;
    const int n=h*w;
    for (int lane=0;lane<batch;++lane) {
        if(selected && !selected[lane]) {out->active[lane]=false;continue;}
        const size_t base=size_t(lane)*n, pb=size_t(lane)*players;
        const bool full=ids[lane]<0;
        const int pid=full?int(-ids[lane]-2):int(ids[lane]);
        bool *visible=out->visible+base;
        std::fill(visible,visible+n,bool(full||s.terminated[lane]));
        if (!full&&!s.terminated[lane]) {
            for (int cell=0;cell<n;++cell) if (s.owner[base+cell]==pid) {
                int y=cell/w,x=cell%w;
                for (int row=std::max(0,y-1);row<=std::min(h-1,y+1);++row)
                    for (int col=std::max(0,x-1);col<=std::min(w-1,x+1);++col)
                        visible[row*w+col]=true;
            }
        }
        for (int cell=0;cell<n;++cell) {
            const size_t at=base+cell;
            const bool seen=visible[cell];
            out->terrain[at]=seen?s.terrain[at]:255;
            out->structure[at]=seen?s.structure[at]:255;
            out->owner[at]=seen?s.owner[at]:-3;
            out->army[at]=seen?s.army[at]:-1;
            out->fog_obstacles[at]=!seen&&(s.terrain[at]==1||s.structure[at]==1);
        }
        for (int p=0;p<players;++p) {
            const size_t at=pb+p;
            const int spawn=s.spawn_pos[at];
            if (!full && spawn>=0 && s.general_pos[at]<0 && !(s.city_known[pb+pid]&(1<<p)))
                out->fog_obstacles[base+spawn]=false;
            out->alive[at]=s.alive[at];
            out->has_kill[at]=s.kills[at]>0;
            out->warn_afk[at]=s.warn_afk[at]!=0;
            out->neutralize_in[at]=(s.leave_tick[at]>=0 && s.general_pos[at]>=0)
                ? std::clamp(double(s.tick[lane]-s.leave_tick[at])/50.,0.,1.) : -1.;
            out->stats[at*3]=stats[at*3];
            out->stats[at*3+1]=stats[at*3+1];
            out->stats[at*3+2]=(full||p==pid||s.terminated[lane])?stats[at*3+2]:-1;
        }
        out->general_pos[lane]=pid<0?-1:s.general_pos[pb+pid];
        out->active[lane]=!done[lane*2]&&!done[lane*2+1]&&pid>=0&&s.alive[pb+pid];
        out->tick[lane]=s.tick[lane];
        out->terminated[lane]=s.terminated[lane];
        out->winner_id[lane]=s.winner[lane];
    }
}

extern "C" void generals_observe(const Batch *source, const i64 *stats, const bool *done,
                                  const i64 *ids, ObservationOutputs *out,
                                  int batch, int h, int w, int players) {
    observe_batch(source,stats,done,ids,out,batch,h,w,players,nullptr);
}

// Same kernels with geometry and storage bound once by the backend owner.
// Observation masking remains exclusively in generals_observe above.
struct BoundCPU {
    Batch *state, *initial;
    Outputs *out;
    int count, height, width, players, threads, auto_reset;
    i64 horizon;
};
extern "C" void generals_bound_step(void *context, const i64 *actions) {
    auto &b=*static_cast<BoundCPU *>(context);
    generals_step(b.state,b.initial,b.out,actions,b.count,b.height,b.width,b.players,
                  b.threads,b.auto_reset,b.horizon);
}
extern "C" void generals_bound_observe(void *context, const i64 *ids, const bool *selected, GeneralsObservation *out) {
    auto &b=*static_cast<BoundCPU *>(context);
    observe_batch(b.state,b.out->stats,b.out->done,ids,out,b.count,b.height,b.width,b.players,selected);
}

extern "C" void generals_bound_snapshot(void *context, GeneralsStateView *out) {
    const auto &s=*static_cast<BoundCPU *>(context)->state;
    *out={s.terrain,s.structure,s.owner,s.army,s.alive,s.general_pos,s.tick,s.terminated,s.winner,
          s.idle_ticks,s.afk_turn,s.end_reason,s.leave_tick,s.death_order,s.kills,s.damage,
          s.spawn_pos,s.city_known,s.capture_mask,s.warn_afk};
}

extern "C" void generals_bound_depart(void *context,int pid,const bool *mask) {
    auto &b=*static_cast<BoundCPU *>(context);
    generals_depart(b.state,b.out,b.count,b.height,b.width,b.players,pid,mask,0,b.horizon);
}
extern "C" void generals_visibility(const int16_t *owner,bool *visible,int h,int w,int pid) {
    std::fill(visible,visible+h*w,false);
    for(int c=0;c<h*w;++c)if(owner[c]==pid)
        for(int y=std::max(0,c/w-1);y<=std::min(h-1,c/w+1);++y)
            for(int x=std::max(0,c%w-1);x<=std::min(w-1,c%w+1);++x)visible[y*w+x]=true;
}
extern "C" void generals_legal(const int16_t *owner,const i64 *army,const uint8_t *terrain,
                               bool *mask,int h,int w,int pid) {
    std::fill(mask,mask+h*w*8,false);
    const int dy[4]={-1,0,1,0},dx[4]={0,1,0,-1};
    for(int c=0;c<h*w;++c)if(owner[c]==pid&&army[c]>1)for(int d=0;d<4;++d){
        int y=c/w+dy[d],x=c%w+dx[d];
        if(y>=0&&y<h&&x>=0&&x<w&&terrain[y*w+x]!=1)mask[c*8+d*2]=mask[c*8+d*2+1]=true;
    }
}

extern "C" void generals_bound_legal(void *context,int pid,bool *out) {
    auto &b=*static_cast<BoundCPU *>(context);int n=b.height*b.width;
    for(int l=0;l<b.count;++l){auto base=size_t(l)*n;auto *mask=out+base*8;
        if(!b.state->alive[size_t(l)*b.players+pid]||b.state->terminated[l])std::fill_n(mask,n*8,false);
        else generals_legal(b.state->owner+base,b.state->army+base,b.state->terrain+base,mask,b.height,b.width,pid);}
}
