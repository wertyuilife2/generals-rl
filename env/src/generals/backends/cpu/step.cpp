// Native single-tick batch API. No policy, Python callbacks, or action history.
#include <algorithm>
#include <cstring>
#include <omp.h>
#include "../rules.h"
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
