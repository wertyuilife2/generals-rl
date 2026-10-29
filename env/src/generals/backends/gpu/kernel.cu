// Cooperative CUDA board growth; the CPU and GPU share ordered game rules.
#include <cuda_runtime.h>
#include <cstdint>
#include "../rules.h"
#ifndef GENERALS_PACKED_METADATA
#define GENERALS_PACKED_METADATA 0
#endif
#ifndef GENERALS_GROUP_THREADS
#define GENERALS_GROUP_THREADS 32
#endif
#ifndef GENERALS_ENVS_PER_BLOCK
#define GENERALS_ENVS_PER_BLOCK 1
#endif
constexpr int P=GENERALS_PLAYERS,THREADS=GENERALS_GROUP_THREADS,ENVS=GENERALS_ENVS_PER_BLOCK;
constexpr int BLOCK_THREADS=THREADS*ENVS;
constexpr bool PACKED=GENERALS_PACKED_METADATA;
static_assert(THREADS==4||THREADS==8||THREADS==16||THREADS==32||THREADS==64||THREADS==128||THREADS==256||THREADS==512||THREADS==1024);
static_assert(THREADS<=32||ENVS==1);
static_assert(BLOCK_THREADS%32==0&&BLOCK_THREADS<=1024);
using generals::I;using generals::U;using generals::Batch;
struct Arguments{
    Batch state,initial;
    const I *actions; I *counters,*stats,*moves;
    int32_t *errors;bool *done;
    int32_t batch;I horizon;int32_t auto_reset;
    I *cache_stats;uint16_t *growth_cells;int32_t *growth_count;
    bool *cache_valid;int32_t *owned_structures;
};
struct Shared{
    generals::Cache cache;
    U counters[9]; I moves[P*9],stats[P*3];
    bool reset_needed;
};
__device__ __forceinline__ void barrier(){
#if GENERALS_GROUP_THREADS>32
    __syncthreads();
#else
    constexpr unsigned mask=0xffffffffu>>(32-THREADS);
    __syncwarp(mask<<((threadIdx.x%32/THREADS)*THREADS));
#endif
}
__global__ void simulate(Arguments a){
    int group=threadIdx.x/THREADS,tid=threadIdx.x%THREADS,lane=blockIdx.x*ENVS+group;
    if(lane>=a.batch)return;
    constexpr int h=GENERALS_HEIGHT,w=GENERALS_WIDTH,n=h*w;
    size_t base=size_t(lane)*n,pb=size_t(lane)*P;
    __shared__ Shared slots[ENVS];Shared &s=slots[group];auto &c=s.cache;
    uint16_t *growth=a.growth_cells+base;
    uint32_t *packed=PACKED?reinterpret_cast<uint32_t*>(a.growth_cells+((size_t(a.batch)*n+1)&~size_t(1)))+base:nullptr;
    generals::Rules<PACKED> rules(a.state,lane,P,h,w,c,growth,packed);
    if(a.state.terminated[lane]||(!a.auto_reset&&a.horizon>0&&a.state.tick[lane]>=a.horizon)){
        for(int k=tid;k<P*9;k+=THREADS){int field=k%9;a.moves[pb*9+k]=(field==2||field==3||field==6)?-1:0;}return;
    }
    for(int k=tid;k<9;k+=THREADS)s.counters[k]=a.counters[size_t(lane)*9+k];
    if(a.cache_valid[lane]){
        if(tid==0)c.growth_count=a.growth_count[lane];
        for(int pid=tid;pid<P;pid+=THREADS){c.total[pid]=a.cache_stats[pb*3+pid*3];
            c.territory[pid]=int(a.cache_stats[pb*3+pid*3+1]);c.cities[pid]=int(a.cache_stats[pb*3+pid*3+2]);
            c.structures[pid]=a.owned_structures[pb+pid];}
    }else{
        if(tid==0)c.growth_count=0;
        for(int pid=tid;pid<P;pid+=THREADS)c.total[pid]=c.territory[pid]=c.cities[pid]=c.structures[pid]=0;
        barrier();
        for(int cell=tid;cell<n;cell+=THREADS){
            int pid=rules.owner[cell],kind=rules.structure[cell];
            if constexpr(PACKED)packed[cell]=uint32_t(rules.terrain[cell])|(uint32_t(kind)<<8)|(uint32_t(uint16_t(pid))<<16);
            if(pid>=0){atomicAdd(reinterpret_cast<unsigned long long*>(c.total+pid),static_cast<unsigned long long>(rules.army[cell]));
                atomicAdd(c.territory+pid,1);if(kind==1)atomicAdd(c.cities+pid,1);
                if(kind){atomicAdd(c.structures+pid,1);growth[atomicAdd(&c.growth_count,1)]=cell;}}
        }
    }
    barrier();
    if(tid==0)rules.before_growth(a.actions+pb,s.moves,s.counters);
    barrier();
    I tick=a.state.tick[lane];
    if(tick%2==0){
        for(int k=tid;k<c.growth_count;k+=THREADS)++rules.army[growth[k]];
        for(int pid=tid;pid<P;pid+=THREADS)c.total[pid]+=c.structures[pid];
    }
    // A city/general receives both growth increments on the land tick.
    barrier();
    if(tick%50==0){
        for(int cell=tid;cell<n;cell+=THREADS)if(rules.owner[cell]>=0)++rules.army[cell];
        for(int pid=tid;pid<P;pid+=THREADS)c.total[pid]+=c.territory[pid];
    }
    barrier();
    if(tid==0){
        rules.after_growth();
        for(int pid=0;pid<P;++pid){s.stats[pid*3]=I(c.total[pid]);s.stats[pid*3+1]=c.territory[pid];s.stats[pid*3+2]=c.cities[pid];}
        ++s.counters[0];bool terminal=a.state.terminated[lane],truncated=!terminal&&a.horizon>0&&tick>=a.horizon;
        a.done[lane*2]=terminal;a.done[lane*2+1]=truncated;s.counters[7]+=terminal;s.counters[8]+=truncated;
        s.reset_needed=a.auto_reset&&(terminal||truncated);a.errors[lane]=0;
    }
    barrier();
    if(s.reset_needed){
        for(int cell=tid;cell<n;cell+=THREADS){
            rules.owner[cell]=a.initial.owner[base+cell];rules.army[cell]=a.initial.army[base+cell];
            rules.structure[cell]=a.initial.structure[base+cell];
        }
        if(tid==0)generals::reset_metadata(a.state,a.initial,lane,P);
    }
    for(int pid=tid;pid<P;pid+=THREADS)a.owned_structures[pb+pid]=c.structures[pid];
    if(tid==0){a.growth_count[lane]=c.growth_count;a.cache_valid[lane]=!s.reset_needed;}
    for(int k=tid;k<P*9;k+=THREADS)a.moves[pb*9+k]=s.moves[k];
    for(int k=tid;k<P*3;k+=THREADS){a.stats[pb*3+k]=s.stats[k];if(a.stats!=a.cache_stats)a.cache_stats[pb*3+k]=s.stats[k];}
    for(int k=tid;k<9;k+=THREADS)a.counters[size_t(lane)*9+k]=I(s.counters[k]);
}
extern "C" int generals_launch(const Arguments *a,void *stream,int *resources){
    if(resources){cudaFuncAttributes attr;cudaError_t status=cudaFuncGetAttributes(&attr,simulate);
        if(status!=cudaSuccess)return status;int blocks;
        status=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,simulate,BLOCK_THREADS,0);
        resources[0]=attr.numRegs;resources[1]=int(attr.localSizeBytes);resources[2]=int(attr.sharedSizeBytes);resources[3]=blocks;return status;}
    simulate<<<(a->batch+ENVS-1)/ENVS,BLOCK_THREADS,0,cudaStream_t(stream)>>>(*a);
    return cudaGetLastError();
}
extern "C" const char *generals_error(int status){return cudaGetErrorString(cudaError_t(status));}
