// Native SDK adapter: same timed entry point as scalar/raw-batch execution.
#pragma once
#include "generals_native.h"
#include "sdk/arena_policy.h"

struct BoundPolicy {
    const std::uintptr_t *handles;
    std::int64_t (*call)(void *, const ArenaObservation *, std::int64_t *);
    std::int64_t *previous;
    const std::uintptr_t *metrics, *headers;
    std::int64_t *cursor;
};
extern "C" std::int64_t arena_bound_one(void *context,const GeneralsBatchView *view,int lane,std::int64_t *m) {
    using I=std::int64_t;
    auto &bound=*static_cast<BoundPolicy *>(context);
    const auto &o=view->observation;
    const int cells=view->height*view->width,players=view->players;
        const std::size_t base=std::size_t(lane)*cells, pb=std::size_t(lane)*players;
        I army[8],land[8],cities[8];
        for(int p=0;p<players;++p) {
            army[p]=o.stats[(pb+p)*3]; land[p]=o.stats[(pb+p)*3+1]; cities[p]=o.stats[(pb+p)*3+2];
        }
        const I *r=bound.previous+lane*10;
        ArenaMoveResult move{r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9]};
        ArenaObservation obs{view->width,view->height,players,int(view->player_ids[lane]),
            o.tick[lane],int(o.general_pos[lane]),o.terrain+base,o.structure+base,o.owner+base,o.army+base,
            reinterpret_cast<const std::uint8_t *>(o.visible)+base,
            reinterpret_cast<const std::uint8_t *>(o.alive)+pb,army,land,cities,
            reinterpret_cast<const std::uint8_t *>(o.fog_obstacles)+base,
            reinterpret_cast<const std::uint8_t *>(o.has_kill)+pb,
            reinterpret_cast<const std::uint8_t *>(o.warn_afk)+pb,o.neutralize_in+pb,r[0]?&move:nullptr};
    return bound.call(reinterpret_cast<void *>(bound.handles[lane]),&obs,m);
}
extern "C" std::int64_t arena_bound_act(void *context, const GeneralsBatchView *view,
                                       std::int64_t *actions) {
    using I=std::int64_t;
    auto &bound=*static_cast<BoundPolicy *>(context);
    const auto &o=view->observation;
    const int cells=view->height*view->width, players=view->players;
    for(int lane=0;lane<view->count;++lane) {
        actions[lane]=0;
        if(!o.active[lane]) continue;
        const std::size_t base=std::size_t(lane)*cells, pb=std::size_t(lane)*players;
        I army[8],land[8],cities[8];
        for(int p=0;p<players;++p) {
            army[p]=o.stats[(pb+p)*3]; land[p]=o.stats[(pb+p)*3+1]; cities[p]=o.stats[(pb+p)*3+2];
        }
        const I *r=bound.previous+lane*10;
        ArenaMoveResult move{r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9]};
        ArenaObservation obs{view->width,view->height,players,int(view->player_ids[lane]),
            o.tick[lane],int(o.general_pos[lane]),o.terrain+base,o.structure+base,o.owner+base,o.army+base,
            reinterpret_cast<const std::uint8_t *>(o.visible)+base,
            reinterpret_cast<const std::uint8_t *>(o.alive)+pb,army,land,cities,
            reinterpret_cast<const std::uint8_t *>(o.fog_obstacles)+base,
            reinterpret_cast<const std::uint8_t *>(o.has_kill)+pb,
            reinterpret_cast<const std::uint8_t *>(o.warn_afk)+pb,o.neutralize_in+pb,r[0]?&move:nullptr};
        auto *m=reinterpret_cast<I *>(bound.metrics[lane]);
        auto *header=reinterpret_cast<volatile I *>(bound.headers[lane]);
        *reinterpret_cast<volatile I *>(bound.cursor)=lane;
        if(header) { header[1]=obs.player; header[2]=obs.tick; header[3]=2; }
        actions[lane]=bound.call(reinterpret_cast<void *>(bound.handles[lane]),&obs,m);
        if(header) header[1]=-1;
        if(actions[lane]<0 || actions[lane]>I(cells)*8) {++m[5];m[7]=2;}
        if(m[7]) return lane;
    }
    return -1;
}
extern "C" void arena_bound_feedback(void *context,const GeneralsFeedback *result) {
    auto &bound=*static_cast<BoundPolicy *>(context);
    // Player ids were bound once in previous[:,1]. Each receipt has exactly
    // the same fields and canonical WAIT source as on_result_batch.
    for(int lane=0;lane<result->count;++lane) {
        const auto *r=result->moves+std::size_t(lane)*9;
        if(!result->active[lane] || !r[0]) continue;
        auto *p=bound.previous+std::size_t(lane)*10;
        p[0]=1; p[2]=result->actions[lane];
        for(int j=1;j<8;++j) p[j+2]=r[j];
        p[4]=p[2]>0?(p[2]-1)/8:-1;
    }
}
