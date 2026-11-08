// Incremental public observations for an exclusive native execution lease.
// Buffers remain caller-owned. A changed output address or observer forces a
// full refresh, so views cannot accidentally retain another player's fog.
#pragma once
#include <vector>

struct CellChanges {
    uint16_t *cells; uint8_t *marked; int *count;
    void changed(int cell) {
        if(!marked[cell]){marked[cell]=1;cells[(*count)++]=uint16_t(cell);}
    }
};
struct ObservationCache {
    struct Seat {
        std::vector<uint8_t> coverage,marked;
        std::vector<uint16_t> dirty;
        const i64 *output=nullptr;
        int count=0; bool full=true;
        explicit Seat(int n):coverage(n),marked(n),dirty(n){}
        void touch(int cell) {
            if(!full&&!marked[cell]) {
                marked[cell]=1;dirty[count++]=uint16_t(cell);
                if(count>int(dirty.size())/2)full=true;
            }
        }
        void clear() {for(int k=0;k<count;++k)marked[dirty[k]]=0;count=0;full=false;}
    };
    int batch,h,w,p,n;bool active=false;
    std::vector<Seat> seats;
    std::vector<int16_t> owners;
    std::vector<uint16_t> cells;
    std::vector<uint8_t> marked,terminal;
    std::vector<int> counts;
    ObservationCache(const Batch &s,int b,int height,int width,int players):
        batch(b),h(height),w(width),p(players),n(h*w),owners(size_t(b)*n),
        cells(size_t(b)*n),marked(size_t(b)*n),terminal(b),counts(b) {
        seats.reserve(size_t(b)*p);
        for(int k=0;k<b*p;++k)seats.emplace_back(n);
    }
    void activate(const Batch &s) {
        if(active)return;
        for(int l=0;l<batch;++l)reset(s,l);
        active=true;
    }
    void neighborhood(Seat &seat,int cell,int delta) {
        int y=cell/w,x=cell%w;
        for(int row=std::max(0,y-1);row<=std::min(h-1,y+1);++row)
            for(int col=std::max(0,x-1);col<=std::min(w-1,x+1);++col){
                int at=row*w+col;auto &v=seat.coverage[at];
                bool before=v!=0;v=uint8_t(int(v)+delta);
                if(before!=(v!=0))seat.touch(at);
            }
    }
    void reset(const Batch &s,int l) {
        size_t base=size_t(l)*n;
        for(int pid=0;pid<p;++pid){auto &seat=seats[l*p+pid];
            seat.full=true;std::fill(seat.coverage.begin(),seat.coverage.end(),0);}
        for(int c=0;c<n;++c){int pid=s.owner[base+c];owners[base+c]=int16_t(pid);
            if(pid>=0)neighborhood(seats[l*p+pid],c,1);}
        terminal[l]=s.terminated[l];
    }
    CellChanges changes(int l){return {cells.data()+size_t(l)*n,marked.data()+size_t(l)*n,&counts[l]};}
    void finish(const Batch &s,int l) {
        size_t base=size_t(l)*n;
        if(counts[l]>n/3||terminal[l]!=s.terminated[l])
            for(int pid=0;pid<p;++pid)seats[l*p+pid].full=true;
        terminal[l]=s.terminated[l];
        for(int k=0;k<counts[l];++k){int c=cells[base+k],old=owners[base+c],now=s.owner[base+c];
            if(old!=now){
                if(old>=0)neighborhood(seats[l*p+old],c,-1);
                if(now>=0)neighborhood(seats[l*p+now],c,1);
                owners[base+c]=int16_t(now);
            }
            for(int pid=0;pid<p;++pid)seats[l*p+pid].touch(c);
            marked[base+c]=0;
        }
        counts[l]=0;
    }
    void invalidate_output(const GeneralsObservation &o) {
        for(int l=0;l<batch;++l)for(int pid=0;pid<p;++pid)
            if(seats[l*p+pid].output==o.army+size_t(l)*n)seats[l*p+pid].full=true;
    }
    void observe(const Batch &s,const i64 *stats,const bool *done,const i64 *ids,
                 const bool *selected,GeneralsObservation &o) {
        activate(s);
        for(int l=0;l<batch;++l){
            if(selected&&!selected[l]){o.active[l]=false;continue;}
            int pid=int(ids[l]);size_t base=size_t(l)*n,pb=size_t(l)*p;
            auto &seat=seats[pb+pid];
            for(int other=0;other<p;++other)if(other!=pid&&seats[pb+other].output==o.army+base)
                seats[pb+other].full=true;
            if(seat.output!=o.army+base){seat.output=o.army+base;seat.full=true;}
            auto write=[&](int c){size_t at=base+c;bool seen=seat.coverage[c]||s.terminated[l];
                o.visible[at]=seen;o.terrain[at]=seen?s.terrain[at]:255;
                o.structure[at]=seen?s.structure[at]:255;o.owner[at]=seen?s.owner[at]:-3;
                o.army[at]=seen?s.army[at]:-1;
                o.fog_obstacles[at]=!seen&&(s.terrain[at]==1||s.structure[at]==1);
            };
            if(seat.full)for(int c=0;c<n;++c)write(c);
            else for(int k=0;k<seat.count;++k)write(seat.dirty[k]);
            seat.clear();
            // Spawn fog depends on discovery metadata as well as cell changes.
            for(int player=0;player<p;++player){size_t at=pb+player;int spawn=s.spawn_pos[at];
                if(spawn>=0){size_t cell=base+spawn;
                    o.fog_obstacles[cell]=!o.visible[cell]&&(s.terrain[cell]==1||s.structure[cell]==1);
                    if(s.general_pos[at]<0&&!(s.city_known[pb+pid]&(1<<player)))o.fog_obstacles[cell]=false;}
            }
            observe_metadata(s,stats,done,ids,o,l,p,false);
        }
    }
};
