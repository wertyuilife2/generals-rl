// Standard 1v1/FFA move ordering and lifecycle, shared by CPU, CUDA and Metal.
// Board growth remains cooperative on CUDA; rare captures/trades run in leader.
#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define RULE_LOCAL thread
#else
#include <cstdint>
#define RULE_LOCAL
#endif
#ifdef __CUDACC__
#define RULE_INLINE __device__ __forceinline__
#else
#define RULE_INLINE inline
#endif
namespace generals {
#ifdef __METAL_VERSION__
using int64_t = long;
using uint64_t = ulong;
using int32_t = int;
using uint32_t = uint;
using int16_t = short;
using uint16_t = ushort;
using uint8_t = uchar;
// Logical row-major indexing over packed cell records and field-major tensors.
// The rule code below is identical on all devices; only addressing differs.
template<class T> struct Array {
    device T *data;
    size_t columns, row_stride, column_stride, offset;
    RULE_INLINE device T &operator[](size_t index) const {
        index += offset;
        return data[(index / columns) * row_stride + (index % columns) * column_stride];
    }
    RULE_INLINE Array operator+(size_t index) const {
        return {data, columns, row_stride, column_stride, offset + index};
    }
};
RULE_INLINE Array<uint64_t> unsigned_armies(Array<int64_t> a) {
    return {reinterpret_cast<device uint64_t *>(a.data), a.columns, a.row_stride, a.column_stride, a.offset};
}
#else
template<class T> using Array = T *;
RULE_INLINE uint64_t *unsigned_armies(int64_t *a) { return reinterpret_cast<uint64_t *>(a); }
#endif
using I = int64_t;
using U = uint64_t;
struct Batch {
    Array<uint8_t> terrain, structure;
    Array<int16_t> owner;
    Array<I> army;
    Array<bool> alive;
    Array<int32_t> general_pos;
    Array<I> tick;
    Array<bool> terminated;
    Array<int32_t> winner;
    Array<I> idle_ticks, afk_turn;
    Array<int32_t> end_reason;
    Array<I> leave_tick;
    Array<int32_t> death_order, kills;
    Array<I> damage;
    Array<int32_t> spawn_pos;
    Array<uint8_t> city_known;
    Array<uint8_t> capture_mask, warn_afk;
};
struct Cache {
    U total[8];
    int territory[8], cities[8], structures[8], growth_count;
    int previous_land[8];
};
struct Move { int pid, source, target; bool half_move; U strength; bool friendly, general; };
RULE_INLINE U transfer(U value, bool half_move) { return half_move ? value/2 : value ? value-1 : 0; }
RULE_INLINE void empty(RULE_LOCAL I *r) {
    for (int k=0;k<9;++k) r[k]=(k==2||k==3||k==6)?-1:0;
}

template<bool Packed=false> struct Rules {
    Batch b; int lane, p, h, w, n; size_t base, pb;
    Array<uint8_t> terrain, structure; Array<int16_t> owner; Array<U> army; Array<bool> alive;
    Array<int32_t> general, deaths, kills, spawn; Array<I> leave, damage;
    Array<uint8_t> known; Array<uint16_t> growth; Array<uint32_t> packed;
    RULE_LOCAL Cache &cache;
    int error = 0;
    RULE_INLINE Rules(Batch batch,int lane_,int p_,int h_,int w_,RULE_LOCAL Cache &c,
                      Array<uint16_t> g,Array<uint32_t> info={}):
        b(batch),lane(lane_),p(p_),h(h_),w(w_),n(h_*w_),base(size_t(lane_)*n),pb(size_t(lane_)*p_),
        terrain(b.terrain+base),structure(b.structure+base),owner(b.owner+base),
        army(unsigned_armies(b.army+base)),alive(b.alive+pb),general(b.general_pos+pb),
        deaths(b.death_order+pb),kills(b.kills+pb),spawn(b.spawn_pos+pb),leave(b.leave_tick+pb),
        damage(b.damage+pb*p),known(b.city_known+pb),growth(g),packed(info),cache(c) {}
    RULE_INLINE int owns(int cell) const {
        if constexpr(Packed) return int16_t(packed[cell]>>16);
        return owner[cell];
    }
    RULE_INLINE void replace(int cell,int pid,U count,int kind) {
        int old=owner[cell], oldkind=structure[cell];
        if(old>=0) {cache.total[old]-=army[cell]; --cache.territory[old];
            cache.cities[old]-=oldkind==1; cache.structures[old]-=oldkind!=0;}
        if(pid>=0) {cache.total[pid]+=count; ++cache.territory[pid];
            cache.cities[pid]+=kind==1; cache.structures[pid]+=kind!=0;}
#if !defined(__CUDACC__) && !defined(__METAL_VERSION__)
        if(pid>=0&&cache.total[pid]>U(INT64_MAX))error=3;
#endif
        owner[cell]=pid; army[cell]=count; structure[cell]=kind;
        if constexpr(Packed) packed[cell]=uint32_t(terrain[cell])|(uint32_t(kind)<<8)|(uint32_t(uint16_t(pid))<<16);
    }
    RULE_INLINE void die(int pid) {
        if(!alive[pid]) return;
        int ordinal=-1;
        for(int k=0;k<p;++k) if(deaths[k]>ordinal) ordinal=deaths[k];
        deaths[pid]=ordinal+1; alive[pid]=false;
        int remaining=0,last=-1;
        for(int k=0;k<p;++k)if(alive[k]){++remaining;last=k;}
        if(remaining==1)b.winner[lane]=last;
    }
    RULE_INLINE void credit(int killer,int victim) {
        ++kills[killer];b.capture_mask[pb+killer]|=1<<victim;
    }
    RULE_INLINE bool visible(int cell,int pid) const {
        int y=cell/w,x=cell%w;
        for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx)
            if(y+dy>=0&&y+dy<h&&x+dx>=0&&x+dx<w&&owner[(y+dy)*w+x+dx]==pid) return true;
        return false;
    }
    RULE_INLINE bool before(RULE_LOCAL const Move &a,RULE_LOCAL const Move &other) const {
        if(a.friendly!=other.friendly) return a.friendly;
        if(a.general!=other.general) return !a.general;
        if(a.strength!=other.strength) return a.strength>other.strength;
        return b.tick[lane]%2 ? a.pid>other.pid : a.pid<other.pid;
    }
    RULE_INLINE bool valid(RULE_LOCAL const Move &m) const {
        return alive[m.pid]&&owner[m.source]==m.pid&&army[m.source]>1;
    }
    RULE_INLINE bool trades(RULE_LOCAL const Move &a,RULE_LOCAL const Move &other) const {
        if(!valid(a)||!valid(other)||a.pid==other.pid) return false;
        if(general[other.pid]!=a.target||general[a.pid]!=other.target) return false;
        U da=army[a.target],db=army[other.target];
        if(other.source==a.target) da-=transfer(army[other.source],other.half_move);
        if(a.source==other.target) db-=transfer(army[a.source],a.half_move);
        return transfer(army[a.source],a.half_move)>da&&transfer(army[other.source],other.half_move)>db;
    }
    RULE_INLINE void trade(RULE_LOCAL const Move &a,RULE_LOCAL const Move &other,RULE_LOCAL I *records,RULE_LOCAL U *counts) {
        int i=a.pid,j=other.pid,gi=general[i],gj=general[j];
        U mi=transfer(army[a.source],a.half_move),mj=transfer(army[other.source],other.half_move);
        replace(a.source,i,army[a.source]-mi,structure[a.source]);
        replace(other.source,j,army[other.source]-mj,structure[other.source]);
        U di=army[gi],dj=army[gj];
        for(int c=0;c<n;++c) if(c!=gi&&c!=gj&&(owner[c]==i||owner[c]==j))
            replace(c,owner[c]==i?j:i,(army[c]+1)/2,structure[c]);
        replace(gi,j,mj-di,2); replace(gj,i,mi-dj,2);
        general[i]=gj;general[j]=gi;
        RULE_LOCAL I *ra=records+i*9,*rb=records+j*9;
        ra[4]=mi;rb[4]=mj;ra[5]=rb[5]=ra[7]=rb[7]=1;ra[8]=rb[8]=7;
        counts[5]+=2; counts[6]+=2;
    }
    RULE_INLINE bool moves(RULE_LOCAL const I *actions,RULE_LOCAL I *records,RULE_LOCAL U *counts) {
        Move pending[8]; int count=0;
        for(int pid=0;pid<p;++pid) {
            RULE_LOCAL I *r=records+pid*9; empty(r);
            if(!alive[pid]) continue;
            r[0]=1;++counts[1];
            I action=actions[pid];
            if(!action){r[1]=1;r[8]=1;++counts[3];continue;}
            ++counts[2]; I encoded=I(U(action)-1),source=encoded/8;
            int direction=(encoded%8)/2,target=-1;
            bool half_move=encoded%2==1,in_bounds=action>0&&source>=0&&source<n;
            if(in_bounds){int y=int(source)/w,x=int(source)%w;
                int ty=y+(direction==0?-1:direction==2?1:0),tx=x+(direction==3?-1:direction==1?1:0);
                in_bounds=ty>=0&&ty<h&&tx>=0&&tx<w;if(in_bounds)target=ty*w+tx;}
            r[2]=source;r[3]=target;
            r[8]=!in_bounds?3:owns(source)!=pid?4:terrain[target]?5:army[source]<=1?6:2;
            if(r[8]!=2){++counts[4];continue;}
            r[1]=1;
            Move m{pid,int(source),target,half_move,army[source],owns(target)==pid,structure[target]==2};
            int pos=count++; while(pos&&before(m,pending[pos-1])){pending[pos]=pending[pos-1];--pos;}
            pending[pos]=m;
        }
        Move ordered[8]; int length=0;
        while(count){
            int chosen=0;
            for(int i=0;i<count;++i){bool blocked=false;
                for(int j=0;j<count;++j) if(i!=j&&pending[j].target==pending[i].source&&pending[j].source!=pending[i].target)blocked=true;
                if(!blocked){chosen=i;break;}}
            ordered[length++]=pending[chosen];
            for(int j=chosen+1;j<count;++j)pending[j-1]=pending[j]; --count;
        }
        bool activity=false,used[8]{};
        for(int i=0;i<length;++i){
            if(used[i])continue;
            Move m=ordered[i];RULE_LOCAL I *r=records+m.pid*9;
            if(owns(m.source)!=m.pid){r[1]=0;r[8]=4;++counts[4];continue;}
            activity=true;bool swapped=false;
            for(int j=i+1;j<length;++j)if(!used[j]&&trades(m,ordered[j])){
                trade(m,ordered[j],records,counts);used[j]=true;swapped=true;break;}
            if(swapped)continue;
            U moving=transfer(army[m.source],m.half_move),defense=army[m.target];
            int defender=owns(m.target),kind=structure[m.target];
            bool friendly=defender==m.pid;
            if(!army[m.source]||(!friendly&&!moving)){r[1]=0;r[8]=6;++counts[4];continue;}
#if !defined(__CUDACC__) && !defined(__METAL_VERSION__)
            if(friendly&&defense>U(INT64_MAX)-moving){error=2;return activity;}
#endif
            r[4]=moving;army[m.source]-=moving;
            bool captured=!friendly&&moving>defense;
            if(friendly)army[m.target]+=moving;
            else{
                cache.total[m.pid]-=moving;
                if(defender>=0)damage[m.pid*p+defender]+=I(moving<defense?moving:defense);
                if(captured){
                    if(defender<0&&kind)growth[cache.growth_count++]=m.target;
                    replace(m.target,m.pid,moving-defense,kind);
                }else{army[m.target]-=moving;if(defender>=0)cache.total[defender]-=moving;}
            }
            r[5]=captured;r[7]=friendly||captured;
            if(captured&&kind==2){
                if(alive[defender])r[6]=defender;
                die(defender);credit(m.pid,defender);general[defender]=-1;
                replace(m.target,m.pid,army[m.target],1);
                for(int c=0;c<n;++c)if(owner[c]==defender)replace(c,m.pid,(army[c]+1)/2,structure[c]);
            }
            counts[5]+=moving>0;counts[6]+=captured;
        }
        return activity;
    }
    RULE_INLINE void before_growth(RULE_LOCAL const I *actions,RULE_LOCAL I *records,RULE_LOCAL U *counts){
        U old_total[8];
        for(int k=0;k<p;++k){old_total[k]=cache.total[k];cache.previous_land[k]=cache.territory[k];}
        bool activity=moves(actions,records,counts);
        b.idle_ticks[lane]=activity?0:b.idle_ticks[lane]+1;
        b.end_reason[lane]=1;
        if(b.tick[lane]>50000||b.idle_ticks[lane]>2000){
            b.end_reason[lane]=b.tick[lane]>50000?3:4;
            int ordered[8],count=0;
            for(int k=0;k<p;++k)if(alive[k]){
                int pos=count++;
                while(pos){int j=ordered[pos-1];
                    bool earlier=old_total[k]<old_total[j]||(old_total[k]==old_total[j]&&
                        (cache.previous_land[k]<cache.previous_land[j]||(cache.previous_land[k]==cache.previous_land[j]&&k<j)));
                    if(!earlier)break;ordered[pos]=j;--pos;}
                ordered[pos]=k;}
            for(int k=0;k<count-1;++k)die(ordered[k]);
        }
        ++b.tick[lane];
    }
    RULE_INLINE void after_growth(){
        I tick=b.tick[lane];
        for(int pid=0;pid<p;++pid)b.warn_afk[pb+pid]=0;
        for(int pid=0;pid<p;++pid)if(leave[pid]==tick&&alive[pid]){die(pid);b.end_reason[lane]=2;}
        if(b.afk_turn[lane]&&tick>=b.afk_turn[lane]-22&&tick<=b.afk_turn[lane])for(int pid=0;pid<p;++pid){
            bool safe=cache.previous_land[pid]>=11||cache.cities[pid]>0;
            if(alive[pid]&&!safe)for(int c=0;c<n&&!safe;++c)if(owner[c]>=0&&owner[c]!=pid&&visible(c,pid))safe=true;
            if(alive[pid]&&!safe){
                if(tick<b.afk_turn[lane])b.warn_afk[pb+pid]=1;
                else {leave[pid]=tick;die(pid);b.end_reason[lane]=5;}}
        }
        for(int pid=0;pid<p;++pid)if(leave[pid]>=0&&tick==leave[pid]+50)neutralize(pid);
        for(int slot=0;slot<p;++slot)if(general[slot]<0&&spawn[slot]>=0)
            for(int pid=0;pid<p;++pid)if(!(known[pid]&(1<<slot))&&visible(spawn[slot],pid))known[pid]|=1<<slot;
        finish();
    }
    RULE_INLINE void neutralize(int pid) {
        int g=general[pid];if(g<0)return;
        if(owner[g]==pid){
            for(int c=0;c<n;++c)if(owner[c]==pid)
                replace(c,!army[c]&&!structure[c]?-1:-2,army[c],c==g?1:structure[c]);
            int killer=-1;I maximum=0;
            for(int k=0;k<p;++k)if(alive[k]&&damage[k*p+pid]>maximum){killer=k;maximum=damage[k*p+pid];}
            if(killer>=0)credit(killer,pid);
            cache.growth_count=0;
            for(int c=0;c<n;++c)if(owner[c]>=0&&structure[c])growth[cache.growth_count++]=c;
        }
        general[pid]=-1;
    }
    RULE_INLINE void surrender(int pid) {
        if(b.terminated[lane]||!alive[pid]||leave[pid]>=0)return;
        leave[pid]=b.tick[lane]>0?b.tick[lane]:1;
        if(b.tick[lane]){die(pid);b.end_reason[lane]=2;finish();}
    }
    RULE_INLINE void finish() {
        int survivors=0,winner=-1;
        for(int pid=0;pid<p;++pid)if(alive[pid]){++survivors;winner=pid;}
        b.terminated[lane]=survivors<=1;
        if(survivors)b.winner[lane]=survivors==1?winner:-1;
        if(survivors==1){
            int last=-1;
            for(int pid=0;pid<p;++pid)if(deaths[pid]>=0&&(last<0||deaths[pid]>deaths[last]))last=pid;
            if(last>=0&&!(b.capture_mask[pb+winner]&(1<<last)))credit(winner,last);
        }
        if(!b.terminated[lane])b.end_reason[lane]=0;
    }
};
RULE_INLINE void reset_metadata(Batch s,Batch initial,int lane,int p){
    size_t pb=size_t(lane)*p;
    s.tick[lane]=initial.tick[lane];s.terminated[lane]=initial.terminated[lane];s.winner[lane]=initial.winner[lane];
    s.idle_ticks[lane]=initial.idle_ticks[lane];s.afk_turn[lane]=initial.afk_turn[lane];s.end_reason[lane]=initial.end_reason[lane];
    for(int k=0;k<p;++k){size_t i=pb+k;s.alive[i]=initial.alive[i];s.general_pos[i]=initial.general_pos[i];
        s.leave_tick[i]=initial.leave_tick[i];s.death_order[i]=initial.death_order[i];s.kills[i]=initial.kills[i];
        s.spawn_pos[i]=initial.spawn_pos[i];s.city_known[i]=initial.city_known[i];
        s.capture_mask[i]=initial.capture_mask[i];s.warn_afk[i]=initial.warn_afk[i];}
    for(int k=0;k<p*p;++k)s.damage[pb*p+k]=initial.damage[pb*p+k];
}
}
