// Native result/recording projection. Rules and fog construction stay in env.
#pragma once
#include "generals_native.h"
#include "sha256.h"
#include <numeric>
#include <sstream>
#include <tuple>
#include <type_traits>
namespace record {
using I=std::int64_t;
inline std::string scalar(bool v){return v?"true":"false";}
template<class T> std::string scalar(T v){return std::to_string(v);}
inline std::string quote(const std::string &v){return "\""+v+"\"";}
template<class T> std::string array(const T *v,int n,int stride=1){std::string s="[";
    for(int i=0;i<n;++i){if(i)s+=',';s+=scalar(v[i*stride]);}return s+"]";}
template<class T> std::string matrix(const T *v,int h,int w){std::string s="[";
    for(int y=0;y<h;++y){if(y)s+=',';s+=array(v+y*w,w);}return s+"]";}
inline std::string joined(const std::vector<std::string> &v){std::string s="[";
    for(std::size_t i=0;i<v.size();++i){if(i)s+=',';s+=v[i];}return s+"]";}
struct Object {std::string s="{";bool first=true;
    void add(const char *k,const std::string &v){if(!first)s+=',';first=false;s+=quote(k)+":"+v;}
    std::string done(){return s+"}";}};
struct Lane {
    GeneralsStateView s;const I *stats;int lane,h,w,p,n;std::size_t base,pb;
    Lane(const GeneralsBackend &b,int l,int height,int width,int players):
        stats(b.stats+std::size_t(l)*players*3),lane(l),h(height),w(width),p(players),n(h*w),
        base(std::size_t(l)*n),pb(std::size_t(l)*p){b.snapshot(b.context,&s);}
    std::string digest(const std::string &topology)const{
        SHA256 hash;hash.add(topology);
        hash.integers(s.structure+base,n);hash.integers(s.owner+base,n);hash.integers(s.army+base,n);
        hash.integers(s.alive+pb,p);hash.integers(s.general_pos+pb,p);hash.integers(s.leave_tick+pb,p);
        hash.integers(s.death_order+pb,p);hash.integers(s.kills+pb,p);hash.integers(s.damage+pb*p,p*p);
        hash.integers(s.spawn_pos+pb,p);hash.integers(s.city_known+pb,p);hash.integers(s.capture_mask+pb,p);hash.integers(s.warn_afk+pb,p);
        hash.add("["+scalar(s.tick[lane])+", "+scalar(s.idle_ticks[lane])+", "+scalar(s.afk_turn[lane])+", "+scalar(s.end_reason[lane])+", "+scalar(s.terminated[lane])+", "+(s.winner[lane]<0?"null":scalar(s.winner[lane]))+"]");
        return hash.finish();
    }
    std::string state()const{Object o;
        o.add("terrain",matrix(s.terrain+base,h,w));o.add("structure",matrix(s.structure+base,h,w));
        o.add("owner",matrix(s.owner+base,h,w));o.add("army",matrix(s.army+base,h,w));
#define PLAYER(key) o.add(#key,array(s.key+pb,p))
        PLAYER(alive);PLAYER(general_pos);PLAYER(leave_tick);PLAYER(death_order);PLAYER(kills);
        PLAYER(spawn_pos);PLAYER(city_known);PLAYER(capture_mask);PLAYER(warn_afk);
#undef PLAYER
        o.add("damage",matrix(s.damage+pb*p,p,p));
#define SCALAR(key) o.add(#key,scalar(s.key[lane]))
        SCALAR(tick);SCALAR(idle_ticks);SCALAR(afk_turn);SCALAR(end_reason);SCALAR(terminated);
#undef SCALAR
        o.add("winner_id",s.winner[lane]<0?"null":scalar(s.winner[lane]));return o.done();
    }
    std::vector<int> ranking()const{std::vector<int> order(p);std::iota(order.begin(),order.end(),0);
        auto key=[&](int k){bool alive=s.alive[pb+k];return std::make_tuple(-int(s.kills[pb+k]>0),!alive,
            alive?-stats[k*3]:-I(s.death_order[pb+k]),alive?-stats[k*3+1]:I(0),k);};
        std::sort(order.begin(),order.end(),[&](int a,int b){return key(a)<key(b);});return order;}
    std::string totals(const I *v)const{Object o;o.add("army",array(v,p,3));o.add("territory",array(v+1,p,3));o.add("cities",array(v+2,p,3));return o.done();}
    std::string board(const GeneralsObservation &obs,int pid,const bool *forfeited,const std::vector<I> *previous=nullptr)const {
        Object o;o.add("tick",scalar(s.tick[lane]));
        if(previous){std::vector<std::string> cells;for(int c=0;c<n;++c){auto at=base+c;
            if((*previous)[c*3]!=s.owner[at]||(*previous)[c*3+1]!=s.army[at]||(*previous)[c*3+2]!=s.structure[at])
                cells.push_back("["+scalar(c)+","+scalar(s.owner[at])+","+scalar(s.army[at])+","+scalar(s.structure[at])+"]");}
            o.add("cells",joined(cells));
        }else{
            o.add("width",scalar(w));o.add("height",scalar(h));o.add("player_id",scalar(pid));o.add("general_pos",scalar(obs.general_pos[lane]));
#define BOARD(key) o.add(#key,matrix(obs.key+base,h,w))
            BOARD(terrain);BOARD(structure);BOARD(owner);BOARD(army);BOARD(visible);BOARD(fog_obstacles);
#undef BOARD
        }
#define PUBLIC(key) o.add(#key,array(obs.key+pb,p))
        PUBLIC(alive);PUBLIC(has_kill);PUBLIC(warn_afk);PUBLIC(neutralize_in);
#undef PUBLIC
        auto order=ranking();o.add("ranking",array(order.data(),p));o.add("stats",totals(obs.stats+pb*3));
        std::vector<int> departed;for(int k=0;k<p;++k)if(forfeited[pb+k])departed.push_back(k);
        o.add("forfeited",array(departed.data(),departed.size()));
        // Web checks only this public capability, never hidden state or game rules.
        bool playing=pid>=0&&!s.terminated[lane]&&s.alive[pb+pid]&&!forfeited[pb+pid];
        o.add("can_play",scalar(playing));o.add("max_action",scalar(n*8));
        return o.done();
    }
    std::string result(const I *invalid,const I *eliminations,const std::string &initial,const std::string &topology)const{
        Object o;o.add("ticks",scalar(s.tick[lane]));o.add("winner_seat",s.winner[lane]<0?"null":scalar(s.winner[lane]));
        static const char *reason[]={"ongoing","capture","surrender","game_limit","idle_limit","afk"};
        o.add("reason",quote(s.terminated[lane]?reason[s.end_reason[lane]]:"tick_limit"));
        std::vector<std::string> deaths;for(int k=0;k<p;++k)deaths.push_back(eliminations[k]<0?"null":scalar(eliminations[k]));
        o.add("elimination_ticks",joined(deaths));o.add("invalid_actions",array(invalid,p));o.add("final_alive",array(s.alive+pb,p));
        o.add("initial_digest",quote(initial));o.add("final_digest",quote(digest(topology)));o.add("final_stats",totals(stats));
        o.add("kills",array(s.kills+pb,p));o.add("death_order",array(s.death_order+pb,p));
        std::vector<std::string> places(p);auto order=ranking();for(int k=0;k<p;++k)
            places[k]=p==2?(s.winner[lane]<0?"1.5":s.winner[lane]==k?"1":"2"):scalar(std::find(order.begin(),order.end(),k)-order.begin()+1);
        o.add("placements",joined(places));return o.done();
    }
};
struct Trace {std::string topology,initial,initial_digest;bool enabled,detailed;int players;
    std::vector<I> actions;
    std::vector<std::string> digests,frames;std::vector<I> previous;
    Trace(const Lane &l,const char *id,int record):topology(id),initial(l.state()),initial_digest(l.digest(topology)),
        enabled(record!=0),detailed(record==2),players(l.p){if(enabled)actions.reserve(1024*l.p);}
    void frame(const Lane &l,const GeneralsObservation &o,const bool *forfeited,bool replace=false){
        if(!detailed)return;
        // Boundary departures replace the current frame; a full frame also resets the delta base.
        auto frame=l.board(o,-1,forfeited,replace||frames.empty()?nullptr:&previous);
        if(replace&&!frames.empty())frames.back()=frame;else frames.push_back(frame);
        previous.resize(l.n*3);for(int c=0;c<l.n;++c){previous[c*3]=l.s.owner[l.base+c];previous[c*3+1]=l.s.army[l.base+c];previous[c*3+2]=l.s.structure[l.base+c];}
    }
    void tick(const Lane &l,const I *joint){if(!enabled)return;
        actions.insert(actions.end(),joint,joint+l.p);
        if(detailed)digests.push_back(quote(l.digest(topology)));
    }
    std::string json(const Lane &l)const{Object o;o.add("initial",initial);
        o.add("action_encoding",quote("generals-v1"));o.add("actions",matrix(actions.data(),actions.size()/players,players));
        o.add("final_digest",quote(l.digest(topology)));
        if(detailed){o.add("digests",joined(digests));o.add("frames",joined(frames));}return o.done();}
};
}
