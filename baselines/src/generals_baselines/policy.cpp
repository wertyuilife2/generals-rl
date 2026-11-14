// Observation-only policies. No engine state, map seed or privileged topology.
// Random draws use NumPy's supplied BitGenerator functions and owned state.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "generals_native.h"
#include "decision.h"

using i64 = std::int64_t;
using wide = __int128;
constexpr int EMPTY=-1, UNKNOWN_OWNER=-3, UNKNOWN=255, MOUNTAIN=1, CITY=1, GENERAL=2;
enum Strategy { MERGE, EXPLORE, ATTACK, EXPAND, CAPTURE, GATHER, FORTIFY };
struct Observation {
    const std::uint8_t *terrain, *structure;
    const std::int16_t *owner;
    const i64 *army;
    const std::uint8_t *visible, *alive;
    int height, width, players, player;
    i64 tick;
    int terminated;
};
struct Random {
    void *state;
    std::uint64_t (*next64)(void *);
    std::uint32_t (*next32)(void *);
    double (*next_double)(void *);
    double uniform() { return next_double(state); }
    std::uint32_t interval(std::uint32_t maximum) {
        std::uint32_t mask=maximum;
        mask |= mask>>1; mask |= mask>>2; mask |= mask>>4; mask |= mask>>8; mask |= mask>>16;
        std::uint32_t value;
        do { value=next32(state)&mask; } while(value>maximum);
        return value;
    }
};
struct Options {
    int max_sources, max_attempts, lifetime, memory_ttl;
    double temperature, weights[3];
};
struct Candidate { int source, target, strategy; double score; };
static bool better(const Candidate &a, const Candidate &b) {
    return a.score>b.score || (a.score==b.score && a.target<b.target);
}
static wide movable(wide army, bool half) { return std::max<wide>(0,half ? army/2 : army-1); }
static i64 encode(int source, int direction, bool half) { return 1+i64(source)*8+direction*2+half; }
static int choose(Random &rng, const double *scores, int count, double temperature, double *probabilities) {
    // A forced choice still consumes the original random draw.
    if(count==1) { rng.uniform(); return 0; }
    double maximum=*std::max_element(scores,scores+count), sum=0;
    for(int i=0;i<count;++i) {
        double weight=std::exp(std::max(-700.,std::min(0.,(scores[i]-maximum)/std::max(temperature,1e-6))));
        probabilities[i]=weight; sum+=weight;
    }
    // Preserve the normalized CDF's arithmetic and the single random draw.
    double cumulative=0;
    for(int i=0;i<count;++i) { probabilities[i]/=sum; cumulative+=probabilities[i]; probabilities[i]=cumulative; }
    double draw=rng.uniform();
    for(int i=0;i<count;++i)
        if(draw<probabilities[i]/cumulative) return i;
    return count-1;
}

class Policy {
public:
    int kind, pid, h=0, w=0, n=0, search_attempts=0, strategy=MERGE;
    i64 path_tick=0, pending=0;
    std::vector<int> terrain, structure, owner, path, route;
    std::vector<i64> army, seen;
    std::vector<std::array<int,4>> graph;
    // Per-seat scratch storage retains capacity across decisions.
    std::vector<int> owned,sources,attacks,cities,frontiers,cellX,cellY,searchParent,searchQueue;
    std::vector<i64> threats;
    std::vector<double> sourceScores,sourceProbabilities,friendlyValue;
    std::vector<std::uint8_t> explorable, unknownNeighbors;
    std::vector<Candidate> choices;
    Policy(int k,int p):kind(k),pid(p) {}
    template<bool CollectSources=false>
    void remember(const Observation &o,const Options &options) {
        if constexpr(CollectSources)sources.clear();
        if(h!=o.height || w!=o.width) {
            h=o.height; w=o.width; n=h*w;
            terrain.assign(n,UNKNOWN); structure.assign(n,UNKNOWN); owner.assign(n,UNKNOWN_OWNER);
            army.assign(n,-1); seen.assign(n,-1); path.clear(); graph.resize(n);
            cellX.resize(n); cellY.resize(n);
            for(int c=0;c<n;++c) {
                cellX[c]=c%w; cellY[c]=c/w;
                graph[c]={c>=w?c-w:-1,cellX[c]<w-1?c+1:-1,c<n-w?c+w:-1,cellX[c]?c-1:-1};
            }
        }
        for(int c=0;c<n;++c) {
            if constexpr(CollectSources)if(o.owner[c]==pid && o.army[c]>1)sources.push_back(c);
            if(o.visible[c]) {
                terrain[c]=o.terrain[c]; structure[c]=o.structure[c]; owner[c]=o.owner[c];
                army[c]=o.army[c]; seen[c]=o.tick;
            } else if(owner[c]!=UNKNOWN_OWNER || army[c]!=-1) {
                // Unseen cells already in the unknown state need no writes.
                if((o.tick-seen[c]>options.memory_ttl && structure[c]!=GENERAL) || owner[c]==pid)
                    owner[c]=UNKNOWN_OWNER;
                army[c]=-1;
                if(owner[c]>=0 && owner[c]<o.players && !o.alive[owner[c]]) {
                    owner[c]=UNKNOWN_OWNER;
                    if(structure[c]==GENERAL) structure[c]=CITY;
                }
            }
        }
    }
    bool half(int s,int source,int target,const Observation &o) const {
        if(kind==1) return s==EXPAND && o.owner[target]==EMPTY && o.army[source]>6;
        if(kind==2) return (s==EXPAND && o.owner[target]==EMPTY) || (s==FORTIFY && o.army[source]>10);
        return false;
    }
    i64 safe(int source,int target,int s,const Observation &o) const {
        int direction=-1;
        for(int d=0;d<4;++d) if(graph[source][d]==target) direction=d;
        if(direction<0 || o.owner[source]!=pid || o.army[source]<2 || o.terrain[target]==MOUNTAIN) return 0;
        bool mode=half(s,source,target,o);
        if(o.owner[target]!=pid && o.owner[target]!=EMPTY && o.owner[target]!=UNKNOWN_OWNER &&
           movable(o.army[source],mode)<=o.army[target]) return 0;
        return encode(source,direction,mode);
    }
    i64 follow(const Observation &o,const Options &options) {
        if(path.size()<2 || o.tick-path_tick>=options.lifetime) { path.clear(); return 0; }
        int target=path.back(), destination=owner[target];
        bool invalid=(strategy==ATTACK && (destination<0 || destination==pid)) ||
            (strategy==CAPTURE && (destination==pid || structure[target]!=CITY)) ||
            ((strategy==EXPLORE || strategy==EXPAND) && destination!=EMPTY && destination!=UNKNOWN_OWNER) ||
            ((strategy==MERGE || strategy==GATHER || strategy==FORTIFY) && destination!=pid);
        if(invalid) { path.clear(); return 0; }
        i64 action=safe(path[0],path[1],strategy,o);
        if(!action) path.clear();
        return action;
    }
    std::vector<Candidate> &candidates(const Observation &o,const Options &options,Random &rng) {
        owned.clear(); sources.clear(); attacks.clear(); cities.clear(); choices.clear();
        int expansionCount=0; explorable.resize(n);
        frontiers.resize(n); threats.resize(n); friendlyValue.resize(n);
        for(int c=0;c<n;++c) {
            if(o.owner[c]==pid) {
                owned.push_back(c); if(o.army[c]>1) sources.push_back(c);
                frontiers[c]=0;
                std::uint64_t threat=0;
                for(int target:graph[c]) if(target>=0 && terrain[target]!=MOUNTAIN) {
                    frontiers[c]+=owner[target]!=pid;
                    if(o.owner[target]>=0 && o.owner[target]!=pid) threat+=std::max<i64>(o.army[target],0);
                }
                threats[c]=i64(threat);
                double value=3.+std::min(double(o.army[c]),100.)*.06+frontiers[c];
                value+=std::min<i64>(threats[c],100)*.15;
                value+=(threats[c]!=0 && structure[c]==GENERAL)*20.;
                friendlyValue[c]=value;
            }
            if(owner[c]>=0 && owner[c]!=pid) attacks.push_back(c);
            if(structure[c]==CITY && owner[c]!=pid) cities.push_back(c);
            explorable[c]=(owner[c]==EMPTY || owner[c]==UNKNOWN_OWNER) && terrain[c]!=MOUNTAIN;
            expansionCount+=explorable[c];
        }
        if(sources.empty()) return choices;
        unknownNeighbors.assign(n,255);
        auto source_score=[&](int c){return double(o.army[c])*(1.+.25*(frontiers[c]!=0));};
        int sourceCount=std::min<int>(sources.size(),options.max_sources);
        std::partial_sort(sources.begin(),sources.begin()+sourceCount,sources.end(),[&](int a,int b){double x=source_score(a),y=source_score(b); return x>y || (x==y && a<b);});
        sources.resize(sourceCount);
        if(sources.size()>1) {
            sourceScores.clear(); for(int c:sources) sourceScores.push_back(std::log1p(double(o.army[c])));
            sourceProbabilities.resize(sourceScores.size());
            int index=choose(rng,sourceScores.data(),int(sourceScores.size()),options.temperature,sourceProbabilities.data()), selected=sources[index];
            sources.erase(sources.begin()+index); sources.insert(sources.begin(),selected);
        }
        const std::array<std::array<int,3>,3> strategies={{{MERGE,EXPLORE,ATTACK},{EXPAND,CAPTURE,GATHER},{FORTIFY,GATHER,EXPAND}}};
        choices.reserve(sources.size()*12);
        for(int source:sources) {
            const int sy=cellY[source],sx=cellX[source];
            auto distance=[&](int target) { return std::abs(cellY[target]-sy)+std::abs(cellX[target]-sx); };
            for(int s:strategies[kind]) {
                // Targets are unique within a group; score/target is a total
                // order, so retaining its first four preserves partial_sort.
                std::array<Candidate,4> group;
                int count=0;
                auto add=[&](Candidate candidate) {
                    int at=count;
                    if(at==4) { if(!better(candidate,group[3])) return; at=3; }
                    else ++count;
                    while(at>0 && better(candidate,group[at-1])) { group[at]=group[at-1]; --at; }
                    group[at]=candidate;
                };
                if(s==ATTACK || s==CAPTURE) {
                    for(int target:(s==ATTACK?attacks:cities)) {
                        i64 defending=std::max<i64>(army[target],0);
                        int d=distance(target);
                        if(o.army[source]-d<=defending) continue;
                        double value=structure[target]==GENERAL?50.:structure[target]==CITY?18.:8.;
                        value=value/(d+1)+std::min(double(o.army[source]-defending),100.)*.01;
                        add({source,target,s,value});
                    }
                } else if(s==EXPLORE || s==EXPAND) {
                    std::array<int,16> nearest, nearestDistance;
                    int count=std::min(16,expansionCount),found=0;
                    int horizontalReach=std::max(sx,w-1-sx);
                    // Manhattan rings ordered by row, then column give exactly
                    // the original (distance, cell index) ordering.
                    for(int d=0;found<count;++d) {
                        for(int y=std::max(0,sy-d);y<=std::min(h-1,sy+d) && found<count;++y) {
                            int dx=d-std::abs(y-sy);
                            // Skip the middle band whose two endpoints are
                            // outside the map, including very narrow maps.
                            if(dx>horizontalReach) { y=sy+d-horizontalReach-1; continue; }
                            int left=sx-dx,right=sx+dx,row=y*w;
                            if(left>=0 && explorable[row+left]) {nearestDistance[found]=d;nearest[found++]=row+left;}
                            if(found<count && dx>0 && right<w && explorable[row+right]) {nearestDistance[found]=d;nearest[found++]=row+right;}
                        }
                    }
                    for(int j=0;j<count;++j) {
                        int target=nearest[j],d=nearestDistance[j]; if(d>=o.army[source]) continue;
                        int unknown=unknownNeighbors[target];
                        if(unknown==255) {
                            unknown=0;for(int neighbor:graph[target])if(neighbor>=0 && terrain[neighbor]==UNKNOWN)++unknown;
                            unknownNeighbors[target]=unknown;
                        }
                        add({source,target,s,(5.+.4*unknown)/(d+1)});
                    }
                } else {
                    if(structure[source]==GENERAL && threats[source]>o.army[source]/2) continue;
                    for(int target:owned) {
                        if(target==source) continue;
                        if(s==FORTIFY ? threats[target]==0 : !(o.army[target]>o.army[source] || (o.army[target]==o.army[source] && target<source))) continue;
                        double value=friendlyValue[target]/(distance(target)+1);
                        add({source,target,s,value});
                    }
                }
                choices.insert(choices.end(),group.begin(),group.begin()+count);
            }
        }
        return choices;
    }
    bool plan(const Candidate &candidate,const Observation &o) {
        ++search_attempts;
        const i64 limit=o.army[candidate.source]-1;
        auto blocked=[&](int c) {
            return terrain[c]==MOUNTAIN ||
                (owner[c]!=pid && owner[c]!=EMPTY && owner[c]!=UNKNOWN_OWNER && army[c]>=limit);
        };
        if(blocked(candidate.source) || blocked(candidate.target)) return false;
        searchParent.assign(n,-1); searchQueue.clear(); searchQueue.reserve(n);
        auto &parent=searchParent; auto &queue=searchQueue;
        parent[candidate.source]=candidate.source; queue.push_back(candidate.source);
        // The first discovery fixes the same parent chain as dequeueing the
        // target, without exploring the rest of its breadth-first frontier.
        for(std::size_t head=0;head<queue.size() && parent[candidate.target]<0;++head) {
            int cell=queue[head];
            for(int next:graph[cell]) if(next>=0 && parent[next]==-1) {
                if(blocked(next)) {parent[next]=-2;continue;}
                parent[next]=cell;queue.push_back(next);
            }
        }
        if(parent[candidate.target]<0) return false;
        route.clear();route.push_back(candidate.target);
        while(route.back()!=candidate.source) route.push_back(parent[route.back()]);
        std::reverse(route.begin(),route.end());
        if(route.size()<2) return false;
        wide strength=o.army[candidate.source];
        for(std::size_t i=1;i<route.size();++i) {
            int source=route[i-1],target=route[i];
            strength=movable(strength,half(candidate.strategy,source,target,o));
            if(owner[target]==pid) strength+=std::max<i64>(0,o.army[target]);
            else strength-=std::max<i64>(0,army[target]);
            if(strength<=0) return false;
        }
        return true;
    }
    i64 random_action(const Observation &o,Random &rng) {
        // NumPy callbacks mutate only RNG state; keep observation and callback
        // addresses local across these opaque calls.
        void *const state=rng.state;
        const auto next32=rng.next32;
        const auto uniform=rng.next_double;
        const auto *const armies=o.army;
        const auto *const owners=o.owner;
        const auto *const structures=o.structure;
        const auto *const neighbors=graph.data();
        const auto *const terrains=terrain.data();
        const int player=pid;
        auto *const shuffled=sources.data();
        const int count=int(sources.size());
        // Descending Fisher-Yates bounds share a rejection mask until the next
        // power of two. Keep exactly the same draws as Random::interval.
        std::uint32_t mask=count>1?std::uint32_t(count-1):0;
        mask|=mask>>1; mask|=mask>>2; mask|=mask>>4; mask|=mask>>8; mask|=mask>>16;
        for(int i=count-1;i>0;--i) {
            if(std::uint32_t(i)<=(mask>>1)) mask>>=1;
            std::uint32_t index;
            do { index=next32(state)&mask; } while(index>std::uint32_t(i));
            std::swap(shuffled[i],shuffled[index]);
        }
        double best=-std::numeric_limits<double>::infinity(); i64 action=0;
        for(int i=0;i<count;++i) {
            const int source=shuffled[i];
            const i64 sourceArmy=armies[source],full=sourceArmy-1,halfArmy=sourceArmy/2;
            const auto adjacent=neighbors[source];
            for(int direction=0;direction<4;++direction) {
                int target=adjacent[direction]; if(target<0 || terrains[target]==MOUNTAIN) continue;
                const int targetOwner=owners[target];
                const i64 targetArmy=armies[target];
                const bool hostile=targetOwner!=player && targetOwner!=EMPTY;
                const bool general=structures[target]==GENERAL;
                bool mode=sourceArmy>6 && uniform(state)<.3;
                i64 moved=mode?halfArmy:full;
                double score=uniform(state)*5.;
                score+=targetOwner==EMPTY?2.:hostile?(moved>targetArmy?1.5:-3.):targetArmy>sourceArmy?.5:0.;
                score+=(hostile && general && moved>targetArmy)*4.;
                if(score>best) { best=score; action=encode(source,direction,mode); }
            }
        }
        return action;
    }
    i64 act(const Observation &o,const Options &options,Random &rng) {
        pending=0; search_attempts=0;
        if(o.player!=pid || o.terminated || !o.alive[pid]) { if(kind!=3) path.clear(); return 0; }
        if(kind==3) {remember<true>(o,options);return random_action(o,rng);}
        remember(o,options);
        if((pending=follow(o,options))) return pending;
        auto &choices=candidates(o,options,rng);
        const std::array<std::array<int,3>,3> order={{{MERGE,EXPLORE,ATTACK},{EXPAND,CAPTURE,GATHER},{FORTIFY,GATHER,EXPAND}}};
        int remaining[7]{};
        for(const auto &candidate:choices) ++remaining[candidate.strategy];
        int attempts=std::min<int>(options.max_attempts,choices.size());
        for(int j=0;j<attempts;++j) {
            int strategies[3],count=0; double weights[3],probabilities[3];
            for(int k=0;k<3;++k) if(remaining[order[kind][k]]) {
                strategies[count]=order[kind][k]; weights[count++]=options.weights[k];
            }
            if(!count) break;
            int s=strategies[choose(rng,weights,count,options.temperature,probabilities)], best=-1;
            for(int k=0;k<int(choices.size());++k) if(choices[k].strategy==s && (best<0 || choices[k].score>choices[best].score)) best=k;
            Candidate candidate=choices[best]; choices.erase(choices.begin()+best); --remaining[s];
            if(!plan(candidate,o)) continue;
            path.swap(route); strategy=s; path_tick=o.tick;
            if((pending=follow(o,options))) return pending;
        }
        int best=-1;
        for(int source=0;source<n;++source) if(o.owner[source]==pid && o.army[source]>1)
            for(int target:graph[source]) if(target>=0 && terrain[target]!=MOUNTAIN && o.owner[target]!=pid) {
                i64 action=safe(source,target,EXPAND,o); if(!action) continue;
                int value=o.structure[target]==GENERAL?20:1;
                if(value>best) { best=value; pending=action; }
            }
        return pending;
    }
};

static thread_local std::string error;

// Only public, already masked observations cross this boundary. One serial
// native call handles independent planners and RNG streams for every lane.
class PolicyBatch {
public:
    int lanes, h, w, n, players;
    std::vector<Policy *> policies;
    std::vector<Options> options;
    std::vector<Random> random;
    PolicyBatch(Policy *const *p,const Options *o,const Random *r,int count,int height,int width,int seats)
        :lanes(count),h(height),w(width),n(height*width),players(seats),
         policies(p,p+count),options(o,o+count),random(r,r+count) {}
};

extern "C" {
const char *baseline_error() { return error.c_str(); }
void *baseline_create(int kind,int pid) {
    try { if(kind<0 || kind>3 || pid<0 || pid>7) throw std::runtime_error("invalid baseline kind/player"); return new Policy(kind,pid); }
    catch(const std::exception &e) {error=e.what(); return nullptr;}
}
void baseline_destroy(Policy *p) { delete p; }
i64 baseline_act(Policy *p,const Observation *o,const Options *options,Random *rng) {
    try { return p->act(*o,*options,*rng); }
    catch(const std::exception &e) {error=e.what(); return -1;}
}
void baseline_result(Policy *p,int player,i64 action,int valid,int owned,int source,int target) {
    if(player!=p->pid || !p->pending || action!=p->pending) return;
    if(valid && owned && p->path.size()>=2 && p->path[0]==source && p->path[1]==target) {
        p->path.erase(p->path.begin()); if(p->path.size()<2) p->path.clear();
    } else p->path.clear();
    p->pending=0;
}
i64 baseline_info(Policy *p,int field) {
    switch(field) {case 0:return p->h;case 1:return p->w;case 2:return p->pending;case 3:return p->search_attempts;default:return p->path.size();}
}
int baseline_copy(Policy *p,int field,i64 *out) {
    int count=field==6?p->path.size():p->n;
    for(int c=0;c<count;++c) switch(field) {
        case 0:out[c]=p->terrain[c];break;case 1:out[c]=p->structure[c];break;case 2:out[c]=p->owner[c];break;
        case 3:out[c]=p->army[c];break;case 4:out[c]=p->seen[c];break;case 6:out[c]=p->path[c];break;
    }
    return count;
}
int baseline_set_path(Policy *p,const i64 *path,int size) {
    try { p->path.assign(path,path+size); return 0; }
    catch(const std::exception &e) {error=e.what(); return -1;}
}
void baseline_set_pending(Policy *p,i64 action) {p->pending=action;}
void *baseline_batch_create(Policy *const *policies,const Options *options,const Random *random,
                            int lanes,int height,int width,int players) {
    try {
        if(lanes<1 || height<1 || width<1 || i64(height)*width>1024*1024 || players<2 || players>8)
            throw std::runtime_error("invalid baseline batch dimensions");
        for(int lane=0;lane<lanes;++lane)
            if(!policies[lane] || policies[lane]->pid>=players)
                throw std::runtime_error("invalid baseline batch player");
        return new PolicyBatch(policies,options,random,lanes,height,width,players);
    } catch(const std::exception &e) {error=e.what(); return nullptr;}
}
void baseline_batch_destroy(PolicyBatch *batch) {delete batch;}
int baseline_batch_act(PolicyBatch *batch,const std::uint8_t *terrain,const std::uint8_t *structure,
                       const std::int16_t *owner,const i64 *army,const std::uint8_t *visible,const std::uint8_t *alive,
                       const i64 *tick,const std::uint8_t *terminated,const std::uint8_t *active,i64 *actions) {
    try {
        for(int lane=0;lane<batch->lanes;++lane) {
            actions[lane]=0;
            if(!active[lane]) continue;
            std::size_t base=std::size_t(lane)*batch->n;
            Observation observation{terrain+base,structure+base,owner+base,army+base,visible+base,
                alive+std::size_t(lane)*batch->players,batch->h,batch->w,batch->players,
                batch->policies[lane]->pid,tick[lane],terminated[lane]};
            actions[lane]=batch->policies[lane]->act(observation,batch->options[lane],batch->random[lane]);
        }
        return 0;
    } catch(const std::exception &e) {error=e.what(); return -1;}
}
void baseline_batch_result(PolicyBatch *batch,const i64 *moves,const i64 *actions,
                          const std::uint8_t *active) {
    for(int lane=0;lane<batch->lanes;++lane) {
        if(active && !active[lane]) continue;
        Policy *policy=batch->policies[lane];
        const i64 *row=moves+std::size_t(lane)*9;
        if(!row[0]) continue;
        i64 action=actions[lane];
        // WAIT's encoded source is canonical -1 even if an engine row uses 0.
        int source=action>0?int((action-1)/8):-1;
        baseline_result(policy,policy->pid,action,int(row[1]),int(row[7]),source,int(row[3]));
    }
}
i64 baseline_bound_act(void *context,const GeneralsBatchView *view,i64 *actions) {
    auto *batch=static_cast<PolicyBatch *>(context);
    const auto &o=view->observation;
    for(int lane=0;lane<batch->lanes;++lane) {
        actions[lane]=0;
        if(!o.active[lane]) continue;
        const std::size_t base=std::size_t(lane)*batch->n;
        Observation observation{o.terrain+base,o.structure+base,o.owner+base,o.army+base,
            reinterpret_cast<const std::uint8_t *>(o.visible)+base,
            reinterpret_cast<const std::uint8_t *>(o.alive)+std::size_t(lane)*batch->players,
            batch->h,batch->w,batch->players,batch->policies[lane]->pid,o.tick[lane],o.terminated[lane]};
        try { actions[lane]=batch->policies[lane]->act(observation,batch->options[lane],batch->random[lane]); }
        catch(const std::exception &e) {error=e.what();return lane;}
    }
    return -1;
}
i64 baseline_bound_one(void *context,const GeneralsBatchView *view,int lane,i64 *metrics) {
    auto *batch=static_cast<PolicyBatch *>(context);
    const auto &o=view->observation;
    const std::size_t base=std::size_t(lane)*batch->n;
    Observation observation{o.terrain+base,o.structure+base,o.owner+base,o.army+base,
        reinterpret_cast<const std::uint8_t *>(o.visible)+base,
        reinterpret_cast<const std::uint8_t *>(o.alive)+std::size_t(lane)*batch->players,
        batch->h,batch->w,batch->players,batch->policies[lane]->pid,o.tick[lane],o.terminated[lane]};
    i64 action=0;int status=0;
    try {action=batch->policies[lane]->act(observation,batch->options[lane],batch->random[lane]);}
    catch(const std::exception &e){error=e.what();status=2;}
    catch(...){error="native policy exception";status=2;}
    metrics[7]=status;return action;
}
void baseline_bound_result(void *context,const GeneralsFeedback *result) {
    baseline_batch_result(static_cast<PolicyBatch *>(context),result->moves,result->actions,
                          reinterpret_cast<const std::uint8_t *>(result->active));
}
}
