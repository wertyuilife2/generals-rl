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
static int choose(Random &rng, const std::vector<double> &scores, double temperature) {
    double maximum=*std::max_element(scores.begin(),scores.end()), sum=0;
    std::vector<double> probabilities;
    for(double value:scores) {
        double weight=std::exp(std::max(-700.,std::min(0.,(value-maximum)/std::max(temperature,1e-6))));
        probabilities.push_back(weight); sum+=weight;
    }
    // Match Generator.choice: normalized probabilities, cumsum, renormalize CDF.
    double cumulative=0;
    for(double &value:probabilities) { value/=sum; cumulative+=value; value=cumulative; }
    double draw=rng.uniform();
    for(std::size_t i=0;i<probabilities.size();++i)
        if(draw<probabilities[i]/cumulative) return int(i);
    return int(probabilities.size())-1;
}

class Policy {
public:
    int kind, pid, h=0, w=0, n=0, search_attempts=0, strategy=MERGE;
    i64 path_tick=0, pending=0;
    std::vector<int> terrain, structure, owner, path;
    std::vector<i64> army, seen;
    std::vector<std::array<int,4>> graph;
    Policy(int k,int p):kind(k),pid(p) {}
    void remember(const Observation &o,const Options &options) {
        if(h!=o.height || w!=o.width) {
            h=o.height; w=o.width; n=h*w;
            terrain.assign(n,UNKNOWN); structure.assign(n,UNKNOWN); owner.assign(n,UNKNOWN_OWNER);
            army.assign(n,-1); seen.assign(n,-1); path.clear(); graph.resize(n);
            for(int c=0;c<n;++c) graph[c]={c>=w?c-w:-1,c%w<w-1?c+1:-1,c<n-w?c+w:-1,c%w?c-1:-1};
        }
        for(int c=0;c<n;++c) {
            if(o.visible[c]) {
                terrain[c]=o.terrain[c]; structure[c]=o.structure[c]; owner[c]=o.owner[c];
                army[c]=o.army[c]; seen[c]=o.tick;
            } else {
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
    std::vector<Candidate> candidates(const Observation &o,const Options &options,Random &rng) {
        std::vector<int> owned,sources,attacks,cities,expansion,frontiers(n);
        std::vector<i64> threats(n);
        for(int c=0;c<n;++c) {
            if(o.owner[c]==pid) {
                owned.push_back(c); if(o.army[c]>1) sources.push_back(c);
                std::uint64_t threat=0;
                for(int target:graph[c]) if(target>=0 && terrain[target]!=MOUNTAIN) {
                    frontiers[c]+=owner[target]!=pid;
                    if(o.owner[target]>=0 && o.owner[target]!=pid) threat+=std::max<i64>(o.army[target],0);
                }
                threats[c]=i64(threat);
            }
            if(owner[c]>=0 && owner[c]!=pid) attacks.push_back(c);
            if(structure[c]==CITY && owner[c]!=pid) cities.push_back(c);
            if((owner[c]==EMPTY || owner[c]==UNKNOWN_OWNER) && terrain[c]!=MOUNTAIN) expansion.push_back(c);
        }
        if(sources.empty()) return {};
        auto source_score=[&](int c){return double(o.army[c])*(1.+.25*(frontiers[c]!=0));};
        std::sort(sources.begin(),sources.end(),[&](int a,int b){double x=source_score(a),y=source_score(b); return x>y || (x==y && a<b);});
        sources.resize(std::min<int>(sources.size(),options.max_sources));
        if(sources.size()>1) {
            std::vector<double> scores; for(int c:sources) scores.push_back(std::log1p(double(o.army[c])));
            int index=choose(rng,scores,options.temperature), selected=sources[index];
            sources.erase(sources.begin()+index); sources.insert(sources.begin(),selected);
        }
        const std::array<std::array<int,3>,3> strategies={{{MERGE,EXPLORE,ATTACK},{EXPAND,CAPTURE,GATHER},{FORTIFY,GATHER,EXPAND}}};
        std::vector<Candidate> result;
        std::vector<int> distance(n);
        for(int source:sources) {
            for(int c=0;c<n;++c) distance[c]=std::abs(c/w-source/w)+std::abs(c%w-source%w);
            for(int s:strategies[kind]) {
                std::vector<Candidate> group;
                if(s==ATTACK || s==CAPTURE) {
                    for(int target:(s==ATTACK?attacks:cities)) {
                        i64 defending=std::max<i64>(army[target],0);
                        if(o.army[source]-distance[target]<=defending) continue;
                        double value=structure[target]==GENERAL?50.:structure[target]==CITY?18.:8.;
                        value=value/(distance[target]+1)+std::min(double(o.army[source]-defending),100.)*.01;
                        group.push_back({source,target,s,value});
                    }
                } else if(s==EXPLORE || s==EXPAND) {
                    auto nearest=expansion;
                    int count=std::min<int>(16,nearest.size());
                    std::partial_sort(nearest.begin(),nearest.begin()+count,nearest.end(),[&](int a,int b){return distance[a]<distance[b] || (distance[a]==distance[b] && a<b);});
                    for(int j=0;j<count;++j) {
                        int target=nearest[j]; if(distance[target]>=o.army[source]) continue;
                        int unknown=0; for(int neighbor:graph[target]) if(neighbor>=0 && terrain[neighbor]==UNKNOWN) ++unknown;
                        group.push_back({source,target,s,(5.+.4*unknown)/(distance[target]+1)});
                    }
                } else {
                    if(structure[source]==GENERAL && threats[source]>o.army[source]/2) continue;
                    for(int target:owned) {
                        if(target==source) continue;
                        if(s==FORTIFY ? threats[target]==0 : !(o.army[target]>o.army[source] || (o.army[target]==o.army[source] && target<source))) continue;
                        double value=3.+std::min(double(o.army[target]),100.)*.06+frontiers[target];
                        value+=std::min<i64>(threats[target],100)*.15;
                        value+=(threats[target]!=0 && structure[target]==GENERAL)*20.;
                        value/=distance[target]+1;
                        group.push_back({source,target,s,value});
                    }
                }
                int count=std::min<int>(4,group.size());
                std::partial_sort(group.begin(),group.begin()+count,group.end(),better);
                result.insert(result.end(),group.begin(),group.begin()+count);
            }
        }
        return result;
    }
    std::vector<int> plan(const Candidate &candidate,const Observation &o) {
        ++search_attempts;
        std::vector<std::uint8_t> blocked(n);
        for(int c=0;c<n;++c) blocked[c]=terrain[c]==MOUNTAIN ||
            (owner[c]!=pid && owner[c]!=EMPTY && owner[c]!=UNKNOWN_OWNER && army[c]>=o.army[candidate.source]-1);
        if(blocked[candidate.source] || blocked[candidate.target]) return {};
        std::vector<int> parent(n,-1), queue; queue.reserve(n);
        parent[candidate.source]=candidate.source; queue.push_back(candidate.source);
        for(std::size_t head=0;head<queue.size();++head) {
            int cell=queue[head]; if(cell==candidate.target) break;
            for(int next:graph[cell]) if(next>=0 && !blocked[next] && parent[next]<0) { parent[next]=cell; queue.push_back(next); }
        }
        if(parent[candidate.target]<0) return {};
        std::vector<int> route{candidate.target};
        while(route.back()!=candidate.source) route.push_back(parent[route.back()]);
        std::reverse(route.begin(),route.end());
        if(route.size()<2) return {};
        wide strength=o.army[candidate.source];
        for(std::size_t i=1;i<route.size();++i) {
            int source=route[i-1],target=route[i];
            strength=movable(strength,half(candidate.strategy,source,target,o));
            if(owner[target]==pid) strength+=std::max<i64>(0,o.army[target]);
            else strength-=std::max<i64>(0,army[target]);
            if(strength<=0) return {};
        }
        return route;
    }
    i64 random_action(const Observation &o,Random &rng) {
        std::vector<int> sources;
        for(int c=0;c<n;++c) if(o.owner[c]==pid && o.army[c]>1) sources.push_back(c);
        for(int i=int(sources.size())-1;i>0;--i) std::swap(sources[i],sources[rng.interval(i)]);
        double best=-std::numeric_limits<double>::infinity(); i64 action=0;
        for(int source:sources) for(int direction=0;direction<4;++direction) {
            int target=graph[source][direction]; if(target<0 || terrain[target]==MOUNTAIN) continue;
            bool mode=o.army[source]>6 && rng.uniform()<.3;
            wide moved=movable(o.army[source],mode);
            double score=rng.uniform()*5.;
            bool hostile=o.owner[target]!=pid && o.owner[target]!=EMPTY;
            score+=o.owner[target]==EMPTY?2.:hostile?(moved>o.army[target]?1.5:-3.):o.army[target]>o.army[source]?.5:0.;
            score+=(hostile && o.structure[target]==GENERAL && moved>o.army[target])*4.;
            if(score>best) { best=score; action=encode(source,direction,mode); }
        }
        return action;
    }
    i64 act(const Observation &o,const Options &options,Random &rng) {
        pending=0; search_attempts=0;
        if(o.player!=pid || o.terminated || !o.alive[pid]) { if(kind!=3) path.clear(); return 0; }
        remember(o,options);
        if(kind==3) return random_action(o,rng);
        if((pending=follow(o,options))) return pending;
        auto choices=candidates(o,options,rng);
        const std::array<std::array<int,3>,3> order={{{MERGE,EXPLORE,ATTACK},{EXPAND,CAPTURE,GATHER},{FORTIFY,GATHER,EXPAND}}};
        int attempts=std::min<int>(options.max_attempts,choices.size());
        for(int j=0;j<attempts;++j) {
            std::vector<int> strategies; std::vector<double> weights;
            for(int k=0;k<3;++k) if(std::any_of(choices.begin(),choices.end(),[&](const Candidate &c){return c.strategy==order[kind][k];})) {
                strategies.push_back(order[kind][k]); weights.push_back(options.weights[k]);
            }
            if(strategies.empty()) break;
            int s=strategies[choose(rng,weights,options.temperature)], best=-1;
            for(int k=0;k<int(choices.size());++k) if(choices[k].strategy==s && (best<0 || choices[k].score>choices[best].score)) best=k;
            Candidate candidate=choices[best]; choices.erase(choices.begin()+best);
            auto route=plan(candidate,o); if(route.empty()) continue;
            path=std::move(route); strategy=s; path_tick=o.tick;
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

// The adapter alone sees engine arrays. Policy::act always receives the same
// local fogged arrays as Python observe(..., "local"), with no privileged data.
// Scratch is reused across lanes/ticks; each Policy and NumPy stream stays owned
// by its Python controller. A call is serialized by the Python batch owner.
class PolicyBatch {
public:
    int lanes, h, w, n, players;
    std::vector<Policy *> policies;
    std::vector<Options> options;
    std::vector<Random> random;
    std::vector<std::uint8_t> terrain, structure, visible;
    std::vector<std::int16_t> owner;
    std::vector<i64> army;
    PolicyBatch(Policy *const *p,const Options *o,const Random *r,int count,int height,int width,int seats)
        :lanes(count),h(height),w(width),n(height*width),players(seats),
         policies(p,p+count),options(o,o+count),random(r,r+count),
         terrain(n),structure(n),visible(n),owner(n),army(n) {}
    i64 act(int lane,const std::uint8_t *raw_terrain,const std::uint8_t *raw_structure,
            const std::int16_t *raw_owner,const i64 *raw_army,const std::uint8_t *alive,
            i64 tick,bool terminated) {
        const int pid=policies[lane]->pid;
        std::fill(visible.begin(),visible.end(),0);
        for(int c=0;c<n;++c) if(raw_owner[c]==pid) {
            int y=c/w,x=c%w;
            for(int dy=std::max(0,y-1);dy<=std::min(h-1,y+1);++dy)
                for(int dx=std::max(0,x-1);dx<=std::min(w-1,x+1);++dx)
                    visible[dy*w+dx]=1;
        }
        for(int c=0;c<n;++c) {
            terrain[c]=visible[c]?raw_terrain[c]:UNKNOWN;
            structure[c]=visible[c]?raw_structure[c]:UNKNOWN;
            owner[c]=visible[c]?raw_owner[c]:UNKNOWN_OWNER;
            army[c]=visible[c]?raw_army[c]:-1;
        }
        Observation observation{terrain.data(),structure.data(),owner.data(),army.data(),
                                visible.data(),alive,h,w,players,pid,tick,terminated};
        return policies[lane]->act(observation,options[lane],random[lane]);
    }
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
                       const std::int16_t *owner,const i64 *army,const std::uint8_t *alive,
                       const i64 *tick,const std::uint8_t *terminated,const std::uint8_t *active,i64 *actions) {
    try {
        for(int lane=0;lane<batch->lanes;++lane) {
            actions[lane]=0;
            if(!active[lane]) continue;
            std::size_t base=std::size_t(lane)*batch->n;
            actions[lane]=batch->act(lane,terrain+base,structure+base,owner+base,army+base,
                                     alive+std::size_t(lane)*batch->players,tick[lane],terminated[lane]);
        }
        return 0;
    } catch(const std::exception &e) {error=e.what(); return -1;}
}
void baseline_batch_result(PolicyBatch *batch,const i64 *moves,const i64 *actions,
                          int action_stride,const std::uint8_t *active) {
    for(int lane=0;lane<batch->lanes;++lane) {
        if(active && !active[lane]) continue;
        Policy *policy=batch->policies[lane];
        const i64 *row=moves+(std::size_t(lane)*batch->players+policy->pid)*9;
        if(!row[0]) continue;
        i64 action=actions[std::size_t(lane)*action_stride+(action_stride==1?0:policy->pid)];
        // WAIT's encoded source is canonical -1 even if an engine row uses 0.
        int source=action>0?int((action-1)/8):-1;
        baseline_result(policy,policy->pid,action,int(row[1]),int(row[7]),source,int(row[3]));
    }
}
}
