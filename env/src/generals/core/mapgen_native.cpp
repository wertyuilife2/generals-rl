// Native map-generation kernels. Random draws use NumPy's live BitGenerator;
// Python retains seed construction and array ownership. No simulator ABI changes.
#include "numpy/random/bitgen.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

namespace {
struct Graph {
    int w, h, n;
    std::vector<std::vector<int>> neighbors;
    Graph(int width, int height, bool curriculum=false):w(width),h(height),n(w*h),neighbors(n) {
        for(int c=0;c<n;++c) {
            auto &a=neighbors[c]; int x=c%w,y=c/w;
            if(y) a.push_back(c-w);
            if(curriculum) {
                if(x+1<w) a.push_back(c+1);
                if(y+1<h) a.push_back(c+w);
                if(x) a.push_back(c-1);
            } else {
                if(y+1<h) a.push_back(c+w);
                if(x) a.push_back(c-1);
                if(x+1<w) a.push_back(c+1);
            }
        }
    }
    int manhattan(int a,int b) const {return std::abs(a/w-b/w)+std::abs(a%w-b%w);}
};

// Generator.integers(high), default int64, takes the uint32 Lemire path for
// our small positive ranges. Preserve rejection draws and NumPy's uint32 cache.
int integer(bitgen_t *r,uint32_t high) {
    if(high<=1) return 0;
    const uint32_t threshold=uint32_t(-high)%high;
    for(;;) {
        uint64_t product=uint64_t(r->next_uint32(r->state))*high;
        if(uint32_t(product)>=threshold) return int(product>>32);
    }
}
double uniform(bitgen_t *r) {return r->next_double(r->state);}

double spawn_score(int source,const uint8_t *mountains,const Graph &g) {
    std::array<int,15> counts{};
    std::vector<uint8_t> seen(g.n,0); seen[source]=1;
    std::deque<std::pair<int,int>> queue{{source,0}};
    while(!queue.empty()) {
        auto [cell,depth]=queue.front();queue.pop_front();++counts[depth];
        if(depth<14) for(int other:g.neighbors[cell]) if(!seen[other]&&!mountains[other]) {
            seen[other]=1;queue.emplace_back(other,depth+1);
        }
    }
    counts[1]=std::min(3,counts[1]);
    double weight=int(std::pow(1.18,15)),score=0;
    for(int d=1;d<15;++d) {
        score+=weight*counts[d]; weight/=d<7?1.18:1.1;
        if(d<5) weight/=1.6;
    }
    return score;
}

struct Group {std::vector<int> cells;bool neutral=false;};
struct Info {int delta,total;bool neutral,valid;};

struct DuelCities {
    const uint8_t *mountains;
    const Graph &g;
    bitgen_t *rng;
    std::array<int,2> generals;
    std::vector<int> remaining,positions;
    std::vector<uint8_t> cities;
    std::vector<std::shared_ptr<Group>> groups;
    std::array<std::vector<int>,2> distances;
    double deficit=0;
    static constexpr int far=17;
    static constexpr double tolerance=3.375;
    DuelCities(const uint8_t *m,const int32_t *k,const Graph &graph,bitgen_t *r):
        mountains(m),g(graph),rng(r),generals{k[0],k[1]},positions(g.n,-1),cities(g.n,0) {
        for(int c=0;c<g.n;++c) if(mountains[c]) {positions[c]=int(remaining.size());remaining.push_back(c);}
        rebuild();
    }
    bool blocked(int c) const {return mountains[c]&&!cities[c];}
    bool take(int c) {
        int i=positions[c];if(i<0)return false;
        int last=remaining.back();remaining.pop_back();positions[c]=-1;
        if(i<int(remaining.size())) {remaining[i]=last;positions[last]=i;}
        return true;
    }
    void put_back(int c) {if(positions[c]<0){positions[c]=int(remaining.size());remaining.push_back(c);}}
    void relax(std::vector<int> &d,int start) {
        std::deque<int> queue{start};
        while(!queue.empty()) {
            int source=queue.front();queue.pop_front();int value=d[source]+1;
            for(int target:g.neighbors[source]) if(value<d[target]) {
                d[target]=value;if(!blocked(target))queue.push_back(target);
            }
        }
    }
    Info info(int c,bool reachable=true) const {
        int a=distances[0][c],b=distances[1][c];
        return {a-b,a+b,a==b||std::min(a,b)>=far,!reachable||(a<g.n+1&&b<g.n+1)};
    }
    static int pair_deficit(Info a,Info b) {return a.delta<=b.delta?a.total-b.total:b.total-a.total;}
    void score() {
        deficit=0;
        for(const auto &group:groups) if(!group->neutral) {
            int sum=0;for(int c:group->cells)sum+=info(c,false).delta;
            deficit+=1.5*sum;
            if(group->cells.size()==2)deficit+=pair_deficit(info(group->cells[0],false),info(group->cells[1],false));
        }
    }
    void rebuild() {
        for(int p=0;p<2;++p){distances[p].assign(g.n,g.n+1);distances[p][generals[p]]=0;relax(distances[p],generals[p]);}
        score();
    }
    void erase(const std::shared_ptr<Group> &group) {
        auto it=std::find(groups.begin(),groups.end(),group);if(it!=groups.end())groups.erase(it);
    }
    void open(int c) {
        cities[c]=1;
        for(auto &d:distances) {
            int best=g.n+1;for(int x:g.neighbors[c])if(!blocked(x))best=std::min(best,d[x]+1);
            d[c]=std::min(d[c],best);if(d[c]<g.n+1)relax(d,c);
        }
        bool removed=false;
        auto snapshot=groups;
        for(auto it=snapshot.rbegin();it!=snapshot.rend();++it) {
            const auto &group=*it;
            if(group->neutral) {
                Info x=info(group->cells[0]);
                if(x.valid&&!x.neutral) {
                    cities[group->cells[0]]=0;put_back(group->cells[0]);erase(group);removed=true;
                }
            }
        }
        if(removed)rebuild();score();
    }
    std::vector<int> partners(Info x) const {
        std::vector<int> result;
        for(int c:remaining) {
            Info y=info(c);if(!y.valid||y.neutral||x.delta*y.delta>=0)continue;
            double partial=deficit+1.5*y.delta,pair=pair_deficit(x,y);
            if(std::max({std::abs(partial),std::abs(pair),std::abs(partial+pair)})<tolerance)result.push_back(c);
        }
        return result;
    }
    void remove(const std::shared_ptr<Group> &group) {
        if(std::find(groups.begin(),groups.end(),group)==groups.end())return;
        erase(group);for(int c:group->cells){cities[c]=0;put_back(c);}rebuild();
    }
    bool repair() {
        const int limit=std::max({4*int(groups.size()),2*int(remaining.size()),1});
        for(int step=0;step<limit;++step) {
            bool changed=false;auto snapshot=groups;
            for(const auto &group:snapshot) {
                if(group->neutral||group->cells.size()!=2)continue;
                Info a=info(group->cells[0]),b=info(group->cells[1]);
                if(!a.valid||!b.valid)continue;
                int bad=a.neutral?0:(b.neutral||((a.delta<0)==(b.delta<0)))?1:-1;
                if(bad<0)continue;
                int orphan=group->cells[bad];group->cells.erase(group->cells.begin()+bad);
                auto extra=std::make_shared<Group>();extra->cells={orphan};extra->neutral=true;groups.push_back(extra);
                score();Info x=info(group->cells[0]);if(!x.valid||x.neutral)return false;
                auto options=partners(x);if(options.empty())return false;
                int c=options[integer(rng,options.size())];take(c);group->cells.push_back(c);open(c);
                if(std::abs(deficit)>tolerance)return false;
                changed=true;break;
            }
            if(!changed)return true;
        }
        return false;
    }
    bool plan(int count) {
        const int limit=std::max(2*int(remaining.size()),8*count);
        for(int step=0;step<limit;++step) {
            int size=0;for(const auto &x:groups)size+=x->cells.size();
            if(size>=count&&std::abs(deficit)<=tolerance)break;
            auto candidates=remaining;bool success=false;
            while(!candidates.empty()) {
                int i=integer(rng,candidates.size()),c=candidates[i];candidates[i]=candidates.back();candidates.pop_back();
                if(!take(c))continue;
                auto group=std::make_shared<Group>();group->cells={c};groups.push_back(group);open(c);
                Info x=info(c);
                if(x.valid) {
                    group->neutral=x.neutral;score();
                    if(group->neutral){success=true;break;}
                    auto options=partners(x);
                    if(!options.empty()) {
                        int partner=options[integer(rng,options.size())];take(partner);group->cells.push_back(partner);open(partner);
                        if(repair()&&std::abs(deficit)<=tolerance){success=true;break;}
                    }
                }
                remove(group);
            }
            if(!success)break;
        }
        return !groups.empty()&&std::abs(deficit)<=tolerance;
    }
};
}

extern "C" int generals_map_spawns(const uint8_t *mountains,int w,int h,bitgen_t *rng,int32_t *out) {
    Graph g(w,h);std::vector<int> candidates;
    for(int c=0;c<g.n;++c)if(!mountains[c])candidates.push_back(c);
    if(candidates.empty())return 0;
    std::vector<double> scores(g.n,-1);int retries=0;
    for(int attempt=0;attempt<1000;++attempt) {
        int a=candidates[integer(rng,candidates.size())],b=candidates[integer(rng,candidates.size())];
        if(g.manhattan(a,b)<=14.4)continue;
        double required=1-std::pow(1-(.5-.005*retries),4)-.02;
        if(required>0) {
            for(int c:{a,b})if(scores[c]<0)scores[c]=spawn_score(c,mountains,g);
            if(std::min(scores[a],scores[b])/std::max(1.,std::max(scores[a],scores[b]))<required){++retries;continue;}
        }
        out[0]=a;out[1]=b;return 1;
    }
    return 0;
}

extern "C" int generals_map_cities(const uint8_t *mountains,const int32_t *generals,int w,int h,
                                    bitgen_t *rng,int count,int32_t *cells,int32_t *armies,int32_t *groups) {
    Graph graph(w,h);DuelCities city(mountains,generals,graph,rng);
    if(!city.plan(count))return -1;
    int k=0,index=0;
    for(const auto &group:city.groups) {
        int army=int(std::floor(40+10*uniform(rng)+.5));
        for(int c:group->cells){cells[k]=c;armies[k]=army;groups[k]=index;++k;}
        ++index;
    }
    return k;
}

extern "C" int generals_map_valid(const uint8_t *mountains,const int32_t *generals,int players,
                                   const int32_t *cities,int count,int w,int h) {
    Graph graph(w,h);
    if(count>=players) for(int p=0;p<players;++p) {
        bool near=false;for(int i=0;i<count;++i)if(graph.manhattan(generals[p],cities[i])<=6){near=true;break;}
        if(!near)return 0;
    }
    std::vector<uint8_t> blocked(graph.n,0);
    for(int p=0;p<players;++p)blocked[generals[p]]=1;
    for(int i=0;i<count;++i)blocked[cities[i]]=1;
    for(int p=1;p<players;++p)for(int q=0;q<2;++q) {
        std::vector<uint8_t> seen(graph.n,0);std::deque<int> queue{generals[p]};seen[generals[p]]=1;
        bool found=false;
        while(!queue.empty()) {
            int c=queue.front();queue.pop_front();if(c==generals[q]){found=true;break;}
            for(int x:graph.neighbors[c])if(!seen[x]&&!mountains[x]&&(x==generals[q]||!blocked[x])){seen[x]=1;queue.push_back(x);}
        }
        if(!found)return 0;
    }
    return 1;
}

extern "C" void generals_map_connect(uint8_t *terrain,int w,int h,int source) {
    Graph g(w,h,true);std::vector<uint8_t> connected(g.n,0),settled(g.n,0);
    std::vector<int> component{source};std::deque<int> queue{source};connected[source]=1;
    while(!queue.empty()) {
        int c=queue.front();queue.pop_front();
        for(int x:g.neighbors[c])if(!terrain[x]&&!connected[x]){connected[x]=1;component.push_back(x);queue.push_back(x);}
    }
    if(int(component.size())==std::count(terrain,terrain+g.n,0))return;
    std::vector<int> distance(g.n,g.n+1),parent(g.n,-1);
    for(int c:component){distance[c]=0;queue.push_back(c);}
    while(!queue.empty()) {
        int c=queue.front();queue.pop_front();if(settled[c])continue;settled[c]=1;
        for(int x:g.neighbors[c]) {
            int candidate=distance[c]+terrain[x];
            if(candidate<distance[x]) {
                distance[x]=candidate;parent[x]=c;
                if(terrain[x])queue.push_back(x);else queue.push_front(x);
            }
        }
    }
    // Traverse original plains in row order before modifying any terrain.
    std::vector<int> carved;
    for(int c=0;c<g.n;++c)if(!terrain[c]) {
        int x=c;while(!connected[x]){connected[x]=1;if(terrain[x])carved.push_back(x);x=parent[x];}
    }
    for(int c:carved)terrain[c]=0;
}

extern "C" int generals_map_distance(const uint8_t *terrain,const uint8_t *structure,int w,int h,int a,int b) {
    Graph g(w,h,true);std::vector<int> distance(g.n,-1);distance[a]=0;std::deque<int> queue{a};
    while(!queue.empty()&&distance[b]<0) {
        int c=queue.front();queue.pop_front();
        for(int x:g.neighbors[c])if(!terrain[x]&&structure[x]!=1&&distance[x]<0){distance[x]=distance[c]+1;queue.push_back(x);}
    }
    return distance[b];
}

extern "C" int generals_map_pair(const int64_t *candidates,int count,int width,int distance,
                                  bitgen_t *rng,int64_t *out) {
    // np.argwhere's row-major pair order, without quadratic temporary arrays.
    std::vector<std::pair<int,int>> pairs;
    std::vector<int> xs(count),ys(count);
    for(int i=0;i<count;++i){xs[i]=candidates[i]%width;ys[i]=candidates[i]/width;}
    for(int a=0;a<count;++a)for(int b=0;b<count;++b)
        if(std::abs(xs[a]-xs[b])+std::abs(ys[a]-ys[b])==distance)pairs.emplace_back(a,b);
    if(pairs.empty())return 0;
    auto [a,b]=pairs[integer(rng,pairs.size())];
    out[0]=candidates[a];out[1]=candidates[b];return 1;
}
