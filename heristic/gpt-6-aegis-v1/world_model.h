// Public-observation estimates. These are heuristic distributions, not facts
// about unseen cells and not calibrated probabilities of winning.
struct StrategicMap {
    int size = 0, players = 0, refreshed = -1000;
    std::vector<std::array<double,8>> capital, control;
    std::vector<std::array<int,8>> first_contact;
    std::vector<double> city, city_information, capital_information, enemy_region;
    std::array<double,8> unallocated_land{};
    std::array<int,8> known_general{};

    template<class Memories, class Edges>
    void update(const ArenaObservation &o, const Memories &mem, const Edges &edges) {
        const int n=o.width*o.height;
        bool changed = n!=size || players!=o.players;
        if(changed) {
            size=n;players=o.players;refreshed=-1000;
            capital.assign(n,{});control.assign(n,{});first_contact.resize(n);
            for(auto &t:first_contact)t.fill(-1);
            city.resize(n);city_information.resize(n);capital_information.resize(n);enemy_region.resize(n);
            known_general.fill(-1);
        }
        std::array<int,8> general;general.fill(-1);
        for(int c=0;c<n;++c) {
            int p=mem[c].owner;
            if(o.visible[c] && p>=0 && o.alive[p] && first_contact[c][p]<0) {
                first_contact[c][p]=o.tick;
                if(p!=o.player) changed=true;
            }
            if(p>=0 && o.alive[p] && mem[c].building==2)general[p]=c;
        }
        changed |= general!=known_general;
        // New enemies/general sightings refresh immediately; quiet fog estimates
        // are amortized. Visible facts below are reconciled every decision.
        bool rebuild=changed || o.tick-refreshed>=8;
        known_general=general;
        if(rebuild) {
            refreshed=o.tick;
            std::vector<int> distance(n),work;
            std::vector<std::array<double,8>> affinity(n);
            int known_cities=0,known_obstacles=0,hidden_obstacles=0;
            for(int c=0;c<n;++c) {
                known_cities+=mem[c].building==1;
                known_obstacles+=mem[c].terrain==1;
                hidden_obstacles+=mem[c].observed<0 && o.fog_obstacles[c];
            }
            // Default-map count is a weak prior, updated by revealed obstacles.
            // Custom maps need not follow it. An obstacle is never a certain city.
            double prior=(5.+2.5*players)/std::max(1.,.24*n);
            double city_rate=std::clamp((8*prior+known_cities)/(8.+known_cities+known_obstacles),.01,.65);
            double remaining=std::max(1.,5.+2.5*players-known_cities);
            if(hidden_obstacles)city_rate=std::min(city_rate,std::max(.01,remaining/hidden_obstacles));
            for(int c=0;c<n;++c) {
                capital[c].fill(0);control[c].fill(0);
                city[c]=mem[c].building==1?1.:mem[c].observed<0 && o.fog_obstacles[c]?city_rate:0.;
            }
            auto distance_from=[&](const std::vector<std::pair<int,int>>& starts) {
                std::fill(distance.begin(),distance.end(),1000000);
                using Node=std::pair<int,int>;
                std::priority_queue<Node,std::vector<Node>,std::greater<Node>> open;
                for(auto [c,d]:starts)if(d<distance[c]){distance[c]=d;open.emplace(d,c);}
                while(!open.empty()) {
                    auto [d,u]=open.top();open.pop();if(distance[u]!=d)continue;
                    for(int v:edges[u])if(v>=0 && mem[v].terrain!=1) {
                        // Hidden obstacle may be a costly city. This graph is
                        // for estimates; the real move planner still blocks it.
                        int next=d+(mem[v].terrain==2?4:1);
                        if(next<distance[v]){distance[v]=next;open.emplace(next,v);}
                    }
                }
            };
            for(int p=0;p<players;++p)if(o.alive[p]) {
                std::vector<std::pair<int,int>> evidence;
                for(int c=0;c<n;++c)if(mem[c].owner==p && o.tick-mem[c].observed<=150)
                    evidence.emplace_back(c,std::min(12,int((o.tick-mem[c].observed)/12)));
                distance_from(evidence);
                for(int c=0;c<n;++c)affinity[c][p]=distance[c]<1000000?std::exp(-.22*distance[c]):.002;
                if(general[p]>=0){capital[general[p]][p]=1;continue;}
                if(p==o.player)continue;
                // Several spatially separated early contacts describe a core,
                // rather than treating the nearest scout as the enemy capital.
                std::vector<std::pair<int,int>> contacts;
                for(int c=0;c<n;++c)if(first_contact[c][p]>=0)contacts.emplace_back(first_contact[c][p],c);
                std::sort(contacts.begin(),contacts.end());
                std::vector<int> landmarks;
                for(auto [t,c]:contacts) {
                    bool distinct=true;
                    for(int u:landmarks)if(std::abs(u%o.width-c%o.width)+std::abs(u/o.width-c/o.width)<4)distinct=false;
                    if(distinct)landmarks.push_back(c);
                    if(landmarks.size()==3)break;
                }
                for(int c=0;c<n;++c)if(mem[c].observed<0 && mem[c].terrain!=1 && !o.fog_obstacles[c]) {
                    int d=std::abs(c%o.width-o.general_pos%o.width)+std::abs(c/o.width-o.general_pos/o.width);
                    double spacing=std::max(9.,9*(1.8-.1*players));
                    capital[c][p]=d>spacing?1.:.03;
                }
                for(int c:landmarks) {
                    distance_from({{c,0}});
                    for(int v=0;v<n;++v)if(capital[v][p]>0) {
                        double d=distance[v];
                        double reachable=d<=first_contact[c][p]+1?1.:.03;
                        capital[v][p]*=reachable*(.05+1./(1+d*d/144.));
                    }
                }
                double total=0;for(int c=0;c<n;++c)total+=capital[c][p];
                if(total>0)for(int c=0;c<n;++c)capital[c][p]/=total;
            }
            std::array<double,8> remaining_land{};
            for(int p=0;p<players;++p)if(o.alive[p] && p!=o.player)remaining_land[p]=o.total_land[p];
            std::vector<double> capacity(n);
            for(int c=0;c<n;++c) {
                if(o.visible[c]) {
                    int p=o.owner[c];if(p>=0 && o.alive[p]) {
                        control[c][p]=1;remaining_land[p]=std::max(0.,remaining_land[p]-1);
                    }
                } else if(mem[c].terrain!=1)capacity[c]=o.fog_obstacles[c]?city[c]:1.;
                for(int p=0;p<players;++p)affinity[c][p]+=.25*n*capital[c][p];
            }
            // Allocate only public unobserved territory counts, sharing each
            // cell's capacity across opponents. Never assign it to two players.
            for(int pass=0;pass<12;++pass) {
                std::array<double,8> mass{};
                for(int c=0;c<n;++c)if(capacity[c]>1e-9)
                    for(int p=0;p<players;++p)mass[p]+=capacity[c]*affinity[c][p];
                std::array<double,8> allocated{};
                for(int c=0;c<n;++c)if(capacity[c]>1e-9) {
                    std::array<double,8> proposal{};double sum=0;
                    for(int p=0;p<players;++p)if(mass[p]>0) {
                        proposal[p]=remaining_land[p]*capacity[c]*affinity[c][p]/mass[p];sum+=proposal[p];
                    }
                    double scale=sum>capacity[c]?capacity[c]/sum:1.;
                    for(int p=0;p<players;++p) {
                        double add=proposal[p]*scale;control[c][p]+=add;allocated[p]+=add;capacity[c]-=add;
                    }
                }
                for(int p=0;p<players;++p)remaining_land[p]=std::max(0.,remaining_land[p]-allocated[p]);
            }
            unallocated_land=remaining_land;
        }
        // A new visible fact always overrides the cached estimate immediately.
        for(int c=0;c<n;++c) {
            if(o.visible[c]) {
                control[c].fill(0);int p=o.owner[c];if(p>=0 && o.alive[p])control[c][p]=1;
                city[c]=o.structure[c]==1?1.:0.;
                for(int p2=0;p2<players;++p2)if(c!=general[p2])capital[c][p2]=0;
            }
            for(int p=0;p<players;++p)if(!o.alive[p])capital[c][p]=control[c][p]=0;
        }
        for(int p=0;p<players;++p) {
            double sum=0;for(int c=0;c<n;++c)sum+=capital[c][p];
            if(sum>0)for(int c=0;c<n;++c)capital[c][p]/=sum;
        }
        for(int c=0;c<n;++c) {
            double ci=0,ki=0,region=0;int count=0;
            for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx) {
                int x=c%o.width+dx,y=c/o.width+dy;
                if(x<0 || x>=o.width || y<0 || y>=o.height)continue;
                int v=y*o.width+x;++count;
                for(int p=0;p<players;++p)if(p!=o.player)region+=control[v][p];
                if(std::abs(dx)<=1 && std::abs(dy)<=1 && !o.visible[v]) {
                    ci+=mem[v].observed<0?city[v]:0.;for(int p=0;p<players;++p)if(p!=o.player && general[p]<0)ki+=capital[v][p];
                }
            }
            city_information[c]=ci;capital_information[c]=ki;enemy_region[c]=count?region/count:0;
        }
    }
};
