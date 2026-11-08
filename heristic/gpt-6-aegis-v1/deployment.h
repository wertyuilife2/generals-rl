// Replace the positional prior only. Assembly and attack operators still price
// their own troops, commands, losses and first-move safety. No fixed route is held.
void prepare_deployment() {
    double scale=.35*(width+height);
    for(int c=0;c<cells;++c)
        formation_forward[c]=.25+.75*std::min(1.,home_distance[c]/scale);
    deployment_active=false;
    if(!deployment_mode)return;
    bool contact=false;
    for(int c=0;c<cells;++c) {
        int p=memory[c].owner;
        contact |= p>=0 && p!=seat && view->alive[p];
    }
    // Until a real contact exists, preserve the established exploratory prior.
    // The world-model's historical sightings are still legal observation memory.
    if(!contact || search_expired())return;
    if(deployment_mode==2)deployment_world.update(*view,memory,edge);
    if(search_expired())return;
    std::vector<double> salience(cells,0);
    double maximum=0;
    for(int c=0;c<cells;++c) {
        if(obstructed(c))continue;
        double territory=0,capital=0,city=0;
        if(deployment_mode==1) {
            const auto &m=memory[c];int p=m.owner;
            if(p>=0 && p!=seat && view->alive[p]) {
                territory=std::exp(-std::max(0,now-m.observed)/150.);
                city=m.building==1?territory:0.;
                capital=m.building==2?1.:0.;
            }
        } else {
            for(int p=0;p<view->players;++p)if(p!=seat && view->alive[p]) {
                territory+=deployment_world.control[c][p];
                capital+=deployment_world.capital[c][p];
            }
            city=territory*deployment_world.city[c];
        }
        // City salience uses25land units as a rounded economic prior. Capital
        // salience is strategic rather than an exact production count.
        salience[c]=territory+24*city+50*capital;
        maximum=std::max(maximum,salience[c]);
    }
    if(maximum<=0)return;
    std::vector<double> cost(cells,1e30);
    using Node=std::pair<double,int>;
    std::priority_queue<Node,std::vector<Node>,std::greater<Node>> open;
    for(int c=0;c<cells;++c)if(salience[c]>0) {
        cost[c]=-scale*std::log(salience[c]/maximum);open.emplace(cost[c],c);
    }
    int expanded=0;
    while(!open.empty()) {
        if((expanded++&63)==0 && search_expired())return;
        auto [d,c]=open.top();open.pop();if(d!=cost[c])continue;
        for(int v:edge[c])if(!obstructed(v) && d+1<cost[v]) {
            cost[v]=d+1;open.emplace(cost[v],v);
        }
    }
    // Max over plausible destinations of normalized salience*exp(-distance/L).
    // This is access potential, not the probability of owning or capturing them.
    for(int c=0;c<cells;++c)formation_forward[c]=.25+.75*std::exp(-cost[c]/scale);
    deployment_cost=std::move(cost);deployment_active=true;
}
