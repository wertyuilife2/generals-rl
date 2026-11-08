// Legal-memory strategic potential, not a guaranteed tactical capture proof.
// Cost includes leave-behind soldiers, stationary defenders and half a soldier
// per travel command for capital production. Reactions are re-observed each tick.
void prepare_menace() {
    menace_need.clear();menace_steps.clear();
    for(int king:objectives) {
        if(search_expired())break;
        std::vector<double> need(cells,1e30);
        std::vector<int> steps(cells,far_away);
        using Node=std::pair<double,int>;
        std::priority_queue<Node,std::vector<Node>,std::greater<Node>> open;
        need[king]=double(defense(king))+1;steps[king]=0;open.emplace(need[king],king);
        int expanded=0;
        while(!open.empty()) {
            if((expanded++ & 63)==0 && search_expired())break;
            auto [cost,c]=open.top();open.pop();if(cost!=need[c])continue;
            double toll=1.5+(c==king || ours(c)?0:double(defense(c)));
            for(int v:edge[c])if(!obstructed(v) && cost+toll<need[v]) {
                need[v]=cost+toll;steps[v]=steps[c]+1;open.emplace(need[v],v);
            }
        }
        menace_need.push_back(std::move(need));menace_steps.push_back(std::move(steps));
    }
}
double menace_value(int cell,Army force) const {
    if(force<=1)return 0;
    double best_value=0;
    for(std::size_t k=0;k<menace_need.size();++k) {
        int d=menace_steps[k][cell];double need=menace_need[k][cell];
        if(d<1 || d>40 || need>=1e29)continue;
        double capability=std::clamp((force-1.)/(need+3.),0.,1.);
        double value=menace*pressure*128*capability*capability/(1+.2*d);
        best_value=std::max(best_value,value);
    }
    return best_value;
}
