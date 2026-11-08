#ifndef AEGIS_SOURCE
#define AEGIS_SOURCE "../policy.cpp"
#endif
#include AEGIS_SOURCE
#include <cassert>
#include <iostream>
#include <memory>

struct FogBoard {
    static constexpr int w=13,n=w*w,home=6*w+2,contact=6*w+10;
    std::array<std::uint8_t,n> terrain{},structure{},visible{},fog{};
    std::array<std::int16_t,n> owner{};
    std::array<std::int64_t,n> army{};
    std::array<std::uint8_t,2> alive{1,1},flags{};
    std::array<std::int64_t,2> total{30,50},land{1,20},cities{};
    std::array<double,2> departure{-1,-1};
    ArenaObservation view{w,w,2,0,100,home,terrain.data(),structure.data(),owner.data(),army.data(),visible.data(),alive.data(),total.data(),land.data(),cities.data(),fog.data(),flags.data(),flags.data(),departure.data(),nullptr};
    FogBoard() {
        terrain.fill(255);structure.fill(255);owner.fill(-3);army.fill(-1);
        for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)reveal(home+dy*w+dx,0,0,-1,0);
        reveal(home,0,2,0,30);reveal(contact,0,0,1,10);
        for(int y=0;y<w;++y)fog[y*w+6]=1;
    }
    void reveal(int c,int t,int s,int p,int a) {
        visible[c]=1;fog[c]=0;terrain[c]=t;structure[c]=s;owner[c]=p;army[c]=a;
    }
    std::unique_ptr<aegis::Agent> agent() {
        auto a=std::make_unique<aegis::Agent>(0,1,"{\"deployment_mode\":2}");
        a->observe(&view);return a;
    }
};

int main() {
    int checks=0;
    FogBoard board;auto a=board.agent();
    const int city=6*board.w+6,mountain=4*board.w+6;
    // The public hint closes a road but does not identify the obstacle type.
    assert(a->obstructed(city));assert(a->memory[city].building==255);
    assert(a->home_distance[board.contact]>=aegis::far_away);
    assert(a->capital_belief[city]==0 && a->deployment_world.capital[city][1]==0);
    assert(a->deployment_world.city[city]>0 && a->deployment_world.city[city]<1);++checks;
    // Counterfactual private fields under the same visibility and hint masks
    // must not change navigation, city inference, capital or territory beliefs.
    FogBoard other;
    for(int c=0;c<other.n;++c)if(!other.visible[c]) {
        other.terrain[c]=c%2;other.structure[c]=c%3;other.owner[c]=1;other.army[c]=900+c;
    }
    auto b=other.agent();
    assert(a->home_distance==b->home_distance && a->capital_belief==b->capital_belief);
    assert(a->deployment_world.city==b->deployment_world.city);
    assert(a->deployment_world.capital==b->deployment_world.capital);
    assert(a->deployment_world.control==b->deployment_world.control);
    assert(a->formation_forward==b->formation_forward);++checks;
    // Revealing a city opens the graph; revealing a mountain keeps it closed.
    board.reveal(city,0,1,-2,40);board.reveal(mountain,1,0,-1,0);++board.view.tick;
    a->observe(&board.view);
    assert(!a->obstructed(city) && a->memory[city].building==1);
    assert(a->home_distance[board.contact]<aegis::far_away);
    assert(a->deployment_world.city[city]==1);
    assert(a->obstructed(mountain) && a->deployment_world.city[mountain]==0);++checks;
    // Once seen, an obstacle's remembered type survives later loss of vision.
    for(int c:{city,mountain}) {
        board.visible[c]=0;board.fog[c]=1;board.terrain[c]=board.structure[c]=255;
        board.owner[c]=-3;board.army[c]=-1;
    }
    ++board.view.tick;a->observe(&board.view);
    assert(!a->obstructed(city) && a->memory[city].building==1);
    assert(a->obstructed(mountain) && a->memory[mountain].terrain==1);++checks;
    std::cout<<checks<<" public fog knowledge scenarios passed\n";
}
