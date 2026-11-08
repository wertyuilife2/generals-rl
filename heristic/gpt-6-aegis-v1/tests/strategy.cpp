#include "../policy.cpp"
#include <cassert>
#include <memory>

struct Board {
    static constexpr int side = 7, count = side * side;
    std::array<std::uint8_t, count> terrain{}, structure{}, visible{}, fog{};
    std::array<std::int16_t, count> owner{};
    std::array<std::int64_t, count> soldiers{};
    std::array<std::uint8_t, 2> alive{1, 1}, kills{}, afk{};
    std::array<std::int64_t, 2> army{100, 100}, land{10, 10}, cities{};
    std::array<double, 2> departure{-1, -1};
    ArenaObservation view{side, side, 2, 0, 200, 0, terrain.data(), structure.data(),
        owner.data(), soldiers.data(), visible.data(), alive.data(), army.data(),
        land.data(), cities.data(), fog.data(), kills.data(), afk.data(), departure.data(), nullptr};
    Board() {
        owner.fill(-1); visible.fill(1);
        owner[0] = 0; soldiers[0] = 30; structure[0] = 2;
    }
    std::unique_ptr<aegis::Agent> agent(const char *params = "{}") {
        auto result = std::make_unique<aegis::Agent>(0, 1, params);
        result->observe(&view); return result;
    }
};
int original_tests() {
    { // Opening accumulation applies only while no opponent is visible;
      // an immediately capturable enemy capital still interrupts it.
        Board b;b.view.tick=10;b.soldiers[0]=6;
        auto a=b.agent("{\"opening_delay\":24}");assert(a->act(&b.view)==0);
        b.owner[1]=1;b.soldiers[1]=2;b.structure[1]=2;b.soldiers[0]=20;
        a=b.agent("{\"opening_delay\":24}");assert(a->act(&b.view)==arena_move(0,1,false));
    }
    { // An expired search budget preserves the already selected legal action,
      // instead of starting another route search or replacing it with wait.
        Board b;auto a=b.agent();a->trace_threats();
        a->choose({0,1,false},2,1,aegis::Offer::EXPLORE);
        a->search_until=std::chrono::steady_clock::time_point::min();
        a->resource_routes(0,false);
        assert(a->best.move.from==0 && a->best.move.to==1 && a->best.utility==2);
    }
    { // A vacated city remains recoverable by its strong adjacent column;
      // a trapped or undersized column must still pay the full exposure cost.
        Board b;b.owner[24]=0;b.structure[24]=1;b.soldiers[24]=100;
        b.owner[25]=1;b.soldiers[25]=6;
        auto old=b.agent("{\"release\":0}");auto a=b.agent("{\"release\":1}");
        assert(a->city_exposure({24,17,false})<old->city_exposure({24,17,false}));
        b.owner[16]=1;b.soldiers[16]=110;
        old=b.agent("{\"release\":0}");a=b.agent("{\"release\":1}");
        assert(a->city_exposure({24,17,false})==old->city_exposure({24,17,false}));
    }
    { // Short-lived land occupations do not earn several growth pulses when
      // a much stronger visible column can immediately retake the tile.
        Board b;b.owner[24]=1;b.soldiers[24]=1;b.owner[25]=1;b.soldiers[25]=80;
        auto a=b.agent("{\"occupation\":1}");
        double exposed=a->tile_reward(24,5,1,23);
        b.owner[25]=-1;b.soldiers[25]=0;a=b.agent("{\"occupation\":1}");
        assert(a->tile_reward(24,5,1,23)>exposed);
    }
    { // A once-funded return is not a binding commitment after an enemy
      // reinforcement makes it insufficient. A larger rescue may replace it.
        Board b;b.soldiers[0]=4;b.owner[7]=b.owner[14]=0;
        b.soldiers[7]=1;b.soldiers[14]=10;b.owner[3]=1;b.soldiers[3]=45;
        auto a=b.agent("{\"revalidate\":1}");a->trace_threats();
        a->campaigns[0]={{14,7,0},200,0,aegis::Offer::RESCUE};
        a->continue_campaigns();
        assert(a->campaigns[0].path.empty());
        b.soldiers[14]=60;a=b.agent("{\"revalidate\":1}");a->trace_threats();
        a->campaigns[0]={{14,7,0},200,0,aegis::Offer::RESCUE};
        a->continue_campaigns();
        assert(a->best.move.from==14 && a->best.move.to==7);
    }
    { // The subset optimization has an exhaustive independent oracle: choose
      // either one surviving corps or any pair with sufficient separation.
        Board b;
        for(int c=0;c<Board::count;++c) {b.owner[c]=0;b.soldiers[c]=2+(c*17)%53;}
        auto a=b.agent();unsigned all=(1u<<a->formations.size())-1;
        for(unsigned removed=0;removed<=all;++removed) {
            double expected=0;
            for(std::size_t i=0;i<a->formations.size();++i) if(!(removed&(1u<<i))) {
                expected=std::max(expected,a->formations[i].first);
                for(std::size_t j=i+1;j<a->formations.size();++j)
                    if(!(removed&(1u<<j)) && a->separated(a->formations[i].second,a->formations[j].second))
                        expected=std::max(expected,a->formations[i].first+.65*a->formations[j].first);
            }
            assert(a->remaining_formation[removed]==expected);
        }
    }
    { // Repeated offers share an exposure result only within the same view.
      // Changing the visible attacker invalidates every cached move score.
        Board b;b.soldiers[0]=10;b.owner[7]=1;b.soldiers[7]=15;
        auto a=b.agent();a->trace_threats();aegis::Move move{0,1,false};
        double initial=a->exposure(move);
        assert(initial==a->uncached_exposure(move) && initial==a->exposure(move));
        b.owner[7]=-1;b.soldiers[7]=0;++b.view.tick;
        a->observe(&b.view);a->trace_threats();
        assert(a->exposure(move)==a->uncached_exposure(move) && a->exposure(move)<initial);
    }
    { // Two small reserves can jointly save a capital within three commands,
      // although neither single-source return can fund the defense.
        Board b;b.terrain.fill(1);
        for(int c:{0,1,2,8,7,14,21})b.terrain[c]=0;
        b.soldiers[0]=5;
        b.owner[1]=b.owner[2]=b.owner[8]=0;b.soldiers[1]=1;
        b.soldiers[2]=b.soldiers[8]=6;b.owner[21]=1;b.soldiers[21]=17;
        auto a=b.agent("{\"fortify\":1}");a->trace_threats();a->defense_tree();
        assert(a->best.move.to==1 && (a->best.move.from==2 || a->best.move.from==8));
        assert(a->best.utility>2000);
    }
    { // A premium for blocking one invasion must not conceal a second lethal
      // approach which bypasses that interception point.
        Board b; b.soldiers[0]=5;
        b.owner[21]=b.owner[3]=1; b.soldiers[21]=30; b.soldiers[3]=40;
        auto a=b.agent("{\"screen\":1}");
        a->threats={{1,{21,14,7,0},30},{1,{3,2,1,0},40}};
        a->blocks[7].push_back({2,28,23});
        assert(a->interception_value(7,60,1)==0);
        a->threats.pop_back();
        assert(a->interception_value(7,60,1)>0);
    }
    { // The same unreplenished army cannot alternate two empty friendly tiles
      // under different rescue labels. A real immediate threat is exempt.
        Board b;b.owner[24]=b.owner[25]=0;b.soldiers[24]=1;b.soldiers[25]=40;b.view.tick=201;
        auto a=b.agent("{\"reversal\":1}");a->last_move={24,25,false};a->last_tick=200;a->last_army=40;
        ArenaMoveResult receipt{0,arena_move(24,1),1,24,25,39,0,-1,1};b.view.previous=&receipt;
        assert(a->reversal_cost({25,24,false})>1000);
        a->deadline=2;assert(a->reversal_cost({25,24,false})==0);
        a->deadline=10;receipt.valid=0;assert(a->reversal_cost({25,24,false})==0);
    }
    { // Actions precede growth. A column traversing its own land across tick
      // 50 receives the pulse, and collects the grown garrison ahead of it.
        Board b;b.view.tick=48;b.soldiers[0]=5;
        for(int c:{7,14,21,28})b.owner[c]=1;
        b.soldiers[28]=20;b.soldiers[21]=2;b.soldiers[14]=b.soldiers[7]=1;
        auto a=b.agent();aegis::Threat t{1,{28,21,14,7,0},20};
        assert(a->arrival(t,{})==22);
        assert(a->anticipated_growth(0,3)==2);
        b.view.tick=200;a->observe(&b.view);
        assert(a->anticipated_growth(0,1)==0); // no growth before the second action
        assert(a->anticipated_growth(0,2)==1);
    }
    { // Public opposing reserves justify some capital insurance without
      // requiring a visible invasion. Full evacuation costs more than half.
        Board b; b.army[1]=200; b.soldiers[0]=40;
        auto a=b.agent("{\"insurance\":2}");a->trace_threats();
        assert(a->exposure({0,1,false})>a->exposure({0,1,true}));
        b.soldiers[0]=5;b.owner[7]=0;b.soldiers[7]=40;
        a=b.agent("{\"insurance\":2}");a->trace_threats();a->insurance_offers();
        assert(a->best.move.from==7 && a->best.move.to==0);
    }
    { // A known capital behind a defended frontier still admits a funded
      // assault. The frontier's soldiers are charged once along the route.
        Board b; b.terrain.fill(1); for (int c : {0,1,2}) b.terrain[c]=0;
        b.soldiers[0]=100; b.owner[1]=b.owner[2]=1;
        b.soldiers[1]=20; b.soldiers[2]=10; b.structure[2]=2;
        auto a=b.agent("{\"siege\":1}"); a->trace_threats(); a->investment_offers();
        assert(a->best.objective==2 && a->best.move.from==0 && a->best.move.to==1);
        b.soldiers[0]=20; a=b.agent("{\"siege\":1}"); a->trace_threats(); a->investment_offers();
        assert(a->best.move.from<0);
    }
    { // A strong column three steps from a city limits the investment's
      // holding time even though it is not in the city's immediate neighbors.
        Board b; b.structure[24]=1; b.owner[24]=-2; b.soldiers[24]=40;
        b.owner[27]=1; b.soldiers[27]=90;
        auto a=b.agent("{\"tenure\":1}");
        assert(a->enemy_near[24]==0 && a->hostile_reach[24][3]>80);
        double risky=a->tile_reward(24,8,1,23);
        b.owner[27]=-1;b.soldiers[27]=0; a=b.agent("{\"tenure\":1}");
        assert(a->tile_reward(24,8,1,23)>risky*3);
    }
    { // Capturing a defended city pays its old garrison once. That garrison
      // is not also a surviving enemy able to retake the captured city.
        Board b; b.owner[24] = 1; b.structure[24] = 1; b.soldiers[24] = 50;
        auto a = b.agent(); assert(a->enemy_near[24] == 0);
        double value = a->tile_reward(24,8,1,23);
        b.owner[25] = 1; b.soldiers[25] = 60; a = b.agent();
        assert(a->enemy_near[24] == 60 && a->tile_reward(24,8,1,23) < value);
    }
    { // A recently seen column remains a bounded threat after entering fog.
      // The belief expires and never walks through currently visible cells.
        Board b; b.soldiers[0] = 5; b.owner[28] = 1; b.soldiers[28] = 50;
        auto a = b.agent("{\"ghost\":1}");
        b.visible[28] = 0; b.owner[28] = -3; b.soldiers[28] = -1; b.view.tick = 201;
        a->observe(&b.view); a->trace_threats();
        assert(a->enemies.empty() && a->home_pressure > 20);
        for (const auto &t : a->threats) assert(t.corridor.front() == 28);
        b.view.tick = 211; a->observe(&b.view); a->trace_threats();
        assert(a->home_pressure == 0 && a->threats.empty());
    }
    { // An urgent capital delivery is completed even when the returning army
      // temporarily blocks one approach. This prevents relief/interception
      // from alternating between the same two cells until the economy dies.
        Board b; b.soldiers[0] = 4; b.owner[7] = b.owner[14] = 0;
        b.soldiers[7] = 1; b.soldiers[14] = 60;
        auto a = b.agent("{\"return_plan\":1}");
        a->campaigns[0] = {{14,7,0},200,0,aegis::Offer::RESCUE};
        a->last_move = {14,7,false}; a->last_tick = 200; a->last_campaign = 0;
        ArenaMoveResult receipt{0,arena_move(14,0),1,14,7,59,0,-1,1};
        b.view.previous = &receipt; b.view.tick = 201;
        b.soldiers[14] = 1; b.soldiers[7] = 60;
        a->observe(&b.view); a->trace_threats(); a->continue_campaigns();
        assert(a->best.move.from == 7 && a->best.move.to == 0);
        b.owner[7] = 1; a->observe(&b.view); a->continue_campaigns();
        assert(a->campaigns[0].path.empty());
    }
    { // An unseen obstacle is not identified as a city or traversed.
        Board b; b.visible[20] = 0; b.fog[20] = 1;
        b.terrain[20] = b.structure[20] = 255; b.owner[20] = -3; b.soldiers[20] = -1;
        auto a = b.agent(); assert(a->obstructed(20)); assert(a->memory[20].building == 255);
        b.visible[20] = 1; b.terrain[20] = 0; b.structure[20] = 1;
        b.owner[20] = -2; b.soldiers[20] = 45; b.view.tick++;
        a->observe(&b.view); assert(!a->obstructed(20)); assert(a->defense(20) == 45);
    }
    { // The source must survive every earlier, visible adjacent attack.
        Board b; b.owner[24] = 0; b.soldiers[24] = 18;
        b.owner[31] = 1; b.soldiers[31] = 18;
        auto a = b.agent(); assert(!a->executable({24, 17, false}));
        b.soldiers[31] = 16; a->observe(&b.view); assert(a->executable({24, 17, false}));
    }
    { // A defended capital cannot be valued as an immediate guaranteed capture.
        Board b; b.owner[24] = 0; b.soldiers[24] = 70;
        b.owner[25] = b.owner[26] = 1; b.structure[25] = 2;
        b.soldiers[25] = 10; b.soldiers[26] = 80;
        auto a = b.agent(); assert(a->king_guard(25) == 89);
        b.soldiers[26] = 2; a = b.agent();
        auto action = a->act(&b.view); assert(action == arena_move(24, 1, false));
    }
    { // A neutral-city investment is discounted when a nearby column can take it.
        Board b; b.structure[24] = 1; b.owner[24] = -2; b.soldiers[24] = 40;
        auto a = b.agent(); double safe = a->tile_reward(24, 8, 2, 23);
        b.owner[25] = 1; b.soldiers[25] = 90; a = b.agent();
        assert(a->tile_reward(24, 8, 2, 23) < safe / 4);
    }
    { // Buying the first city requires combining separate, affordable branches.
        Board b;
        for (int c : {16, 17, 23, 24}) { b.owner[c] = 0; b.soldiers[c] = 16; }
        b.owner[25] = -2; b.structure[25] = 1; b.soldiers[25] = 40;
        auto a = b.agent(); a->trace_threats(); a->investment_offers();
        assert(a->best.purpose == aegis::Offer::INVEST);
        assert(a->best.objective == 25 && a->executable(a->best.move));
        assert(a->ours(a->best.move.to));
        for (int c : {16, 17, 23, 24}) b.soldiers[c] = 3;
        a = b.agent(); a->trace_threats(); a->investment_offers();
        assert(a->best.move.from < 0); // Shared paths cannot invent funding.
    }
    { // A visible counterattack makes abandoning a held city expensive.
        Board b; b.owner[24] = 0; b.soldiers[24] = 20; b.structure[24] = 1;
        b.owner[25] = 1; b.soldiers[25] = 15;
        auto a = b.agent();
        assert(a->city_exposure({}) == 0);
        assert(a->city_exposure({24, 17, false}) > 10);
        assert(a->city_exposure({24, 25, false}) == 0);
    }
    { // The cheap route to a junction cannot afford the city. A longer route
      // collects two distinct garrisons and must survive label pruning.
        Board b; b.terrain.fill(1);
        for (int c : {0,17,18,24,25,26}) b.terrain[c] = 0;
        for (int c : {17,18,24,25}) b.owner[c] = 0;
        b.soldiers[17] = b.soldiers[18] = 20;
        b.soldiers[24] = 5; b.soldiers[25] = 1;
        b.owner[26] = -2; b.structure[26] = 1; b.soldiers[26] = 40;
        // Isolate route feasibility from strategic rewards: this is a search
        // regression, not a requirement to buy every affordable city.
        auto a = b.agent("{\"muster\":0,\"probe\":0,\"pressure\":0,\"phases\":0}");
        a->trace_threats(); a->resource_routes(24,false);
        assert(a->best.objective == 26 && a->best.move.to == 17);
    }
    { // More income reduces peaceful investment's marginal value, but does
      // not remove the value of denying an enemy's income.
        Board b; b.owner[24] = -2; b.structure[24] = 1;
        auto early = b.agent("{\"phases\":1}"); double initial = early->tile_reward(24, 40, 1, 23);
        b.cities[0] = 8; auto late = b.agent("{\"phases\":1}");
        double neutral = late->tile_reward(24, 40, 1, 23);
        assert(neutral < initial * .6);
        b.owner[24] = 1; late = b.agent("{\"phases\":1}");
        assert(late->tile_reward(24, 40, 1, 23) > neutral * 2);
    }
    { // A second independent formation is valuable; adjacent fragments of
      // one army must not be rewarded as two fronts.
        Board b; b.soldiers[0] = 1;
        for (int c : {21, 27}) { b.owner[c] = 0; b.soldiers[c] = 80; }
        auto a = b.agent(); double separated = a->remaining_formation[0];
        b.owner[27] = -1; b.soldiers[27] = 0; b.owner[22] = 0; b.soldiers[22] = 80;
        a = b.agent(); assert(a->remaining_formation[0] < separated);
    }
    { // Discovering a general does not erase the benefit of scouting a new
      // region. Revisited stale fog also carries positive information value.
        Board b; b.owner[48] = 1; b.structure[48] = 2;
        for (int c : {25, 26, 32, 33}) { b.visible[c] = 0; b.owner[c] = -3; b.soldiers[c] = -1; }
        auto a = b.agent("{\"muster\":0,\"pressure\":0}");
        assert(!a->objectives.empty() && a->mission_value(25, 80, 0) > 0);
    }
    { // A campaign advances only on a matching successful engine receipt;
      // an invalid action or blocked route must discard the old plan.
        Board b; b.owner[7] = 0; b.soldiers[7] = 30;
        auto a = b.agent();
        a->campaigns[0] = {{7,14,21}, 200, 0, aegis::Offer::EXPLORE};
        a->last_move = {7,14,false}; a->last_tick = 200; a->last_campaign = 0;
        ArenaMoveResult receipt{0,arena_move(7,2),1,7,14,29,1,-1,1};
        b.view.previous = &receipt; b.view.tick = 201; b.owner[14] = 0; b.soldiers[14] = 29;
        a->observe(&b.view); assert(a->campaigns[0].path.front() == 14);
        b.terrain[21] = 1; a->observe(&b.view); a->continue_campaigns();
        assert(a->campaigns[0].path.empty());
        a->campaigns[0].path = {14,15}; a->last_move = {14,15,false};
        a->last_tick = 201; a->last_campaign = 0; b.view.tick = 202; receipt.valid = 0;
        a->observe(&b.view); assert(a->campaigns[0].path.empty());
    }
    { // Pulling guards down the same invasion corridor does not add protection.
      // An off-corridor reserve does, and must start before its return window closes.
        Board b; b.terrain.fill(1);
        for (int c : {0,1,2,7,14,21}) b.terrain[c] = 0;
        for (int c : {1,2,7,14}) b.owner[c] = 0;
        b.soldiers[0] = 5; b.soldiers[1] = b.soldiers[14] = 1;
        b.soldiers[7] = 40; b.soldiers[2] = 14; b.owner[21] = 1; b.soldiers[21] = 60;
        auto a = b.agent(); a->trace_threats(); a->relief_offers();
        assert(a->best.move.from == 2 && a->best.move.to == 1);
        assert(a->best.utility > 100);
    }
    { // Production may wait for a useful wave; joining an already formed
      // adjacent army does not pay a batching penalty.
        Board b; b.land[0] = 30; b.soldiers[0] = 3;
        auto a = b.agent(); assert(a->launch_cost({0,1,false}) > 1);
        b.owner[1] = 0; b.soldiers[1] = 20; a = b.agent();
        assert(a->launch_cost({0,1,false}) == 0);
    }
    { // An assembled corps at the capital is ready to depart, not available
      // for an endless sequence of new rallies in nearby cells.
        Board b;
        for (int c = 0; c < 40; ++c) { b.owner[c] = 0; b.soldiers[c] = 1; }
        b.soldiers[0] = 30; b.army[0] = 69; b.land[0] = 40;
        auto a = b.agent(); a->trace_threats(); a->rally_offers();
        assert(a->best.move.from < 0);
    }
    { // Even a three-soldier enemy can take a capital after its garrison leaves.
      // A distant larger stack must not crowd this immediate threat out.
        Board b; b.soldiers[0] = 13; b.owner[7] = 1; b.soldiers[7] = 3;
        auto a = b.agent(); a->trace_threats();
        assert(a->exposure({0,1,false}) >= 1e6);
        assert(a->exposure({0,1,true}) < 1e6);
        auto action = a->act(&b.view); assert(action != arena_move(0,1,false));
    }
    { // A field army can intercept a lethal approach without taking an owned
      // supply route all the way back to the capital.
        Board b; b.terrain.fill(1);
        for (int c : {0,7,14,21,28,15,16}) b.terrain[c] = 0;
        b.soldiers[0] = 5; b.owner[16] = 0; b.soldiers[16] = 60;
        b.owner[28] = 1; b.soldiers[28] = 40;
        auto a = b.agent(); a->trace_threats(); a->resource_routes(16,false);
        assert(a->best.purpose == aegis::Offer::RESCUE && a->best.objective == 14);
        assert(a->best.move.to == 15);
    }
    { // Capital hypotheses use only unexplored, unobstructed cells. A small
      // scout has information value, and a discovered general resolves the prior.
        Board b;
        for (int c : {40,41,42}) {
            b.visible[c] = 0; b.terrain[c] = b.structure[c] = 255;
            b.owner[c] = -3; b.soldiers[c] = -1;
        }
        b.fog[42] = 1;
        auto a = b.agent("{\"probe\":0}");
        assert(a->capital_belief[42] == 0 && a->capital_belief[39] == 0);
        assert(std::abs(a->capital_belief[40] + a->capital_belief[41] - 1) < 1e-9);
        assert(a->mission_value(41,1,0) > 0);
        b.owner[48] = 1; b.structure[48] = 2; a->observe(&b.view);
        for (double p : a->capital_belief) assert(p == 0);
    }
    { // A landing can be locally legal yet lose the whole column to a visible
      // reinforcement or a pursuer that becomes adjacent next turn.
        Board b; b.owner[24] = 0; b.soldiers[24] = 20;
        b.owner[26] = 1; b.soldiers[26] = 30;
        auto a = b.agent("{\"counter\":1}"); assert(a->executable({24,25,false}));
        assert(a->counter_cost({24,25,false}) > 20);
        b.owner[26] = -1; b.soldiers[26] = 0; b.owner[27] = 1; b.soldiers[27] = 30;
        a = b.agent("{\"counter\":1}"); assert(a->counter_cost({24,25,false}) > 10);
        b.terrain[26] = 1; a = b.agent("{\"counter\":1}"); assert(a->counter_cost({24,25,false}) == 0);
    }
    { // A formed field column stops an unfunded raid before exhausting itself,
      // but can still merge into friendly reinforcements and resume operations.
        Board b; b.owner[24] = 0; b.soldiers[24] = 12;
        auto a = b.agent("{\"preserve\":1}"); a->trace_threats(); a->resource_routes(24,false);
        assert(a->best.move.from < 0);
        b.owner[25] = 0; b.soldiers[25] = 8;
        a = b.agent("{\"preserve\":1}"); a->trace_threats(); a->resource_routes(24,false);
        assert(a->best.move.to == 25);
    }
    return 0;
}

int existing_tests() {
    original_tests();
    { // Identical geometry with a stronger capital demands a stronger army.
        Board b;b.owner[48]=1;b.structure[48]=2;b.soldiers[48]=5;
        auto light=b.agent("{\"menace\":1}");double a=light->menace_value(40,30);
        b.soldiers[48]=80;auto heavy=b.agent("{\"menace\":1}");
        assert(a>heavy->menace_value(40,30));
        assert(heavy->menace_value(40,100)>heavy->menace_value(40,30));
    }
    { // Extra force past mission sufficiency is not extra pressure value.
        Board b;b.owner[48]=1;b.structure[48]=2;b.soldiers[48]=5;
        auto a=b.agent("{\"menace\":1}");
        assert(a->menace_value(40,100)==a->menace_value(40,1000));
        assert(a->menace_value(47,100)>a->menace_value(40,100));
    }
    { // A costly defender makes an available detour preferable.
        Board b;b.owner[48]=1;b.structure[48]=2;b.soldiers[48]=5;
        b.owner[47]=1;b.soldiers[47]=100;
        auto a=b.agent("{\"menace\":1}");assert(a->menace_need[0][46]<20);
        b.terrain.fill(1);for(int c:{0,46,47,48})b.terrain[c]=0;
        a=b.agent("{\"menace\":1}");assert(a->menace_need[0][46]>100);
        assert(a->menace_value(0,100)==0);
    }
    { // Unknown capitals introduce no fictitious pressure target.
        Board b;auto a=b.agent("{\"menace\":1}");
        assert(a->menace_need.empty() && a->menace_value(7,30)==0);
    }
    return 0;
}

int transit_tests() {
    existing_tests();
    Board b;b.terrain.fill(1);b.owner[0]=-1;b.structure[0]=0;b.soldiers[0]=0;
    for(int c:{1,2,8,15}) b.terrain[c]=0;
    b.view.general_pos=1;b.structure[1]=2;b.owner[1]=0;b.soldiers[1]=20;
    b.owner[2]=0;b.soldiers[2]=6;b.owner[8]=0;b.soldiers[8]=1;
    b.owner[15]=1;b.soldiers[15]=10;b.structure[15]=2;
    auto old=b.agent("{\"home_transit\":0}");old->trace_threats();old->resource_routes(2,false);
    assert(old->best.move.from<0);
    auto a=b.agent("{\"home_transit\":1}");a->trace_threats();a->resource_routes(2,false);
    assert(a->best.move.from==2 && a->best.move.to==1 && a->best.objective==15);
    assert(a->best.path==std::vector<int>({2,1,8,15}));
    return 0;
}

#include <iostream>
int main() {
    transit_tests();
    std::uint64_t checks=0, cached=0;
    for(int sample=0;sample<200;++sample) {
        Board b;unsigned rng=31+sample*719;
        auto next=[&](){rng=rng*1664525u+1013904223u;return rng;};
        b.view.tick=sample*13;
        for(int c=1;c<Board::count;++c) {
            b.owner[c]=int(next()%3)-1;b.soldiers[c]=next()%90;
            b.structure[c]=(next()%13==0)?1:0;
            if(b.owner[c]<0 && !b.structure[c])b.soldiers[c]=0;
        }
        b.owner[48]=1;b.structure[48]=2;b.soldiers[48]=20;
        b.soldiers[0]=2+sample%40;
        auto a=b.agent("{\"home_transit\":1}");a->trace_threats();
        for(const auto &t:a->threats) {
            assert(a->arrival(t,{})==a->uncached_arrival(t,{}));++checks;
            for(int c:a->owned)for(int v:a->edge[c])if(v>=0)for(bool split:{false,true}) {
                aegis::Move m{c,v,split};
                assert(a->arrival(t,m)==a->uncached_arrival(t,m));++checks;
                cached+=!(a->threat_touch[c]&t.touch_bit)&&!(a->threat_touch[v]&t.touch_bit);
            }
        }
        // A new observation invalidates the cache even at the same tick.
        a->observe(&b.view);assert(!a->arrival_cache_live);
    }
    assert(cached>10000);
    std::cout<<"arrival comparisons "<<checks<<" cache hits "<<cached<<"\n";
}
