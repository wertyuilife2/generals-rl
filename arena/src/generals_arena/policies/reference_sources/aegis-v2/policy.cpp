// Aegis: an independent, single-thread command auction over resource-priced
// routes. This implementation was written afresh; it includes no other policy.
#include "arena_policy.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace aegis {
constexpr int far_away = 1000000;
constexpr int maximum_cells = 4096;
using Army = std::int64_t;

struct Move {
    int from = -1, to = -1;
    bool split = false;
};
struct Memory {
    int terrain = 255, building = 255, owner = -3, observed = -100000;
    Army soldiers = 0;
};
struct Offer {
    Move move;
    double utility = -1e30;
    int objective = -1;
    enum Purpose { EXPLORE, RAID, INVEST, SUPPLY, RESCUE, FINISH } purpose = EXPLORE;
    int assembly = -1;
    std::vector<int> path;
    int rally = -1;
};
struct Threat {
    int player;
    std::vector<int> corridor;
    Army initial = 0;
    Army nominal_arrival = 0;
    unsigned touch_bit = 0;
};
struct Campaign {
    std::vector<int> path;
    int started = 0, kings = 0;
    Offer::Purpose purpose = Offer::EXPLORE;
};
struct Interception {
    int arrival;
    Army force, shortage;
};
double option(const char *json, const char *key, double fallback) {
    if (!json) return fallback;
    const char *found = std::strstr(json, key);
    if (!found || !(found = std::strchr(found, ':'))) return fallback;
    char *end = nullptr;
    double x = std::strtod(found + 1, &end);
    return end != found + 1 && std::isfinite(x) ? x : fallback;
}

#include "world_model.h"

class Agent {
public:
    struct RouteLabel {
        int cell, parent, first, steps;
        Army force;
        double price, reward, losses;
        unsigned consumed;
        std::uint64_t visited;
    };
    using RouteNode = std::pair<double, int>;
    std::vector<RouteLabel> route_labels;
    std::vector<RouteNode> route_open;
    std::vector<std::array<int, 3>> route_chosen;
    int seat, width = 0, height = 0, cells = 0, home = -1, now = 0;
    int column = -1, intent = -1, last_tick = -2;
    bool goal_rally = true;
    int deployment_mode = 0;
    StrategicMap deployment_world;
    std::vector<double> deployment_cost;
    bool deployment_active = false;
    bool delivered_income = true;
    std::vector<int> delivery_distance;
    bool reroute_home = true;
    mutable std::array<std::vector<Threat>,4> departure_routes;
    mutable std::array<bool,4> departure_ready{};
    bool home_transit = true;
    bool liquidity_horizon = true, capital_disclosed = false;
    unsigned home_revealed_to = 0;
    double economic_horizon = 120;
    bool online_progress = true;
    int assembly_goal = -1, assembly_root = -1;
    Move last_move;
    const ArenaObservation *view = nullptr;
    std::vector<Memory> memory;
    std::vector<std::array<int, 4>> edge;
    std::vector<int> home_distance, contact_distance, owned_home_distance, king_distance, unknown;
    std::vector<double> information;
    std::vector<int> cell_x, cell_y;
    std::vector<double> formation_forward;
    mutable std::vector<double> frontier_cache;
    mutable std::vector<unsigned> compatible_cache;
    mutable int retained_cache_cell = -1;
    mutable Army retained_cache_force = 0;
    mutable double retained_cache_value = 0;
    std::vector<double> capital_belief, capital_information;
    std::vector<int> known_enemy_distance;
    std::vector<int> owned, mobile, enemies, objectives;
    std::vector<Threat> threats;
    std::vector<unsigned> threat_touch;
    bool arrival_cache_live = false;
    std::vector<std::vector<Interception>> blocks;
    std::vector<Army> enemy_near;
    std::vector<int> queue;
    double production = 1, horizon = 120, recon = .4, mobilize = 1, caution = 1, investment = 1;
    double troop_price = .15;
    double hold = 1;
    int route_budget = 1024;
    int half_budget = 256;
    double muster = 1, probe = 1, pressure = 1, phases = 1, desired_force = 30;
    double economy_weight = 1, readiness = 0;
    int continuity = 8;
    int opening_delay = 24;
    std::vector<std::pair<double, int>> formations;
    std::vector<unsigned> formation_bits;
    std::array<double, 512> remaining_formation{};
    std::array<Campaign, 2> campaigns;
    int last_campaign = -1;
    double menace = 1;
    std::vector<std::vector<double>> menace_need;
    std::vector<std::vector<int>> menace_steps;
    bool forward_rally = true;
    double rally = 1;
    double waves = 1;
    double development = 1;
    double intercept = 1;
    double hunt = .5;
    double counter = 0, preserve = 0, combat = 1, enemy_price = .25;
    double local = 0, setup = .8;
    double return_plan = 0;
    double ghost = 0;
    double coordinated = 0;
    double tenure = 0;
    double skirmish = 0;
    double siege = 0;
    double frontier = 0, pulse = 0;
    double insurance = 0;
    double lead = 0, command_setup = .8;
    double fairness = 0, origin_score = 0;
    double fork = 0;
    double mass = 0, formation_scale = 1;
    double income = 0;
    double reversal = 0;
    double screen = 2, fortify = 1;
    double revalidate = 1, occupation = 1, release = 1;
    std::vector<int> owned_cities;
    mutable std::vector<double> exposure_cache;
    Army last_army = 0;
    int score_cursor = 0;
    std::vector<unsigned char> origin_blocks;
    std::vector<double> origin_scores;
    Army reserve_guard = 0;
    std::vector<std::array<Army,7>> hostile_reach;
    int safety_slack = -far_away;
    int rally_root = -1, rally_started = 0;
    Army rally_target = 0;
    Army home_pressure = 0;
    int deadline = far_away;
    Offer best;
    std::chrono::steady_clock::time_point search_until = std::chrono::steady_clock::time_point::max();
    bool search_expired() const { return std::chrono::steady_clock::now() >= search_until; }

    Agent(int player, std::uint64_t, const char *params) : seat(player) {
        deployment_mode = int(std::clamp(option(params, "\"deployment_mode\"", 0), 0., 2.));
        route_labels.reserve(4096); route_open.reserve(4096); best.path.reserve(32);
        delivered_income = option(params, "\"delivered_income\"", 1) > .5;
        online_progress = option(params, "\"online_progress\"", 1) > .5;
        liquidity_horizon = option(params, "\"liquidity_horizon\"", 1) > .5;
        reroute_home = option(params, "\"reroute_home\"", 1) > .5;
        home_transit = option(params, "\"home_transit\"", 1) > .5;
        horizon = std::clamp(option(params, "\"horizon\"", 120.0), 20., 120.);
        recon = std::clamp(option(params, "\"recon\"", 0.4), 0., 4.);
        mobilize = std::clamp(option(params, "\"mobilize\"", 1.0), 0., 4.);
        caution = std::clamp(option(params, "\"caution\"", 1.0), .2, 3.);
        investment = std::clamp(option(params, "\"investment\"", 1.0), 0., 3.);
        troop_price = std::clamp(option(params, "\"troop_price\"", 0.15), .05, 3.);
        goal_rally = option(params, "\"goal_rally\"", 1) != 0;
        hold = std::clamp(option(params, "\"hold\"", 1.0), 0., 4.);
        route_budget = int(std::clamp(option(params, "\"route_budget\"", 1024), 128., 1024.));
        half_budget = int(std::clamp(option(params, "\"half_budget\"", 256), 128., 1024.));
        muster = std::clamp(option(params, "\"muster\"", 0.5), 0., 4.);
        probe = std::clamp(option(params, "\"probe\"", 0.4), 0., 4.);
        pressure = std::clamp(option(params, "\"pressure\"", 0.5), 0., 4.);
        phases = std::clamp(option(params, "\"phases\"", 0.7), 0., 1.);
        menace = std::clamp(option(params, "\"menace\"", 1), 0., 3.);
        forward_rally = option(params, "\"forward_rally\"", 1) > .5;
        rally = std::clamp(option(params, "\"rally\"", 1), 0., 4.);
        waves = std::clamp(option(params, "\"waves\"", 1), 0., 4.);
        development = std::clamp(option(params, "\"development\"", 1), 0., 1.);
        intercept = std::clamp(option(params, "\"intercept\"", 1), 0., 2.);
        hunt = std::clamp(option(params, "\"hunt\"", 0.5), 0., 3.);
        counter = std::clamp(option(params, "\"counter\"", 0), 0., 3.);
        preserve = std::clamp(option(params, "\"preserve\"", 0), 0., 1.);
        combat = std::clamp(option(params, "\"combat\"", 0.5), 0., 3.);
        enemy_price = std::clamp(option(params, "\"enemy_price\"", 1), 0., 1.);
        local = std::clamp(option(params, "\"local\"", 1), 0., 2.);
        setup = std::clamp(option(params, "\"setup\"", -0.25), -.5, 2.);
        return_plan = std::clamp(option(params, "\"return_plan\"", 0), 0., 1.);
        ghost = std::clamp(option(params, "\"ghost\"", 0), 0., 1.);
        coordinated = std::clamp(option(params, "\"coordinated\"", 0), 0., 1.);
        tenure = std::clamp(option(params, "\"tenure\"", 1), 0., 1.);
        skirmish = std::clamp(option(params, "\"skirmish\"", 0), 0., 2.);
        siege = std::clamp(option(params, "\"siege\"", 0), 0., 1.);
        frontier = std::clamp(option(params, "\"frontier\"", 1), 0., 3.);
        pulse = std::clamp(option(params, "\"pulse\"", 1), 0., 1.);
        insurance = std::clamp(option(params, "\"insurance\"", 0), 0., 3.);
        lead = std::clamp(option(params, "\"lead\"", 0), 0., 3.);
        fairness = std::clamp(option(params, "\"fairness\"", 0), 0., 1.);
        fork = std::clamp(option(params, "\"fork\"", 0), 0., 2.);
        mass = std::clamp(option(params, "\"mass\"", 0), 0., 2.);
        income = std::clamp(option(params, "\"income\"", 0), 0., 1.);
        reversal = std::clamp(option(params, "\"reversal\"", 1), 0., 1.);
        screen = std::clamp(option(params, "\"screen\"", 2), 0., 2.);
        fortify = std::clamp(option(params, "\"fortify\"", 1), 0., 1.);
        revalidate = std::clamp(option(params, "\"revalidate\"", 1), 0., 1.);
        occupation = std::clamp(option(params, "\"occupation\"", 1), 0., 1.);
        release = std::clamp(option(params, "\"release\"", 1), 0., 1.);
        continuity = int(std::clamp(option(params, "\"continuity\"", 8), 0., 24.));
        opening_delay = int(std::clamp(option(params, "\"opening_delay\"", 24), 0., 36.));
    }
    bool ours(int c) const { return c >= 0 && view->owner[c] == seat; }
    bool obstructed(int c) const {
        return c < 0 || memory[c].terrain == 1 || memory[c].terrain == 2;
    }
    Army troops(int c) const { return std::max<Army>(0, view->army[c]); }
    Army sent(const Move &m) const {
        return m.from < 0 ? 0 : m.split ? troops(m.from) / 2 : std::max<Army>(0, troops(m.from) - 1);
    }
    int direction(const Move &m) const {
        if (m.from < 0) return -1;
        for (int d = 0; d < 4; ++d) if (edge[m.from][d] == m.to) return d;
        return -1;
    }
    Army defense(int c) const {
        if (ours(c)) return 0;
        const Memory &m = memory[c];
        if (view->visible[c]) return troops(c);
        if (m.owner < 0) return m.soldiers;
        int age = now - m.observed;
        Army count = m.building > 0 && m.building < 3 ? m.soldiers + age / 2 :
                     std::max<Army>(1, m.soldiers * std::max(0, 16 - age) / 16);
        return std::min(count, std::max<Army>(1, view->total_army[m.owner]));
    }
    Army anticipated_growth(int c, int elapsed) const {
        if (elapsed <= 0 || memory[c].owner < 0) return 0;
        Army growth=(now+elapsed)/50-now/50;
        if (memory[c].building > 0 && memory[c].building < 3) growth+=(now+elapsed)/2-now/2;
        return growth;
    }
    Army column_growth(int c, int completed_steps) const {
        if (completed_steps<=0) return 0;
        int tick=now+completed_steps;
        return (tick%50==0) + (tick%2==0 && memory[c].building>0 && memory[c].building<3);
    }
    void distances(const std::vector<int> &starts, std::vector<int> &result, bool friendly = false) {
        result.assign(cells, far_away);
        queue.clear();
        for (int c : starts) if (c >= 0 && result[c] != 0) {
            result[c] = 0; queue.push_back(c);
        }
        for (std::size_t q = 0; q < queue.size(); ++q) {
            int u = queue[q];
            for (int v : edge[u]) if (!obstructed(v) && result[v] == far_away && (!friendly || ours(v))) {
                result[v] = result[u] + 1; queue.push_back(v);
            }
        }
    }
    bool previous_move_receipt() const {
        return last_move.from >= 0 && last_tick + 1 == now && view->previous &&
            view->previous->valid && view->previous->action ==
                arena_move(last_move.from, direction(last_move), last_move.split);
    }
    bool observed_departure() const {
        if (!online_progress || view->previous || last_tick + 1 != now ||
            last_move.from < 0 || last_move.to < 0 || !ours(last_move.to) ||
            troops(last_move.to) <= 1 || !view->visible[last_move.from]) return false;
        Army departed = last_move.split ? last_army / 2 : last_army - 1;
        if (departed <= 0) return false;
        Army growth = now / 50 - last_tick / 50;
        int building = memory[last_move.from].building;
        if (building == 1 || building == 2) growth += now / 2 - last_tick / 2;
        // This does not acknowledge execution. The observed destination can
        // support the remaining route and the source no longer holds the
        // dispatched force. The whole route is re-evaluated before bidding.
        return troops(last_move.from) <= last_army - departed + growth;
    }
    void observe(const ArenaObservation *observation) {
        arrival_cache_live = false;
        departure_ready.fill(false);
        retained_cache_cell = -1;
        view = observation; now = int(view->tick); home = view->general_pos;
        if (!cells) {
            width = view->width; height = view->height; cells = width * height;
            memory.resize(cells); edge.resize(cells); unknown.resize(cells);
            cell_x.resize(cells);cell_y.resize(cells);formation_forward.resize(cells);
            frontier_cache.resize(cells);compatible_cache.resize(cells);
            enemy_near.resize(cells); queue.reserve(cells);
            formation_bits.resize(cells); information.resize(cells);
            route_chosen.resize(cells);
            blocks.resize(cells);
            capital_belief.resize(cells); capital_information.resize(cells);
            hostile_reach.resize(cells);
            exposure_cache.resize(cells*8+1);
            origin_blocks.resize(cells); origin_scores.assign(cells,0);
            for (int c = 0; c < cells; ++c) {
                int x = c % width, y = c / width;
                cell_x[c]=x;cell_y[c]=y;
                edge[c] = {y ? c - width : -1, x + 1 < width ? c + 1 : -1,
                           y + 1 < height ? c + width : -1, x ? c - 1 : -1};
                origin_blocks[c] = view->visible[c] ? view->terrain[c] == 1 || view->structure[c] == 1 : view->fog_obstacles[c];
            }
            if (fairness && view->players == 2) origin_score = opening_score(home);
        }
        std::fill(exposure_cache.begin(),exposure_cache.end(),std::numeric_limits<double>::quiet_NaN());
        if (last_tick + 1 == now && last_move.from == column && ours(last_move.to)) {
            bool accepted = view->previous ? view->previous->valid &&
                view->previous->action == arena_move(last_move.from, direction(last_move), last_move.split) :
                troops(last_move.to) > 1;
            if (accepted) column = last_move.to;
        }
        if (last_campaign >= 0) {
            auto &path = campaigns[last_campaign].path;
            bool accepted = (previous_move_receipt() || observed_departure()) &&
                ours(last_move.to) && path.size() > 1 && path[0] == last_move.from && path[1] == last_move.to;
            if (accepted) path.erase(path.begin()); else path.clear();
            last_campaign = -1;
        }
        owned.clear(); owned_cities.clear(); mobile.clear(); enemies.clear(); objectives.clear();
        for (int c = 0; c < cells; ++c) {
            Memory &m = memory[c];
            if (view->visible[c]) {
                m = {int(view->terrain[c]), int(view->structure[c]), int(view->owner[c]), now, troops(c)};
            } else {
                if (m.terrain == 255 && view->fog_obstacles[c]) m.terrain = 2;
                if (m.owner == seat || (m.owner >= 0 && !view->alive[m.owner])) {
                    m.owner = -3; m.soldiers = 0;
                    if (m.building == 2) m.building = 1;
                }
            }
            if (ours(c)) { owned.push_back(c); if (m.building == 1) owned_cities.push_back(c); if (troops(c) > 1) mobile.push_back(c); }
            if (view->visible[c] && m.owner >= 0 && m.owner != seat && view->alive[m.owner]) enemies.push_back(c);
            if (m.owner >= 0 && m.owner != seat && m.building == 2 && view->alive[m.owner]) objectives.push_back(c);
        }
        // An opponent that has seen our capital can keep aiming at it after
        // its scout leaves. Only living, actually disclosed-to rivals count.
        for(int c:enemies)if(std::abs(c%width-home%width)<=1 && std::abs(c/width-home/width)<=1)
            home_revealed_to|=1u<<memory[c].owner;
        capital_disclosed=false;
        for(int p=0;p<view->players;++p)
            capital_disclosed|=view->alive[p] && (home_revealed_to&(1u<<p));
        if (!ours(column) || troops(column) < 3) column = -1;
        if (!ours(assembly_root) || ours(assembly_goal)) assembly_goal = assembly_root = -1;
        distances({home}, home_distance);
        distances(enemies, contact_distance);
        distances(objectives, king_distance);
        distances({home}, owned_home_distance, true);
        locate_capitals();
        trace_fronts();
        production = .5 * (view->total_cities[seat] + 1) + .02 * view->total_land[seat];
        Army opponent_land = 0;
        for (int p=0;p<view->players;++p) if (p!=seat && view->alive[p])
            opponent_land=std::max(opponent_land,view->total_land[p]);
        command_setup=setup+(now>=100?lead*std::clamp((view->total_land[seat]-opponent_land-3.)/10.,0.,1.):0);
        // A production base funds sustained operations. Further peaceful
        // purchases have lower marginal value; enemy production denial does not.
        readiness = std::clamp((production - 1.1) / 2.4, 0., 1.);
        economy_weight = 1 - .55 * phases * readiness;
        for (int c = 0; c < cells; ++c) {
            unknown[c] = 0; enemy_near[c] = 0; information[c] = 0;
            int x = c % width, y = c / width;
            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                int xx = x + dx, yy = y + dy;
                if (xx < 0 || yy < 0 || xx >= width || yy >= height) continue;
                int v = yy * width + xx;
                unknown[c] += memory[v].observed < 0 && memory[v].terrain != 1 && memory[v].terrain != 2;
                if (!view->visible[v] && !obstructed(v))
                    information[c] += memory[v].observed < 0 ? 1. :
                        .3 * std::clamp((now - memory[v].observed) / 50., 0., 1.);
                // The target's own defenders are paid by capture already;
                // only other cells can launch the subsequent counterattack.
                if (v != c && view->visible[v] && memory[v].owner >= 0 && memory[v].owner != seat)
                    enemy_near[c] = std::max(enemy_near[c], troops(v));
            }
        }
        std::sort(mobile.begin(), mobile.end(), [&](int a, int b) {
            double aa = double(troops(a)) * (a == column ? 1.4 : 1.);
            double bb = double(troops(b)) * (b == column ? 1.4 : 1.);
            return aa != bb ? aa > bb : a < b;
        });
        Army opposing_spare = 0;
        for (int p = 0; p < view->players; ++p) if (p != seat && view->alive[p])
            opposing_spare = std::max(opposing_spare, view->total_army[p] - view->total_land[p]);
        desired_force = std::clamp(12 + .35 * (width + height) + .35 * opposing_spare, 24., 160.+1888*mass);
        formation_scale=std::min(4.,1+mass*std::max(0.,std::sqrt(desired_force/40)-1));
        reserve_guard = now < 100 ? 0 : Army(std::clamp(.18 * opposing_spare,6.,60.));
        economic_horizon=horizon;
        if(liquidity_horizon && capital_disclosed) {
            // Credit home troops plus one nearby column that can return on
            // owned land. Extra road garrisons and future growth are omitted.
            int return_steps=std::clamp(contact_distance[home],1,8);
            Army support=0;
            for(int c:mobile)if(c!=home && owned_home_distance[c]<=return_steps)
                support=std::max(support,std::max<Army>(0,troops(c)-1));
            double ready=std::clamp(double(troops(home)+support)/desired_force,0.,1.);
            economic_horizon=horizon*(.25+.75*ready);
        }
        if(delivered_income) prepare_delivery();
        if(menace) prepare_menace();
        prepare_deployment();
        std::fill(frontier_cache.begin(),frontier_cache.end(),std::numeric_limits<double>::quiet_NaN());
        std::fill(compatible_cache.begin(),compatible_cache.end(),0);
        formations.clear(); std::fill(formation_bits.begin(), formation_bits.end(), 0);
        for (int c : mobile) formations.emplace_back(formation(c, troops(c)), c);
        std::sort(formations.begin(), formations.end(), std::greater<std::pair<double,int>>());
        if (formations.size() > 9) formations.resize(9);
        for (std::size_t i = 0; i < formations.size(); ++i)
            formation_bits[formations[i].second] = 1u << i;
        // Evaluate each available subset once. Its best formation either
        // excludes the first member, or pairs that member with the strongest
        // compatible survivor. This is the same two-corps objective as the
        // exhaustive pair scan, with bounded work per subset.
        std::array<unsigned,9> compatible{};
        for(std::size_t i=0;i<formations.size();++i)
            for(std::size_t j=i+1;j<formations.size();++j)
                if(separated(formations[i].second,formations[j].second))compatible[i]|=1u<<j;
        std::array<double,512> available{};
        unsigned all=(1u<<formations.size())-1;
        for(unsigned mask=1;mask<=all;++mask) {
            unsigned first=unsigned(__builtin_ctz(mask)),rest=mask&(mask-1);
            double value=std::max(available[rest],formations[first].first);
            unsigned partner=rest&compatible[first];
            if(partner)value=std::max(value,formations[first].first+.65*formations[__builtin_ctz(partner)].first);
            available[mask]=value;
        }
        for(unsigned mask=0;mask<=all;++mask)remaining_formation[mask]=available[all^mask];
    }
    void trace_fronts() {
        if (!tenure) return;
        std::fill(hostile_reach.begin(),hostile_reach.end(),std::array<Army,7>{});
        std::vector<int> starts = enemies;
        std::sort(starts.begin(),starts.end(),[&](int a,int b) {return troops(a)>troops(b);});
        if (starts.size()>12) starts.resize(12);
        std::vector<int> distance(cells), walk;
        std::vector<Army> remaining(cells);
        for (int source : starts) {
            if (troops(source)<4) continue;
            std::fill(distance.begin(),distance.end(),far_away);
            std::fill(remaining.begin(),remaining.end(),0);
            distance[source]=0; remaining[source]=troops(source); walk.assign(1,source);
            for (std::size_t i=0;i<walk.size();++i) {
                int u=walk[i], d=distance[u]+1;
                if (d>6 || remaining[u]<=1) continue;
                for (int v:edge[u]) if (!obstructed(v) && distance[v]>=d) {
                    Army arriving=remaining[u]-1;
                    hostile_reach[v][d]=std::max(hostile_reach[v][d],arriving);
                    Army guard=ours(v)?troops(v):defense(v);
                    Army next=arriving+(memory[v].owner==memory[source].owner?guard:-guard);
                    if (distance[v]==far_away) {distance[v]=d;walk.push_back(v);}
                    remaining[v]=std::max(remaining[v],next);
                }
            }
        }
    }
    void locate_capitals() {
        std::fill(capital_belief.begin(), capital_belief.end(), 0);
        std::fill(capital_information.begin(), capital_information.end(), 0);
        if (!hunt) return;
        if (fairness && view->players == 2) for (int budget=0;score_cursor<cells && budget<32;++score_cursor,++budget)
            if (!origin_blocks[score_cursor]) origin_scores[score_cursor]=opening_score(score_cursor);
        bool known[8]{};
        for (int c : objectives) known[memory[c].owner] = true;
        int unresolved = 0;
        for (int p = 0; p < view->players; ++p) unresolved += p != seat && view->alive[p] && !known[p];
        if (!unresolved) return;
        std::vector<int> territory;
        for (int c = 0; c < cells; ++c) if (memory[c].owner >= 0 && memory[c].owner != seat &&
            view->alive[memory[c].owner] && !known[memory[c].owner]) territory.push_back(c);
        distances(territory, known_enemy_distance);
        // The standard map's public separation is only a soft prior, so custom
        // layouts and moved capitals do not become impossible hypotheses.
        double spacing = std::max(9., 9 * (1.8 - .1 * view->players));
        double total = 0;
        for (int c = 0; c < cells; ++c) if (memory[c].observed < 0 && !obstructed(c)) {
            int md = std::abs(c % width - home % width) + std::abs(c / width - home / width);
            double prior = md > spacing ? 1. : .05;
            if (fairness && view->players == 2 && origin_scores[c] > 0 && origin_score > 0) {
                double ratio=std::min(origin_scores[c],origin_score)/std::max(origin_scores[c],origin_score);
                double mismatch=std::max(0.,.91-ratio)/.07;
                // The published opening-accessibility constraint is a soft
                // prior only: relaxed generation and custom maps remain possible.
                prior *= 1-fairness+fairness*(.05+.95*std::exp(-mismatch*mismatch));
            }
            if (!territory.empty()) {
                double d = std::min(40, known_enemy_distance[c]);
                prior /= 1 + d * d / 36.;
            }
            capital_belief[c] = prior; total += prior;
        }
        if (!total) return;
        for (double &p : capital_belief) p *= unresolved / total;
        for (int c = 0; c < cells; ++c) {
            int x = c % width, y = c / width;
            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                int xx = x + dx, yy = y + dy;
                if (xx >= 0 && yy >= 0 && xx < width && yy < height)
                    capital_information[c] += capital_belief[yy * width + xx];
            }
        }
    }
    double opening_score(int source) const {
        std::vector<int> depth(cells,-1), walk{source}; depth[source]=0;
        std::array<int,15> count{};
        for (std::size_t i=0;i<walk.size();++i) {
            int u=walk[i],d=depth[u];++count[d];if(d==14)continue;
            for(int v:edge[u])if(v>=0 && !origin_blocks[v] && depth[v]<0){depth[v]=d+1;walk.push_back(v);}
        }
        count[1]=std::min(3,count[1]);double weight=int(std::pow(1.18,15)),score=0;
        for(int d=1;d<=14;++d){score+=weight*count[d];weight/=d<7?1.18:1.1;if(d<5)weight/=1.6;}
        return score;
    }
    #include "menace.h"
    #include "deployment.h"
    bool separated(int a, int b) const {
        return std::abs(cell_x[a]-cell_x[b])+std::abs(cell_y[a]-cell_y[b]) >= 6;
    }
    double formation(int c, Army force) const {
        double ready = std::clamp((force - 1.) / desired_force, 0., 1.);
        double forward = formation_forward[c];
        double value = muster * 60 * forward * ready * ready;
        if (!menace && pressure && king_distance[c] <= 12 && king_distance[c] > 0) {
            double force_ready = std::clamp((force - king_distance[c] - 4.) / desired_force, 0., 1.);
            value += pressure * 45 * force_ready * force_ready / (1 + .25 * king_distance[c]);
        }
        if(menace) value += menace_value(c,force);
        return value*formation_scale;
    }
    Army rally_size(Army spare) const {
        double fraction=.14+.21*mass*std::clamp((now-150.)/200.,0.,1.);
        return Army(std::min(desired_force,std::max(20.,18+fraction*spare)));
    }
    double mission_value(int cell, Army force, unsigned consumed, int retained_cell = -1, Army retained_force = 0) const {
        double value = 0;
        if ((muster || pressure) && !formations.empty()) {
            double candidate = formation(cell, force);
            double after = std::max(candidate, remaining_formation[consumed]);
            unsigned partners = compatible_formations(cell) & ~consumed;
            if(partners) {
                double other=formations[__builtin_ctz(partners)].first;
                after=std::max(after,std::max(candidate,other)+.65*std::min(candidate,other));
            }
            if (retained_cell >= 0 && retained_force > 1) {
                if(retained_cache_cell!=retained_cell || retained_cache_force!=retained_force) {
                    retained_cache_cell=retained_cell;retained_cache_force=retained_force;
                    retained_cache_value=formation(retained_cell,retained_force);
                }
                double retained=retained_cache_value;
                after = std::max(after, retained);
                if (separated(retained_cell, cell))
                    after = std::max(after, std::max(candidate, retained) + .65 * std::min(candidate, retained));
                unsigned partners=compatible_formations(retained_cell)&~consumed;
                if(partners) {
                    double other=formations[__builtin_ctz(partners)].first;
                    after=std::max(after,std::max(retained,other)+.65*std::min(retained,other));
                }
            }
            value += after - remaining_formation[0];
            // Before a sustainable production base exists, troops invested in
            // a first neutral city are not valued as a permanent loss of an
            // established assault corps. Mature armies retain their full cost.
            if (value < 0 && memory[cell].building == 1 && memory[cell].owner < 0)
                value *= 1 - development * .75 * (1 - readiness);
        }
        if (probe && information[cell] > 0) {
            double ready = std::min(1., double(force) / desired_force);
            value += probe * (12 + 1.2 * std::min(35, home_distance[cell])) *
                (std::min(3., information[cell]) / 3.) * ready * ready * (objectives.empty() ? 1. : .55);
        }
        if (hunt) value += hunt * 400 * capital_information[cell] *
            (.35 + .65 * std::clamp(force / 24., 0., 1.));
        return value;
    }

    // Relative priority comes from the environment protocol, not from a policy.
    bool enemy_precedes(const Move &m, int source, int destination, int player) const {
        if (destination == m.from && source != m.to) return true;
        if (m.to == source && m.from != destination) return false;
        bool ally = memory[destination].owner == player, mine = ours(m.to);
        if (ally != mine) return ally;
        bool king = memory[destination].building == 2, own_king = memory[m.to].building == 2;
        if (king != own_king) return !king;
        if (troops(source) != troops(m.from)) return troops(source) > troops(m.from);
        return now % 2 ? player > seat : player < seat;
    }
    bool executable(const Move &m) const {
        if (!ours(m.from) || obstructed(m.to) || direction(m) < 0 || sent(m) == 0) return false;
        Army attacks[8]{};
        for (int e : edge[m.from]) if (e >= 0 && view->visible[e] &&
            memory[e].owner >= 0 && memory[e].owner != seat &&
            enemy_precedes(m, e, m.from, memory[e].owner)) {
            int player = memory[e].owner;
            attacks[player] = std::max(attacks[player], std::max<Army>(0, troops(e) - 1));
        }
        Army total = 0; for (Army x : attacks) total += x;
        return total < troops(m.from) - 1;
    }

    void trace_threats() {
        std::fill(exposure_cache.begin(),exposure_cache.end(),std::numeric_limits<double>::quiet_NaN());
        arrival_cache_live = false;
        threats.clear(); home_pressure = 0; deadline = far_away;
        for (auto &b : blocks) b.clear();
        struct Start { int cell, player; Army force; };
        std::vector<Start> starts;
        for (int e : enemies) if (troops(e) > 1 && home_distance[e] <= 12)
            starts.push_back({e,memory[e].owner,troops(e)});
        if (ghost) {
            std::vector<int> candidates;
            for (int e = 0; e < cells; ++e) if (!view->visible[e] && memory[e].owner >= 0 &&
                memory[e].owner != seat && view->alive[memory[e].owner] && memory[e].soldiers >= 12 &&
                now - memory[e].observed <= 10 && home_distance[e] <= 20) candidates.push_back(e);
            std::sort(candidates.begin(),candidates.end(),[&](int a,int b) {
                return defense(a) > defense(b);
            });
            if (candidates.size() > 4) candidates.resize(4);
            std::vector<int> depth(cells), walk;
            for (int e : candidates) {
                std::fill(depth.begin(),depth.end(),far_away);
                walk.assign(1,e); depth[e] = 0; int nearest = e;
                for (std::size_t j = 0; j < walk.size(); ++j) {
                    int u = walk[j];
                    if (home_distance[u] < home_distance[nearest]) nearest = u;
                    if (depth[u] >= now - memory[e].observed) continue;
                    for (int v : edge[u]) if (!obstructed(v) && !view->visible[v] && depth[v] == far_away) {
                        depth[v] = depth[u]+1; walk.push_back(v);
                    }
                }
                Army force = defense(e) - depth[nearest];
                if (force > 1 && home_distance[nearest] <= 12)
                    starts.push_back({nearest,memory[e].owner,force});
            }
        }
        std::sort(starts.begin(), starts.end(), [&](const Start &aa_, const Start &bb_) {
            int a = aa_.cell, b = bb_.cell;
            int urgency_a = std::max(0, 3 - home_distance[a]);
            int urgency_b = std::max(0, 3 - home_distance[b]);
            if (urgency_a != urgency_b) return urgency_a > urgency_b;
            double aa = double(aa_.force) / (1 + home_distance[a]);
            double bb = double(bb_.force) / (1 + home_distance[b]);
            return aa != bb ? aa > bb : a < b;
        });
        if (starts.size() > 8) starts.resize(8);
        using Node = std::pair<double, int>;
        std::vector<double> cost(cells);
        std::vector<int> predecessor(cells), depth(cells);
        for (const Start &start : starts) for (double avoidance : {0., 6.}) {
            int e = start.cell;
            std::fill(cost.begin(), cost.end(), 1e30);
            std::fill(predecessor.begin(), predecessor.end(), -1);
            std::fill(depth.begin(), depth.end(), far_away);
            std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open;
            cost[e] = 0; depth[e] = 0; open.emplace(0, e);
            int enemy = start.player;
            while (!open.empty()) {
                auto [score, u] = open.top(); open.pop();
                if (score != cost[u]) continue;
                if (u == home) break;
                if (depth[u] == 12) continue;
                for (int v : edge[u]) if (!obstructed(v)) {
                    double resistance = memory[v].owner == enemy || v == home ? 0 :
                                        ours(v) ? double(troops(v)) : double(defense(v));
                    double next = score + 1 + avoidance * resistance / (start.force + 1.);
                    if (next < cost[v]) {
                        cost[v] = next; predecessor[v] = u; depth[v] = depth[u] + 1;
                        open.emplace(next, v);
                    }
                }
            }
            if (predecessor[home] < 0 || depth[home] > 12) continue;
            Threat t{enemy, {}, start.force};
            for (int c = home; c >= 0; c = predecessor[c]) t.corridor.push_back(c);
            std::reverse(t.corridor.begin(), t.corridor.end());
            threats.push_back(std::move(t));
        }
        // Most possible commands do not touch a given invasion corridor.
        // Their arriving enemy force is identical, regardless of move order.
        // Home guard changes are charged separately by exposure().
        threat_touch.assign(cells, 0);
        unsigned index = 0;
        for (Threat &t : threats) {
            t.nominal_arrival = uncached_arrival(t, Move{});
            t.touch_bit = 1u << index++;
            for (int c : t.corridor) if (c != home) threat_touch[c] |= t.touch_bit;
        }
        arrival_cache_live = true;
        for (const Threat &t : threats) {
            Army incoming = arrival(t, Move{});
            int d = int(t.corridor.size()) - 1;
            Army home_growth=anticipated_growth(home,d-1);
            if (incoming > troops(home) + home_growth) {
                home_pressure = std::max(home_pressure, incoming - home_growth);
                deadline = std::min(deadline, d);
                if (intercept) {
                    // Upper-bound a pursuing column without relying on guards
                    // that our own interception route might collect first.
                    Army force = t.initial;
                    for (int j = 1; j <= d; ++j) {
                        int c = t.corridor[j]; force += column_growth(t.corridor[j-1],j-1)-1;
                        blocks[c].push_back({j, force, incoming - troops(home) - home_growth});
                        if (memory[c].owner == t.player)
                            force += defense(c) + anticipated_growth(c,j-1);
                    }
                }
            }
        }
    }
    double interception_value(int c, Army force, int steps) const {
        double value = 0;
        for (const auto &b : blocks[c]) if (steps <= b.arrival && force > b.force + 2) {
            double utility = (20 + 4. * std::min<Army>(40, b.shortage)) / (.8 + steps);
            if (steps + 1 >= b.arrival && (!coordinated || safety_slack <= 2)) utility += 300;
            value = std::max(value, intercept * utility);
        }
        if (screen && value > 0) {
            // A block earns a capital-rescue premium only if it covers every
            // currently lethal corridor. Blocking one small column must not
            // divert a timely return from another column that can bypass it.
            for (const Threat &t : threats) {
                int length = int(t.corridor.size()) - 1;
                if (arrival(t,Move{}) <= troops(home)+anticipated_growth(home,length-1)) continue;
                Army approaching = t.initial;
                bool covered = false;
                for (int j=1;j<=length;++j) {
                    approaching += column_growth(t.corridor[j-1],j-1)-1;
                    int v=t.corridor[j];
                    if (v==c) {covered=steps<=j && force>approaching+2;break;}
                    // Do not rely on our garrisons: the proposed route may
                    // collect them before the attacking column reaches them.
                    if (memory[v].owner==t.player) approaching+=defense(v)+anticipated_growth(v,j-1);
                }
                if (!covered) return 0;
            }
        }
        return value;
    }
    Army arrival(const Threat &t, const Move &m) const {
        if (arrival_cache_live && t.touch_bit &&
            (m.from < 0 || !(threat_touch[m.from] & t.touch_bit)) &&
            (m.to < 0 || !(threat_touch[m.to] & t.touch_bit))) return t.nominal_arrival;
        return uncached_arrival(t, m);
    }
    Army uncached_arrival(const Threat &t, const Move &m) const {
        const auto &path = t.corridor;
        Army moving = sent(m), force = t.initial;
        bool earlier = m.from >= 0 && enemy_precedes(m, path[0], path[1], t.player);
        if (m.to == path[0] && !earlier) {
            force -= moving;
            if (force <= 0) return 0;
        }
        for (std::size_t j = 1; j < path.size(); ++j) {
            int c = path[j];
            force += column_growth(path[j-1],int(j)-1);
            if (--force <= 0) return 0;
            if (c == home) break;
            bool changed = !(j == 1 && earlier);
            int owner = memory[c].owner;
            Army guard = (owner == seat ? troops(c) : defense(c)) + anticipated_growth(c,int(j)-1);
            if (changed && c == m.from) guard -= moving;
            if (changed && c == m.to) {
                if (owner == seat) guard += moving;
                else { guard = std::abs(guard - moving); if (moving > defense(c)) owner = seat; }
            }
            if (owner == t.player) force += guard;
            else force -= guard;
            if (force <= 0) return 0;
        }
        return force;
    }

    const std::vector<Threat>& bypass_routes(int blocked) const {
        int dir=direction({home,blocked,false});
        static const std::vector<Threat> empty;
        if(dir<0)return empty;
        auto &result=departure_routes[dir];
        if(departure_ready[dir])return result;
        departure_ready[dir]=true;result.clear();
        using Node=std::pair<double,int>;
        std::vector<double> cost(cells);
        std::vector<int> parent(cells),depth(cells),seen;
        // Current routes can all pass through the square receiving our army.
        // A rival is free to bypass that newly defended square. Keep these
        // alternatives separate from the existing cached corridor bitmask.
        for(const Threat &old:threats) {
            int source=old.corridor.front();
            if(source==blocked || std::find(seen.begin(),seen.end(),source)!=seen.end())continue;
            seen.push_back(source);
            std::fill(cost.begin(),cost.end(),1e30);std::fill(parent.begin(),parent.end(),-1);
            std::fill(depth.begin(),depth.end(),far_away);
            std::priority_queue<Node,std::vector<Node>,std::greater<Node>> open;
            cost[source]=0;depth[source]=0;open.emplace(0,source);
            while(!open.empty()) {
                auto [score,u]=open.top();open.pop();if(score!=cost[u])continue;
                if(u==home)break;
                if(depth[u]>=12)continue;
                for(int v:edge[u])if(v!=blocked && !obstructed(v)) {
                    double resistance=memory[v].owner==old.player || v==home?0:
                        ours(v)?double(troops(v)):double(defense(v));
                    double next=score+1+6.*resistance/(old.initial+1.);
                    if(next<cost[v]) {cost[v]=next;parent[v]=u;depth[v]=depth[u]+1;open.emplace(next,v);}
                }
            }
            if(parent[home]<0 || depth[home]>12)continue;
            Threat threat{old.player,{},old.initial};
            for(int c=home;c>=0;c=parent[c])threat.corridor.push_back(c);
            std::reverse(threat.corridor.begin(),threat.corridor.end());result.push_back(std::move(threat));
        }
        return result;
    }

    double exposure(const Move &m) const {
        int dir=m.from<0?-1:direction(m);
        int key=m.from<0?cells*8:dir<0?-1:m.from*8+dir*2+int(m.split);
        if(key>=0 && !std::isnan(exposure_cache[key]))return exposure_cache[key];
        double value=uncached_exposure(m);
        if(key>=0)exposure_cache[key]=value;
        return value;
    }
    double uncached_exposure(const Move &m) const {
        Army guard = troops(home) - (m.from == home ? sent(m) : 0) + (m.to == home ? sent(m) : 0);
        Army waves[8][13]{};
        int removed = -1;
        if (m.to >= 0 && memory[m.to].building == 2 && !ours(m.to) && sent(m) > king_guard(m.to))
            removed = memory[m.to].owner;
        for (const Threat &t : threats) if (t.player != removed) {
            int d = int(t.corridor.size()) - 1;
            waves[t.player][d] = std::max(waves[t.player][d], arrival(t, m));
        }
        if(reroute_home && m.from==home && sent(m)>0)for(const Threat &t:bypass_routes(m.to))if(t.player!=removed) {
            int d=int(t.corridor.size())-1;
            waves[t.player][d]=std::max(waves[t.player][d],uncached_arrival(t,m));
        }
        double penalty = 0;
        Army cumulative[8]{};
        for (int d = 1; d <= 12; ++d) {
            Army total = 0;
            for (int p = 0; p < view->players; ++p) { cumulative[p] = std::max(cumulative[p], waves[p][d]); total += cumulative[p]; }
            Army shortage = total - guard - anticipated_growth(home,d-1);
            if (shortage > 0) penalty = std::max(penalty, d == 1 ? 1e6 + double(shortage) :
                caution * double(shortage) * 18. / (d * d));
        }
        return penalty + city_exposure(m) + launch_cost(m) + counter_cost(m) +
            insurance * std::max<Army>(0,reserve_guard-guard) + reversal_cost(m);
    }
    double reversal_cost(const Move &m) const {
        if (!reversal || m.from<0 || m.from==home || m.to==home || m.split || last_move.split ||
            !(previous_move_receipt() || observed_departure()) ||
            m.from!=last_move.to || m.to!=last_move.from || !ours(m.to) ||
            troops(m.from)<12 || troops(m.to)>2 || troops(m.from)>last_army+2 || deadline<=2) return 0;
        // Reversing the same unreinforced column through empty friendly cells
        // is not a second rescue. Immediate danger still outranks this cost.
        return 2000*reversal;
    }
    double counter_cost(const Move &m) const {
        if (!counter || m.from < 0 || m.to < 0) return 0;
        Army arriving = sent(m) + (ours(m.to) ? troops(m.to) : -defense(m.to));
        if (arriving <= 0) return 0;
        Army immediate = 0, pursuit = 0;
        for (int near : edge[m.to]) if (!obstructed(near) && near != m.from) {
            if (view->visible[near] && memory[near].owner >= 0 && memory[near].owner != seat)
                immediate = std::max(immediate, troops(near) - 1);
            // A stronger pursuer one move away can trap the arriving column.
            // Under the source-dependency rule it then attacks before we leave.
            for (int e : edge[near]) if (e >= 0 && e != m.to && e != m.from && view->visible[e] &&
                memory[e].owner >= 0 && memory[e].owner != seat && troops(e) > 1) {
                int player = memory[e].owner;
                Army guard = ours(near) ? troops(near) : defense(near);
                Army force = troops(e) - 1 + (memory[near].owner == player ? guard : -guard);
                pursuit = std::max(pursuit, force - 1);
            }
        }
        double loss = immediate >= arriving ? 6 + std::min<Army>(120, arriving) :
                      pursuit > arriving ? 2 + .75 * std::min<Army>(120, arriving) : 0;
        return counter * loss;
    }
    double launch_cost(const Move &m) const {
        if (!waves || m.from < 0 || memory[m.from].building != 2 || view->total_land[seat] < 20) return 0;
        // Dispatch production in useful waves instead of spending one command
        // to ship each new soldier. Joining a nearby formed army is already a
        // useful dispatch and needs no batching penalty.
        double threshold = std::min(12., desired_force / 3);
        double arriving = double(sent(m)) + (ours(m.to) ? troops(m.to) : 0);
        return waves * 2.4 * std::max(0., 1 - arriving / threshold);
    }
    double city_exposure(const Move &m) const {
        if (!hold) return 0;
        double loss = 0;
        for (int c : owned_cities) if (enemy_near[c] > 1) {
            Army attacks[8]{};
            for (int e : edge[c]) if (e >= 0 && view->visible[e] &&
                memory[e].owner >= 0 && memory[e].owner != seat) {
                Army strength = troops(e) - 1;
                if (m.to == e && !enemy_precedes(m, e, c, memory[e].owner)) strength -= sent(m);
                attacks[memory[e].owner] = std::max(attacks[memory[e].owner], strength);
            }
            Army force = 0; for (Army x : attacks) force += x;
            Army guard = troops(c) - (m.from == c ? sent(m) : 0) + (m.to == c ? sent(m) : 0);
            if (force > guard) {
                double cost=std::min(30., horizon * .5) * std::sqrt(double(force - guard) / (force + 1.));
                if(release && m.from==c) {
                    Army arriving=sent(m)+(ours(m.to)?troops(m.to):-defense(m.to));
                    // A large departing column can recapture a city beside
                    // it. Charge the forced return command and occupation,
                    // not the full horizon of lost production.
                    if(arriving>force+enemy_near[m.to]+4)cost=std::min(cost,2.+.2*force);
                }
                loss+=hold*cost;
            }
        }
        return loss;
    }
    void choose(const Move &m, double value, int target, Offer::Purpose purpose, int assembly = -1) {
        best.move = m; best.utility = value; best.objective = target; best.purpose = purpose;
        best.assembly = assembly; best.path.clear(); best.rally = -1;
    }
    void bid(const Move &m, double utility, int target, Offer::Purpose purpose, int assembly = -1) {
        if (!executable(m)) return;
        if (memory[m.to].building == 2 && memory[m.to].owner >= 0 && !ours(m.to) && sent(m) <= king_guard(m.to)) return;
        double value = utility - exposure(m);
        if (target == intent && (m.from == column || purpose == Offer::SUPPLY)) value += .25;
        if (value > best.utility) choose(m, value, target, purpose, assembly);
    }
    Army king_guard(int c) const {
        Army relief = 0;
        for (int v : edge[c]) if (v >= 0 && view->visible[v] && memory[v].owner == memory[c].owner)
            relief = std::max(relief, std::max<Army>(0, troops(v) - 1));
        return defense(c) + relief;
    }
    void immediate_moves() {
        for (int c : mobile) for (int v : edge[c]) if (!obstructed(v)) {
            for (bool half : {false, true}) {
                Move m{c, v, half}; Army strength = sent(m);
                if (hold && ours(v) && memory[v].building == 1 && enemy_near[v] > troops(v))
                    bid(m, 0, v, Offer::RESCUE);
                if (memory[v].building == 2 && memory[v].owner >= 0 && !ours(v) && strength > king_guard(v))
                    bid(m, 100000, v, Offer::FINISH);
                if (skirmish && view->visible[v] && memory[v].owner >= 0 && !ours(v)) {
                    bool king = memory[v].building == 2;
                    if (strength <= defense(v) || (king && strength <= king_guard(v))) {
                        double damage = std::min(strength,defense(v));
                        double rate = king ? .2 : home_distance[v] <= 8 ? .18 : .04;
                        double value = skirmish * damage * rate;
                        // A forcing strike can be useful before one stack is
                        // large enough for guaranteed capture. Failure to
                        // reinforce a directly threatened capital is terminal.
                        if (king && strength > defense(v)) value += skirmish * 100;
                        value -= exposure(m);
                        if (value > best.utility) choose(m,value,v,king ? Offer::FINISH : Offer::RESCUE);
                    }
                }
                if (!ours(v) && view->visible[v] && memory[v].owner >= 0 &&
                    strength > defense(v) && home_distance[v] <= 7 && troops(v) > 4) {
                    double reward = .25 * std::min<Army>(150, troops(v)) / (1 + .2 * home_distance[v]);
                    bid(m, reward, v, Offer::RESCUE);
                }
            }
        }
    }
    void relief_offers() {
        safety_slack = -far_away;
        if (home_pressure <= troops(home)) return;
        struct Delivery { int source, next, distance, slack; Army useful, before; std::vector<int> path; };
        std::vector<Delivery> deliveries;
        std::vector<int> parent(cells), distance(cells), work;
        std::vector<unsigned char> collected(cells);
        int considered = 0;
        for (int source : mobile) {
            if (source == home || owned_home_distance[source] > deadline || troops(source) < 4) continue;
            if (++considered > 24) break;
            std::fill(parent.begin(), parent.end(), -1);
            std::fill(distance.begin(), distance.end(), far_away);
            work.assign(1, home); distance[home] = 0;
            for (std::size_t i = 0; i < work.size() && distance[source] == far_away; ++i) {
                int u = work[i]; if (distance[u] >= deadline) continue;
                for (int v : edge[u]) if (ours(v) && distance[v] == far_away &&
                    (v == source || enemy_near[v] < troops(source) + troops(v) / 2)) {
                    parent[v] = u; distance[v] = distance[u] + 1; work.push_back(v);
                }
            }
            if (distance[source] > deadline) continue;
            Army delivered = troops(source) - distance[source];
            for (int c = parent[source]; c != home; c = parent[c]) delivered += troops(c);
            std::fill(collected.begin(), collected.end(), 0);
            for (int c = source; c != home; c = parent[c]) collected[c] = 1;
            // Guards on the invasion route already defend the capital.
            // Relocating them behind the same attacker adds no new protection.
            Army before = 0, after = 0;
            for (const Threat &t : threats) {
                int length = int(t.corridor.size()) - 1;
                before = std::max(before, arrival(t, Move{}) - troops(home) - anticipated_growth(home,length-1));
                Army force = t.initial;
                for (std::size_t i = 1; i < t.corridor.size(); ++i) {
                    force += column_growth(t.corridor[i-1],int(i)-1);
                    int c = t.corridor[i]; if (--force <= 0 || c == home) break;
                    Army guard = ours(c) && collected[c] ? 1 :
                        (ours(c) ? troops(c) : defense(c)) + anticipated_growth(c,int(i)-1);
                    force += memory[c].owner == t.player ? guard : -guard;
                    if (force <= 0) break;
                }
                after = std::max(after, force - troops(home) - delivered - anticipated_growth(home,length-1));
            }
            Army useful = std::max<Army>(0, before - after);
            if (useful <= 0) continue;
            int slack = deadline - distance[source];
            if (useful >= before) safety_slack = std::max(safety_slack,slack);
            Delivery delivery{source,parent[source],distance[source],slack,useful,before,{}};
            if (return_plan || screen>=2) for (int c = source; c >= 0; c = parent[c]) {
                delivery.path.push_back(c); if (c == home) break;
            }
            deliveries.push_back(std::move(delivery));
        }
        for (const auto &delivery : deliveries) {
            int source = delivery.source, slack = delivery.slack;
            Army useful = delivery.useful, before = delivery.before;
            double utility = 3. * useful / delivery.distance + 2. * useful / (1 + slack);
            // Missing the last return window is irreversible. This is not a
            // general preference for retreat: only new, deliverable protection
            // against an already lethal visible corridor gets the deadline bid.
            bool urgent = useful >= before && ((coordinated || screen>=2) ? safety_slack <= 2 : slack <= 2);
            if (urgent) utility += 1000;
            double previous = best.utility;
            bid({source, delivery.next, false}, utility, home, Offer::RESCUE);
            if ((return_plan || screen>=2) && best.utility > previous && urgent) best.path = delivery.path;
        }
    }
    void defense_tree() {
        if (!fortify || home_pressure<=troops(home) || deadline<1 || deadline>12) return;
        std::vector<int> parent(cells,-2),depth(cells),walk{home};
        std::vector<unsigned char> chosen(cells,0);
        parent[home]=-1;chosen[home]=1;depth[home]=0;
        for(std::size_t i=0;i<walk.size();++i) {
            int u=walk[i];if(depth[u]>=deadline)continue;
            for(int v:edge[u]) if(ours(v) && parent[v]==-2) {
                parent[v]=u;depth[v]=depth[u]+1;walk.push_back(v);
            }
        }
        auto shortage = [&](Army delivered) {
            Army need=0;
            for(const auto &t:threats) {
                Army force=t.initial;
                for(std::size_t j=1;j<t.corridor.size();++j) {
                    force+=column_growth(t.corridor[j-1],int(j)-1)-1;
                    int c=t.corridor[j];if(force<=0 || c==home)break;
                    Army guard=ours(c) && chosen[c] ? 1 :
                        (ours(c)?troops(c):defense(c))+anticipated_growth(c,int(j)-1);
                    force+=memory[c].owner==t.player?guard:-guard;
                    if(force<=0)break;
                }
                need=std::max(need,force-troops(home)-delivered-
                    anticipated_growth(home,int(t.corridor.size())-2));
            }
            return need;
        };
        Army delivered=0, before=shortage(0), remaining=before;
        int commands=0;
        std::vector<int> branch;
        while(commands<deadline && remaining>0) {
            int best_branch=-1,best_edges=0;Army best_gain=0,best_shortage=remaining;
            double best_density=0;
            for(int c:walk) if(!chosen[c] && troops(c)>1) {
                branch.clear();Army gain=0;
                for(int v=c;!chosen[v];v=parent[v]) {branch.push_back(v);gain+=troops(v)-1;}
                if(commands+int(branch.size())>deadline)continue;
                for(int v:branch)chosen[v]=1;
                Army after=shortage(delivered+gain);
                for(int v:branch)chosen[v]=0;
                double density=double(remaining-after)/branch.size();
                if(density>best_density) {
                    best_density=density;best_branch=c;best_edges=int(branch.size());
                    best_gain=gain;best_shortage=after;
                }
            }
            if(best_branch<0)break;
            for(int v=best_branch;!chosen[v];v=parent[v])chosen[v]=1;
            commands+=best_edges;delivered+=best_gain;remaining=best_shortage;
        }
        if(!commands || remaining>=before)return;
        double utility=3.*(before-remaining)/commands+2.*(before-remaining)/(1+deadline-commands);
        if(remaining==0 && deadline-commands<=2)utility+=2000;
        for(int c:walk) if(c!=home && chosen[c] && troops(c)>1) {
            bool leaf=true;
            for(int v:edge[c]) if(v>=0 && chosen[v] && parent[v]==c)leaf=false;
            if(leaf)bid({c,parent[c],false},utility,home,Offer::RESCUE);
        }
    }
    void insurance_offers() {
        Army needed = reserve_guard - troops(home);
        if (!insurance || needed <= 0 || home_pressure > troops(home)) return;
        for (int source : mobile) if (source != home && owned_home_distance[source] <= 4) {
            int length = owned_home_distance[source];
            for (int next : edge[source]) if (ours(next) && owned_home_distance[next]+1 == length) {
                for (bool half : {false,true}) {
                    Move move{source,next,half};
                    Army gain = std::min(needed,sent(move)-length+1);
                    if (gain<=0) continue;
                    double utility = length == 1 ? 0 : insurance * double(gain) / length;
                    bid(move,utility,home,Offer::RESCUE);
                }
            }
        }
    }

    double loss_price(int c, Army guard) const {
        return double(guard) * (memory[c].owner >= 0 && memory[c].owner != seat ? enemy_price : 1.);
    }
    unsigned compatible_formations(int c) const {
        unsigned &bits=compatible_cache[c];
        if(!bits) {
            bits=1u<<16;
            for(std::size_t i=0;i<formations.size();++i)
                if(separated(c,formations[i].second))bits|=1u<<i;
        }
        return bits&511u;
    }
    double frontier_reward(int c) const {
        double &cached=frontier_cache[c];
        if(!std::isnan(cached))return cached;
            int exits = 0; double closed = 0;
            for (int v : edge[c]) if (!obstructed(v)) {
                if (!ours(v) && defense(v) <= 1) ++exits;
                if (ours(v)) {
                    bool alternative = false;
                    for (int u : edge[v]) if (u != c && !obstructed(u) && !ours(u) && defense(u) <= 1)
                        alternative = true;
                    if (!alternative) closed += .75;
                }
            }
        cached=frontier*((exits?1+.25*std::min(2,exits-1):0)-closed);
        return cached;
    }
    // Production becomes useful when it can join a supply road or a fighting
    // group. Mark owned shortest paths to productive hubs and current fronts;
    // surplus land away from them has additional collection commands to pay.
    void prepare_delivery() {
        std::vector<int> parent(cells,-2), road;
        std::vector<unsigned char> marked(cells,0);
        std::vector<int> walk{home}; parent[home]=-1;
        for(std::size_t i=0;i<walk.size();++i)
            for(int v:edge[walk[i]]) if(ours(v) && parent[v]==-2) {
                parent[v]=walk[i];walk.push_back(v);
            }
        auto mark=[&](int c) {
            if(!marked[c]) { marked[c]=1;road.push_back(c); }
            if(parent[c]==-2)return;
            for(int v=parent[c];v>=0 && !marked[v];v=parent[v]) {
                marked[v]=1;road.push_back(v);
            }
        };
        mark(home);
        for(int c:owned_cities)mark(c);
        int bodies=0;
        for(int c:mobile)if(c!=home && troops(c)>=std::max(12.,desired_force*.35)) {
            mark(c);if(++bodies==2)break;
        }
        // A visible contact and the approach to a known general are actual
        // destinations; unseen geography never supplies hidden enemy data.
        for(int c:owned)if(enemy_near[c]>1)mark(c);
        if(!objectives.empty()) {
            int front=home;
            for(int c:owned)if(king_distance[c]<king_distance[front] ||
                (king_distance[c]==king_distance[front] && home_distance[c]<home_distance[front]))front=c;
            mark(front);
        }
        distances(road,delivery_distance);
    }
    double delivered_fraction(int c) const {
        if(!delivered_income || readiness<=0)return 1.;
        return 1./(1.+.3*readiness*std::min(30,delivery_distance[c]));
    }
    double tile_reward(int c, Army remaining, int step, int previous) const {
        const Memory &m = memory[c];
        bool opponent = m.owner >= 0 && m.owner != seat;
        double battle = opponent ? combat * defense(c) / (1 + .05 * step) : 0;
        if (m.building == 2 && opponent) return 600 + .2 * view->total_army[m.owner];
        if (m.building == 1) {
            double holding = horizon;
            if (enemy_near[c] > remaining + 4) holding = 3;
            else if (contact_distance[c] < 6 && remaining < 15) holding = std::min(holding, 12.);
            if (tenure) for (int d=1;d<=6;++d) if (hostile_reach[c][d]>remaining+d/2+4)
                holding=std::min(holding,2.+d*2.);
            double own_holding=liquidity_horizon?std::min(holding,economic_horizon):holding;
            double earnings = .5 * own_holding * economy_weight + (opponent ? .5 * (tenure ? holding : horizon) : 0);
            return earnings / (1 + step * .012) + battle;
        }
        double land_earnings = horizon * .02;
        if (pulse) land_earnings = (now+int(horizon))/50 - (now+step-1)/50;
        if(occupation && opponent) {
            int holding=int(horizon);
            for(int d=1;d<=6;++d)if(hostile_reach[c][d]>remaining+d/2+4)holding=std::min(holding,d*2+2);
            double actual=std::max(0,(now+step+holding)/50-(now+step-1)/50);
            land_earnings=(1-occupation)*land_earnings+occupation*std::min(land_earnings,actual);
        }
        double own_earnings=land_earnings;
        double value = land_earnings * (opponent ? 1 + economy_weight : economy_weight);
        if(liquidity_horizon && capital_disclosed) {
            double earned=pulse?(now+int(economic_horizon))/50-(now+step-1)/50:economic_horizon*.02;
            earned=std::clamp(earned,0.,land_earnings);
            value=earned*economy_weight+(opponent?land_earnings:0);
            own_earnings=earned;
        }
        if(delivered_income)value-=own_earnings*economy_weight*(1-delivered_fraction(c));
        if(frontier && now<200)value+=frontier_reward(c);
        int fresh = unknown[c];
        if (previous >= 0) fresh = std::min(fresh, 3);
        bool unknown_king = objectives.empty();
        double contact = contact_distance[c] < 10 ? 1.8 : .65;
        double outward = home_distance[c] > 5 ? 1. : .4;
        value += recon * fresh * (unknown_king ? .6 : .15) * contact * outward;
        return value + battle;
    }
    void local_offers() {
        if (!local) return;
        // Small frontier garrisons can perform useful work even when they do
        // not enter the eight expensive route searches. Charge the same army
        // and command costs, and retain the full first-move safety checks.
        for (int c : mobile) for (int v : edge[c]) if (!obstructed(v) && !ours(v)) {
            for (bool half : {false, true}) {
                Move m{c,v,half};
                Army force = sent(m) - defense(v);
                if (force <= 0 || !executable(m)) continue;
                double reward = tile_reward(v,force,1,c) +
                    mission_value(v,force,formation_bits[c],c,troops(c)-sent(m));
                if (fork && half && now<200 && troops(c)-sent(m)>=2) {
                    int outlets=0;
                    for(int u:edge[c]) if(u!=v && !obstructed(u) && !ours(u) &&
                        troops(c)-sent(m)-1>defense(u)) ++outlets;
                    reward+=fork*std::min(2,outlets)*horizon*.02;
                }
                double value = local * reward / (1 + command_setup +
                    loss_price(v,defense(v)) * troop_price / std::max(1.,production));
                if (enemy_near[v] > force) value -= std::min(8., double(enemy_near[v]-force)*.035);
                bid(m,value,v,memory[v].building == 2 ? Offer::FINISH :
                    memory[v].building == 1 ? Offer::INVEST : memory[v].owner >= 0 ? Offer::RAID : Offer::EXPLORE);
            }
        }
    }
    // Preserve three different simple routes per tile: the cheapest, the
    // strongest after travel, and the highest yielding. Every route owns its
    // ancestor chain, so a friendly cell's army can be collected only once.
    void resource_routes(int source, bool half) {
        Army starting = half ? troops(source) / 2 : troops(source) - 1;
        if (starting < 1) return;
        double risk[4];
        for (int d = 0; d < 4; ++d) {
            Move m{source, edge[source][d], half};
            risk[d] = executable(m) ? exposure(m) : 1e30;
        }
        auto &labels = route_labels; labels.clear();
        auto &chosen = route_chosen;
        std::fill(chosen.begin(), chosen.end(), std::array<int,3>{-1,-1,-1});
        auto &open = route_open; open.clear();
        auto push = [&](double cost, int id) {
            open.emplace_back(cost, id);
            std::push_heap(open.begin(), open.end(), std::greater<RouteNode>());
        };
        labels.push_back({source,-1,-1,0,starting+1,0,0,0,formation_bits[source],1ULL<<(source&63)});
        chosen[source] = {0,0,0}; push(0,0);
        double replacement = std::max(1., production) / troop_price;
        auto quality = [&](const RouteLabel &a, int criterion) {
            if (criterion == 0) return -a.price;
            if (criterion == 1) return double(a.force) - std::max(1., production) * a.price * .5;
            return a.reward / (.8 + a.price) + .02 * a.force;
        };
        int expanded = 0;
        int budget = half ? std::min(route_budget, half_budget) : route_budget;
        while (!open.empty() && expanded < budget && labels.size() < 4090) {
            if ((expanded & 63) == 0 && search_expired()) break;
            int id = open.front().second;
            std::pop_heap(open.begin(), open.end(), std::greater<RouteNode>()); open.pop_back();
            const RouteLabel current = labels[id];
            const auto &slots = chosen[current.cell];
            if (slots[0] != id && slots[1] != id && slots[2] != id) continue;
            if (current.steps >= 28) continue;
            ++expanded;
            for (int d = 0; d < 4; ++d) {
                int v = edge[current.cell][d];
                if (obstructed(v) || v == source || (!home_transit && v == home && source != home)) continue;
                int first = current.steps ? current.first : d;
                if (risk[first] >= 1e29) continue;
                bool repeated = false;
                if(current.visited & (1ULL<<(v&63)))
                    for (int p = id; p >= 0; p = labels[p].parent) if (labels[p].cell == v) { repeated = true; break; }
                if (repeated) continue;
                int steps = current.steps + 1;
                Army guard = ours(v) ? 0 : defense(v) + anticipated_growth(v,steps-1);
                Army force = current.force + (income ? column_growth(current.cell,current.steps) : 0) - 1 +
                    (ours(v) ? troops(v)+(income ? anticipated_growth(v,steps-1) : 0) : -guard);
                if (force < 1) continue;
                bool king = memory[v].building == 2 && memory[v].owner >= 0;
                double blocking = interception_value(v, force, steps);
                if (preserve && troops(source) >= 12 && home_distance[source] >= 3 &&
                    force < 12 && !king && blocking <= 0) continue;
                double loss = loss_price(v, guard);
                RouteLabel next{v,id,first,steps,force,current.price + 1 + loss / replacement,
                    current.reward + (ours(v) ? 0 : tile_reward(v,force,steps,current.cell)),current.losses + loss,
                    current.consumed | formation_bits[v],current.visited | (1ULL<<(v&63))};
                int replacement_mask = 0;
                for (int k = 0; k < 3; ++k) if (chosen[v][k] < 0 || quality(next,k) > quality(labels[chosen[v][k]],k) + 1e-8)
                    replacement_mask |= 1 << k;
                if (!replacement_mask) continue;
                int index = int(labels.size()); labels.push_back(next);
                for (int k = 0; k < 3; ++k) if (replacement_mask & (1 << k)) chosen[v][k] = index;
                push(next.price,index);
                double campaign_value = mission_value(v, force, next.consumed, source, troops(source) - starting);
                if (ours(v) && campaign_value <= 0 && blocking <= 0) continue;
                // Enemy capture requirements do not apply to friendly reinforcements.
                if (king && !ours(v) && (force < steps / 2 + 3 || (steps == 1 && starting <= king_guard(v)))) continue;
                double rate = (next.reward + campaign_value) / (command_setup + steps + next.losses / replacement) + blocking;
                if (starting >= 20 && force >= 8 && objectives.empty() && unknown[v] && contact_distance[v] <= 5)
                    rate += recon * (2 + unknown[v] * .6) / (1 + steps * .25);
                if (enemy_near[v] > force && !king) rate -= std::min(8., double(enemy_near[v] - force) * .035);
                Move m{source,edge[source][first],half};
                double value = rate - risk[first] + (v == intent && source == column ? .25 : 0);
                if (value > best.utility) {
                    choose(m,value,v,king ? Offer::FINISH : blocking > 0 ? Offer::RESCUE : ours(v) ? Offer::SUPPLY :
                        memory[v].building == 1 ? Offer::INVEST : memory[v].owner >= 0 ? Offer::RAID : Offer::EXPLORE);
                    if (continuity) {
                        for (int p = index; p >= 0; p = labels[p].parent) best.path.push_back(labels[p].cell);
                        std::reverse(best.path.begin(), best.path.end());
                    }
                }
            }
        }
    }
    // Fund an identified investment before asking whether one existing stack
    // can afford it. Greedily add the densest new branch to a shared delivery
    // tree. Every selected cell and edge is charged once; no tree knapsack or
    // inherited policy planner is used.
    void investment_offers() {
        if (investment <= 0 || now < 40) return;
        std::vector<std::pair<double, int>> projects;
        for (int c = 0; c < cells; ++c) if (!ours(c) && !obstructed(c) &&
            (memory[c].building == 1 || (memory[c].building == 2 && memory[c].owner >= 0))) {
            int nearest = far_away;
            for (int u : owned) nearest = std::min(nearest,
                std::abs(u % width - c % width) + std::abs(u / width - c / width));
            double prize = memory[c].building == 2 ? 600 : horizon * .5;
            projects.emplace_back(-prize / (nearest + 1 + defense(c) / std::max(1., production)), c);
        }
        std::sort(projects.begin(), projects.end());
        if (projects.size() > 4) projects.resize(4);
        std::vector<int> to_goal(cells), goal_parent(cells), parent(cells), depth(cells), walk;
        std::vector<unsigned char> selected(cells);
        for (auto project : projects) {
            if (search_expired()) return;
            int goal = project.second;
            bool assault = siege && memory[goal].building == 2 && memory[goal].owner >= 0;
            std::fill(to_goal.begin(), to_goal.end(), far_away);
            std::fill(goal_parent.begin(), goal_parent.end(), -1);
            walk.assign(1, goal); to_goal[goal] = 0;
            std::vector<int> roots;
            bool committed = assembly_goal == goal && ours(assembly_root);
            if (assault) {
                using Node = std::pair<double,int>;
                std::priority_queue<Node,std::vector<Node>,std::greater<Node>> open;
                std::vector<double> cost(cells,1e30); cost[goal]=0; open.emplace(0,goal);
                while (!open.empty()) {
                    auto [value,u]=open.top();open.pop(); if (value!=cost[u]) continue;
                    if (committed && u==assembly_root) break;
                    if (!committed && ours(u)) {roots.push_back(u);if (roots.size()>=3) break;continue;}
                    if (to_goal[u]>=16) continue;
                    for (int v:edge[u]) if (!obstructed(v)) {
                        double next=value+1+(ours(u)?0:defense(u)*troop_price/std::max(1.,production));
                        if (next<cost[v]) {
                            cost[v]=next;to_goal[v]=to_goal[u]+1;goal_parent[v]=u;open.emplace(next,v);
                        }
                    }
                }
            }
            for (std::size_t q = 0; !assault && q < walk.size() && (committed ? to_goal[assembly_root] == far_away : roots.size() < 2); ++q) {
                int u = walk[q];
                if (ours(u) && !committed) { roots.push_back(u); continue; }
                if (to_goal[u] >= (committed ? 12 : 6)) continue;
                for (int v : edge[u]) if (!obstructed(v) && to_goal[v] == far_away &&
                    (ours(v) || defense(v) <= 3)) {
                    to_goal[v] = to_goal[u] + 1; goal_parent[v] = u; walk.push_back(v);
                }
            }
            if (committed && to_goal[assembly_root] < far_away) roots = {assembly_root};
            for (int root : roots) {
                Army reinforcement = 0;
                if (assault) for (int v : edge[goal]) if (v >= 0 && view->visible[v] &&
                    memory[v].owner == memory[goal].owner) {
                    bool on_path = false;
                    for (int c = root; c >= 0 && c != goal; c = goal_parent[c]) on_path |= c == v;
                    if (!on_path) reinforcement = std::max(reinforcement,std::max<Army>(0,troops(v)-1));
                }
                Army price = defense(goal) + reinforcement + to_goal[root];
                double economic_loss = loss_price(goal, defense(goal));
                for (int c = goal_parent[root]; c >= 0 && c != goal; c = goal_parent[c])
                    if (!ours(c)) { price += defense(c); economic_loss += loss_price(c, defense(c)); }
                Army guard = root == home && !enemies.empty() ? home_pressure + 8 : 1;
                Army needed = price + 12 + guard + (assault ? 16 : 0);
                if (troops(root) >= needed) {
                    double prize = tile_reward(goal, troops(root) - price - guard, to_goal[root], root);
                    unsigned consumed = formation_bits[root];
                    for (int c = goal_parent[root]; c >= 0; c = goal_parent[c]) consumed |= formation_bits[c];
                    prize += mission_value(goal, troops(root) - price - guard, consumed);
                    double value = investment * prize / (.8 + to_goal[root] + troop_price * economic_loss / std::max(1., production));
                    bid({root, goal_parent[root], false}, value, goal, Offer::INVEST);
                    continue;
                }
                std::fill(parent.begin(), parent.end(), -2);
                std::fill(selected.begin(), selected.end(), 0);
                walk.assign(1, root); parent[root] = -1; depth[root] = 0; selected[root] = 1;
                for (std::size_t q = 0; q < walk.size(); ++q) {
                    int u = walk[q]; if (depth[u] >= (assault ? 10 : 7)) continue;
                    for (int v : edge[u]) if (ours(v) && parent[v] == -2 && !(v == home && root != home)) {
                        parent[v] = u; depth[v] = depth[u] + 1; walk.push_back(v);
                    }
                }
                Army funded = troops(root); int commands = 0;
                int command_limit = assault ? 24 : 10;
                for (int round = 0; round < command_limit && funded < needed; ++round) {
                    int branch = -1, edges = 0; Army gain = 0; double density = 0;
                    for (int c : walk) if (!selected[c] && troops(c) > 1) {
                        Army available = 0; int length = 0;
                        for (int v = c; !selected[v]; v = parent[v]) {
                            available += std::max<Army>(0, troops(v) - 1); ++length;
                        }
                        if (length + commands > command_limit) continue;
                        double value = double(std::min(available, needed - funded)) / length;
                        if (value > density) { density = value; branch = c; edges = length; gain = available; }
                    }
                    if (branch < 0) break;
                    for (int v = branch; !selected[v]; v = parent[v]) selected[v] = 1;
                    commands += edges; funded += gain;
                }
                if (funded < needed || !commands) continue;
                Army residual = funded - price - guard;
                double prize = tile_reward(goal, residual, commands + to_goal[root], root);
                unsigned consumed = 0;
                for (int c : walk) if (selected[c]) consumed |= formation_bits[c];
                for (int c = goal_parent[root]; c >= 0; c = goal_parent[c]) consumed |= formation_bits[c];
                prize += mission_value(goal, residual, consumed);
                double value = investment * prize / (.8 + commands + to_goal[root] + troop_price * economic_loss / std::max(1., production));
                // Execute a funded leaf first; internal relays wait until their
                // incoming branch has arrived. Recompute from actual troops.
                for (int c : walk) if (c != root && selected[c] && troops(c) > 1) {
                    bool leaf = true;
                    for (int v : edge[c]) if (v >= 0 && parent[v] == c && selected[v]) leaf = false;
                    if (leaf) bid({c, parent[c], false}, value, goal, Offer::INVEST, root);
                }
            }
        }
    }
    void supply_offers() {
        if (now < 45 || !mobilize || mobile.empty()) return;
        if (column < 0 || column == home) {
            double score = -1;
            for (int c : mobile) if (c != home && troops(c) >= 6) {
                double contact = contact_distance[c] < far_away ? contact_distance[c] : std::max(0, 20 - home_distance[c]);
                double value = std::sqrt(double(troops(c))) / (1 + .13 * contact);
                if (value > score) { score = value; column = c; }
            }
        }
        if (column < 0 || column == home) return;
        Army enemy_mass = 0, visible_mass = 0, spare = 0;
        for (int p = 0; p < view->players; ++p) if (p != seat && view->alive[p])
            enemy_mass = std::max(enemy_mass, std::max<Army>(0, view->total_army[p] - view->total_land[p]));
        for (int e : enemies) if (std::abs(e % width - column % width) + std::abs(e / width - column / width) <= 10)
            visible_mass = std::max(visible_mass, troops(e));
        for (int c : owned) if (c != column && c != home) spare += std::max<Army>(0, troops(c) - 1);
        Army wanted = std::max<Army>(20, std::max(visible_mass + 12, enemy_mass * 2 / 5));
        if (troops(column) >= wanted || spare * 3 < troops(column)) return;
        std::vector<int> distance;
        distances({column}, distance, true);
        for (int donor : mobile) {
            if (donor == column || distance[donor] > 8 || distance[donor] < 1 || troops(donor) < 8) continue;
            for (int next : edge[donor]) if (ours(next) && distance[next] + 1 == distance[donor]) {
                Move m{donor, next, donor == home};
                Army gain = std::min(sent(m) - distance[donor] + 1, wanted - troops(column));
                if (gain < 4 || gain < 3 * distance[donor]) continue;
                double utility = mobilize * std::sqrt(double(gain)) * 1.6 / (1 + .45 * distance[donor]);
                bid(m, utility, column, Offer::SUPPLY);
            }
        }
    }
    std::vector<int> goal_rally_roots(const std::vector<int>& ready,Army target) {
        // Spatial hints only. Every proposed root must still pass the existing
        // funded-tree construction and the same first-command safety checks.
        std::vector<std::pair<double,int>> destinations;
        for(int c=0;c<cells;++c)if(!ours(c) && !obstructed(c)){
            double prize=tile_reward(c,target,1,-1)+hunt*400*capital_information[c];
            if(prize>0)destinations.emplace_back(-prize,c);
        }
        std::sort(destinations.begin(),destinations.end());
        std::vector<int> goals;
        for(auto [negative,c]:destinations){
            bool close=false;
            for(int other:goals)if(std::abs(cell_x[c]-cell_x[other])+std::abs(cell_y[c]-cell_y[other])<4)close=true;
            if(close)continue;
            goals.push_back(c);if(goals.size()==6)break;
        }
        std::vector<double> quality(cells,0),cost(cells);
        std::vector<int> steps(cells);
        using Node=std::pair<double,int>;
        for(int goal:goals){
            if(search_expired())break;
            double prize=tile_reward(goal,target,1,-1)+hunt*400*capital_information[goal];
            std::fill(cost.begin(),cost.end(),1e30);std::fill(steps.begin(),steps.end(),far_away);
            std::priority_queue<Node,std::vector<Node>,std::greater<Node>> open;
            cost[goal]=0;steps[goal]=0;open.emplace(0,goal);int expanded=0;
            while(!open.empty()){
                if((expanded++&31)==0 && search_expired())break;
                auto [value,c]=open.top();open.pop();if(value!=cost[c])continue;
                if(steps[c]>=24)continue;
                double toll=1+(ours(c)?0:loss_price(c,defense(c))*troop_price/std::max(1.,production));
                for(int v:edge[c])if(!obstructed(v) && value+toll<cost[v]){
                    cost[v]=value+toll;steps[v]=steps[c]+1;open.emplace(cost[v],v);
                }
            }
            for(int c:mobile)if(cost[c]<1e29)quality[c]=std::max(quality[c],prize/(3+cost[c]));
        }
        std::vector<std::pair<double,int>> candidates;
        for(int c:mobile){
            if((!forward_rally && home_distance[c]>12) || (contact_distance[c]<=2 && enemy_near[c]>=troops(c)))continue;
            if(!ready.empty() && !separated(c,ready.front()))continue;
            if(quality[c]>0)candidates.emplace_back(-quality[c],c);
        }
        std::sort(candidates.begin(),candidates.end());
        std::vector<int> roots;
        for(auto [score,c]:candidates){roots.push_back(c);if(roots.size()==3)break;}
        return roots;
    }
    void rally_offers() {
        if (!rally || production < 1.05 || deadline <= 5) return;
        Army spare = std::max<Army>(0, view->total_army[seat] - view->total_land[seat]);
        Army target = rally_size(spare);
        if (spare < target + 6) return;
        if (!ours(rally_root) || troops(rally_root) >= rally_target || now - rally_started >= 32)
            rally_root = -1;
        std::vector<int> ready;
        // A formed corps stays formed while crossing the capital's vicinity.
        // Hysteresis prevents a growing target from repeatedly reassembling it.
        for (int c : mobile) if (troops(c) >= std::max<Army>(18, target * 3 / 4)) ready.push_back(c);
        std::vector<int> roots;
        if (rally_root >= 0) roots.push_back(rally_root);
        else {
            bool two_fronts = false;
            for (std::size_t i = 1; i < ready.size(); ++i) two_fronts |= separated(ready[0], ready[i]);
            if (two_fronts || (!ready.empty() && readiness < .3)) return;
            if(goal_rally)roots=goal_rally_roots(ready,target);
            for (int c : mobile) {
                // Mature supply networks can be far from our general. Rank
                // their funded projects in the same command auction.
                if ((!forward_rally && home_distance[c] > 12) ||
                    (contact_distance[c] <= 2 && enemy_near[c] >= troops(c))) continue;
                if (!ready.empty() && !separated(c, ready.front())) continue;
                if(std::find(roots.begin(),roots.end(),c)==roots.end())roots.push_back(c);
                if (roots.size() == 6) break;
            }
        }
        std::vector<int> parent(cells), depth(cells), walk;
        std::vector<unsigned char> selected(cells);
        for (int root : roots) {
            if (search_expired()) return;
            Army wanted = root == rally_root ? rally_target : target;
            if (troops(root) >= wanted) continue;
            std::fill(parent.begin(), parent.end(), -2);
            std::fill(selected.begin(), selected.end(), 0);
            walk.assign(1, root); parent[root] = -1; depth[root] = 0; selected[root] = 1;
            for (std::size_t i = 0; i < walk.size(); ++i) {
                int u = walk[i]; if (depth[u] >= 7) continue;
                for (int v : edge[u]) if (ours(v) && parent[v] == -2 &&
                    enemy_near[v] < troops(v) + wanted &&
                    std::find(ready.begin(), ready.end(), v) == ready.end()) {
                    parent[v] = u; depth[v] = depth[u] + 1; walk.push_back(v);
                }
            }
            Army funded = troops(root); int commands = 0;
            for (int round = 0; round < 18 && funded < wanted; ++round) {
                int branch = -1, edges = 0; Army gain = 0; double density = 0;
                for (int c : walk) if (!selected[c] && troops(c) > 1) {
                    Army available = 0; int length = 0;
                    for (int v = c; !selected[v]; v = parent[v]) {
                        available += troops(v) - 1; ++length;
                    }
                    if (commands + length > 18) continue;
                    double score = double(std::min(available, wanted - funded)) / length;
                    if (score > density) { density = score; branch = c; edges = length; gain = available; }
                }
                if (branch < 0) break;
                for (int c = branch; !selected[c]; c = parent[c]) selected[c] = 1;
                commands += edges; funded += gain;
            }
            if (funded < wanted || !commands) continue;
            unsigned consumed = 0;
            for (int c : walk) if (selected[c]) consumed |= formation_bits[c];
            double activation = rally * 40 * formation_scale * std::min(1., double(funded - troops(root)) / wanted);
            double value = (activation + mission_value(root, funded, consumed)) / (.8 + commands);
            for (int c : walk) if (c != root && selected[c] && troops(c) > 1) {
                bool leaf = true;
                for (int v : edge[c]) if (v >= 0 && parent[v] == c && selected[v]) leaf = false;
                if (!leaf) continue;
                double before = best.utility;
                bid({c, parent[c], false}, value, root, Offer::SUPPLY);
                if (best.utility > before) best.rally = root;
            }
        }
    }
    void continue_campaigns() {
        if (!continuity) return;
        for (Campaign &plan : campaigns) {
            auto &path = plan.path;
            bool returning = (return_plan || screen>=2) && plan.purpose == Offer::RESCUE && !path.empty() && path.back() == home;
            if (returning) {
                bool viable = path.size() > 1 && now - plan.started < 24;
                for (int c : path) viable &= ours(c);
                Move move = viable ? Move{path[0],path[1],false} : Move{};
                if (!viable || !executable(move)) { path.clear(); continue; }
                // The purpose of an urgent return is delivered protection.
                // Temporary coverage of one predicted invasion corridor must
                // not reverse the delivery halfway and start it again.
                double previous = best.utility;
                double commitment = screen>=2 && home_pressure<=troops(home) ? 0 : 1500;
                if (revalidate && commitment>0) {
                    Army delivered=troops(path[0]);
                    for(std::size_t j=1;j<path.size();++j) delivered+=troops(path[j])-1;
                    bool covered=true;
                    for(const auto &t:threats) {
                        Army force=t.initial;
                        for(std::size_t j=1;j<t.corridor.size();++j) {
                            force+=column_growth(t.corridor[j-1],int(j)-1)-1;
                            int c=t.corridor[j];if(force<=0 || c==home)break;
                            bool moved=ours(c) && std::find(path.begin(),path.end()-1,c)!=path.end()-1;
                            Army guard=moved?1:(ours(c)?troops(c):defense(c))+anticipated_growth(c,int(j)-1);
                            force+=memory[c].owner==t.player?guard:-guard;
                            if(force<=0)break;
                        }
                        int length=int(t.corridor.size())-1;
                        if (force>delivered+anticipated_growth(home,length-1) ||
                            (force>troops(home)+anticipated_growth(home,length-1) && int(path.size())-1>length)) covered=false;
                    }
                    if (!covered) {path.clear();continue;}
                }
                bid(move,commitment,home,Offer::RESCUE);
                if (best.utility > previous) best.path = path;
                continue;
            }
            if (path.size() < 2 || now - plan.started >= continuity ||
                plan.kings != int(objectives.size()) || !ours(path[0]) ||
                (plan.purpose != Offer::SUPPLY && ours(path.back()))) {
                path.clear(); continue;
            }
            Move move{path[0], path[1], false};
            if (!executable(move)) { path.clear(); continue; }
            Army force = troops(path[0]); double reward = 0, losses = 0;
            unsigned consumed = formation_bits[path[0]];
            bool viable = true;
            for (std::size_t i = 1; i < path.size(); ++i) {
                int c = path[i];
                if (obstructed(c)) { viable = false; break; }
                Army guard = ours(c) ? 0 : defense(c)+anticipated_growth(c,int(i)-1);
                force += (income ? column_growth(path[i-1],int(i)-1) : 0) +
                    (ours(c) ? troops(c)+(income ? anticipated_growth(c,int(i)-1) : 0) : -guard) - 1;
                if (force < 1) { viable = false; break; }
                if (preserve && troops(path[0]) >= 12 && home_distance[path[0]] >= 3 && force < 12 &&
                    plan.purpose != Offer::FINISH && plan.purpose != Offer::RESCUE) {
                    viable = false; break;
                }
                consumed |= formation_bits[c]; losses += loss_price(c, guard);
                if (!ours(c)) reward += tile_reward(c, force, int(i), path[i-1]);
                if (i == 1 && enemy_near[c] > force + 8) {
                    // The captured tile's old defenders are already paid for.
                    for (int e : edge[c]) if (e >= 0 && e != path[0] && view->visible[e] &&
                        memory[e].owner >= 0 && memory[e].owner != seat && troops(e) > force + 8)
                        viable = false;
                }
            }
            if (!viable) { path.clear(); continue; }
            reward += mission_value(path.back(), force, consumed);
            double value = reward / (command_setup + path.size() - 1 + losses * troop_price / std::max(1., production));
            if (enemy_near[path.back()] > force && plan.purpose != Offer::FINISH)
                value -= std::min(8., double(enemy_near[path.back()] - force) * .035);
            double before = best.utility;
            bid(move, value + .4, path.back(), plan.purpose);
            if (best.utility > before) best.path = path;
        }
    }
    void remember_campaign() {
        if (!continuity || best.path.size() < 2) return;
        int slot = -1;
        for (int i = 0; i < 2; ++i) if (!campaigns[i].path.empty() && campaigns[i].path[0] == best.move.from) slot = i;
        if (slot < 0) {
            for (int i = 0; i < 2; ++i) if (campaigns[i].path.empty()) { slot = i; break; }
        }
        if (slot < 0) slot = separated(campaigns[0].path[0], best.move.from) ? 1 : 0;
        Campaign &plan = campaigns[slot];
        bool unchanged = plan.path == best.path;
        if (!unchanged) {
            plan.path = best.path; plan.started = now;
            plan.kings = int(objectives.size()); plan.purpose = best.purpose;
        }
        last_campaign = slot;
    }
    std::int64_t act(const ArenaObservation *observation) {
        if (!observation || observation->width * observation->height > maximum_cells ||
            observation->general_pos < 0 || !observation->alive[seat]) return 0;
        // Leave headroom inside the native 10 ms limit. Expensive searches
        // keep their best legal offer and stop adding work after 6 ms.
        search_until=std::chrono::steady_clock::now()+std::chrono::milliseconds(6);
        observe(observation); trace_threats();
        choose(Move{}, -exposure(Move{}), -1, Offer::EXPLORE);
        // Batch the first dispatch before the land-growth pulse. Visible
        // opponents immediately return control to ordinary tactical decisions.
        if(now<opening_delay && enemies.empty() && home_pressure==0) {
            last_tick=now;last_move=Move{};return 0;
        }
        immediate_moves();
        relief_offers();
        defense_tree();
        insurance_offers();
        continue_campaigns();
        // Re-auction command allocation at least every four ticks; between
        // auctions both plans are checked against the current legal view.
        bool continuing = continuity && !best.path.empty() && now % 4 && home_pressure <= troops(home);
        if (!continuing && best.utility < 50000 && !search_expired()) {
            local_offers();
            int count = 0;
            for (int c : mobile) {
                if (search_expired()) break;
                if (++count > 8) break;
                resource_routes(c, false);
                if (memory[c].building > 0 || troops(c) >= 24) resource_routes(c, true);
            }
            if (!search_expired()) investment_offers();
            if (!search_expired()) supply_offers();
            if (!search_expired()) rally_offers();
        }
        if (best.move.from < 0) { last_tick = now; last_move = Move{}; return 0; }
        if (best.rally >= 0 && rally_root != best.rally) {
            rally_root = best.rally; rally_started = now;
            Army spare = std::max<Army>(0, view->total_army[seat] - view->total_land[seat]);
            rally_target = rally_size(spare);
        }
        if (best.assembly >= 0) { assembly_goal = best.objective; assembly_root = best.assembly; }
        else if (best.move.from == assembly_root || best.purpose == Offer::RESCUE)
            assembly_goal = assembly_root = -1;
        if (best.purpose != Offer::SUPPLY && best.purpose != Offer::RESCUE) {
            intent = best.objective;
            if (troops(best.move.from) >= 12) column = best.move.from;
        }
        remember_campaign();
        last_army = troops(best.move.from); last_move = best.move; last_tick = now;
        return arena_move(best.move.from, direction(best.move), best.move.split);
    }
};
} // namespace aegis

extern "C" void *arena_create(int player, std::uint64_t seed, const char *params_json) {
    return new aegis::Agent(player, seed, params_json);
}
extern "C" std::int64_t arena_act(void *instance, const ArenaObservation *observation) {
    return static_cast<aegis::Agent *>(instance)->act(observation);
}
extern "C" void arena_destroy(void *instance) { delete static_cast<aegis::Agent *>(instance); }
