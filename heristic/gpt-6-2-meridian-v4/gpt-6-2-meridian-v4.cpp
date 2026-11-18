// Meridian v4 -- a single-threaded, observation-only Generals policy.
// Before first income, sample complete legal expansion schedules. Thereafter
// compare short marches, resource expeditions with shared supply branches,
// and deadline-constrained defense. Each search returns its first command.
#include "arena_policy.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>

#include <vector>
#include <queue>
#include <functional>

namespace meridian {
constexpr int capacity = 4096, unreachable = 10000;
using Clock = std::chrono::steady_clock;
struct Cell {
    int owner = -3, army = 0, kind = 0, seen = -1, firstEnemy = -1;
    bool blocked = false;
};
struct Move {
    int from = -1, to = -1;
    bool half = false;
};
struct Offer {
    Move move;
    float value = 0;
    int objective = -1;
};
struct Target {
    int cell;
    float value;
};
class Player {
    const ArenaObservation *obs = nullptr;
    int seat, n = 0, w = 0, h = 0, home = -1, tick = 0, live = 0;
    std::array<int, 8> oldArmy{}, economicTowers{}, lastChange{};
    int lastGoal = -1, lastFrom = -1, lastTo = -1, reserve = 0;
    int selectedEnemy = -1, strongestEnemy = 0;
    int attackWindowEnd = -1, deficitTurns = 0;
    bool pressAttack = false;
    std::array<Cell, capacity> cells{};
    std::array<std::array<int, 4>, capacity> edges;
    std::array<int, capacity> homeDist, contact, ownDist, kingDist, frontier, queue;
    std::array<int, capacity> distance, pathNext, pathNeed;
    std::array<int, capacity> mark;
    std::array<int, capacity> enemyPower, interest;
    std::array<float, capacity> cellReward;
    std::vector<int> owned, mobile, enemies, kings, structures, frontierSeeds;
    Offer best;
    Clock::time_point deadline;
    bool expired() const { return Clock::now() >= deadline; }
    bool mine(int c) const { return c >= 0 && cells[c].owner == seat; }
    bool hostile(int c) const { return c >= 0 && cells[c].owner >= 0 && !mine(c); }
    int md(int a, int b) const { return std::abs(a % w - b % w) + std::abs(a / w - b / w); }
    int available(int c) const { return std::max(0, cells[c].army - (c == home ? std::max(1, reserve) : 1)); }
    std::array<float, capacity> occupancy{};
    std::array<int, capacity> fogGuard{};
    // The public land total fixes how many enemy tiles are missing from memory.
    // Rank locations by their distance behind the observed front, then soften
    // that boundary to estimate garrisons and capture value in fog.
    void inferTerritory() {
        if (obs->players != 2 || tick < 70 || (tick % 4 != 0 && tick != 70))
            return;
        std::vector<int> seeds, possible;
        int knownEnemyLand = 0;
        for (int c = 0; c < n; ++c) {
            occupancy[c] = 0;
            fogGuard[c] = 0;
            if (hostile(c)) {
                seeds.push_back(c);
                ++knownEnemyLand;
            }
            if (!obs->visible[c] && !hostile(c) && open(c))
                possible.push_back(c);
        }
        if (seeds.empty() && focal < 0)
            return;
        int missing = std::clamp(int(obs->total_land[1 - seat]) - knownEnemyLand, 0, int(possible.size()));
        if (!missing)
            return;
        if (seeds.empty())
            seeds.push_back(focal);
        flood(seeds, distance, [&](int c) { return open(c) && !mine(c); });
        std::vector<float> weights;
        weights.reserve(possible.size());
        for (int c : possible) {
            float back = focal < 0 ? 0.f : .16f * std::min(24, focalDist[c]);
            float age = cells[c].seen < 0 ? 0.f : 10.f / (1.f + .12f * (tick - cells[c].seen));
            float adj = 0;
            for (int v : edges[c])
                if (hostile(v))
                    adj += .45f;
            float cost = std::min(48, distance[c]) + back + age - adj + .2f * std::max(0, 9 - homeDist[c]);
            weights.push_back(cost);
        }
        auto ranked = weights;
        std::nth_element(ranked.begin(), ranked.begin() + missing - 1, ranked.end());
        float border = ranked[missing - 1];
        for (int i = 0; i < int(possible.size()); ++i) {
            int c = possible[i];
            float d = border - weights[i];
            float probability = .5f + .5f * d / (1.5f + std::abs(d));
            if (missing == int(possible.size()))
                probability = 1.f;
            occupancy[c] = probability;
            float average = 1.f + std::min(2.5f, std::max(0.f, d) * .10f) * std::min(1.f, tick / 225.f);
            fogGuard[c] = int(.5f + probability * average);
        }
    }
    bool wall(int c) const { return c < 0 || cells[c].blocked; }
    int guard(int c) const {
        auto &v = cells[c];
        if (obs->players == 2 && !obs->visible[c] && !hostile(c) && !v.kind)
            return fogGuard[c];
        if (!hostile(c) || v.seen == tick)
            return v.army;
        int age = tick - v.seen;
        int income = v.kind ? std::min(age / 2, 45) : std::min(age / 50, 4);
        // An unobserved mobile stack can leave. Fixed production remains
        // relevant when deciding whether a distant expedition can succeed.
        return (v.army <= 4 || age < 3 ? v.army : std::max(2, v.army / (1 + age / 6))) + income;
    }
    int dir(Move m) const {
        for (int d = 0; d < 4; ++d)
            if (edges[m.from][d] == m.to)
                return d;
        return -1;
    }
    bool savingForCity = false;
    void bid(Move m, float value, int target) {
        if (!mine(m.from) || wall(m.to) || cells[m.from].army < 2)
            return;
        // An adversary can hit the source before it leaves (pursuit
        // dependency), even when the proposed destination is friendly.
        if (target != home && value < 9000 && enemyPower[m.from] >= cells[m.from].army - 1) {
            bool pinned = false;
            for (int e : edges[m.from])
                if (hostile(e) && cells[e].seen == tick && cells[e].army >= cells[m.from].army &&
                    (e != m.to || cells[e].army > cells[m.from].army))
                    pinned = true;
            if (pinned)
                value = value * .32f - .55f;
        }
        if (!m.half && target != home && m.from == home && reserve > 1 && value < 9000) {
            int half = cells[home].army / 2;
            if (half > 0 && (mine(m.to) || half > guard(m.to) + 2) && cells[home].army - half >= reserve)
                m.half = true;
        }
        if (m.from == home && reserve > 1) {
            int left = m.half ? (cells[home].army + 1) / 2 : 1;
            if (left < reserve && target != home && !(hostile(m.to) && cells[m.to].kind == 2))
                value -= 12 + std::min(100, reserve - left) * 1.5f;
        }
        if (savingForCity && cells[m.from].kind && cells[m.from].army < 54 && target != home && target >= 0 &&
            cells[target].kind == 0 && !(hostile(m.to) && cells[m.to].kind))
            value -= 3.4f;
        if (obs->players == 2 && tick >= 50 && tick < 155 && reserve < 5 && target >= 0 && target != home &&
            !cells[target].kind) {
            if (cells[m.from].kind && (tick % 50) < 22 && cells[m.from].army < 22 && !hostile(m.to) &&
                contact[m.from] > 3)
                value *= 0.08f;
            if (!mine(m.to) && !cells[m.to].kind && cells[m.from].army <= 3 && guard(m.to) == 0 &&
                enemyPower[m.to] < 2 && tick % 50 < 22)
                value = value * 1.55f + .45f;
        }
        if (obs->players == 2 && !mine(m.to) && cells[m.to].kind == 1) {
            if (!hostile(m.to))
                value *= cityCommitment(m.to);
            else if (enemyPower[m.to] < cells[m.from].army - guard(m.to) - 1)
                value *= 1.55f;
        }
        if (obs->players == 2 && m.from == reliefSource && m.to != reliefNext && value < 150.f &&
            target != home && !(hostile(m.to) && cells[m.to].kind == 2))
            value *= .28f;
        if (m.from == lastTo && m.to == lastFrom && mine(m.to))
            value -= 2.5f;
        if (target >= 0 && target == lastGoal)
            value *= 1.10f;
        if (value > best.value)
            best = {m, value, target};
    }
    void geometry() {
        w = obs->width;
        h = obs->height;
        n = w * h;
        for (int i = 0; i < n; ++i) {
            edges[i] = {i >= w ? i - w : -1, i % w + 1 < w ? i + 1 : -1, i + w < n ? i + w : -1,
                        i % w ? i - 1 : -1};
        }
    }
    template <class Pass>
    int flood(const std::vector<int> &seeds, std::array<int, capacity> &dist, Pass pass) {
        std::fill_n(dist.begin(), n, unreachable);
        int head = 0, tail = 0;
        for (int c : seeds)
            if (c >= 0 && dist[c]) {
                dist[c] = 0;
                queue[tail++] = c;
            }
        while (head < tail) {
            int c = queue[head++];
            for (int v : edges[c])
                if (v >= 0 && dist[v] == unreachable && pass(v)) {
                    dist[v] = dist[c] + 1;
                    queue[tail++] = v;
                }
        }
        return tail;
    }
    bool open(int c) const { return !wall(c) && (cells[c].kind != 1 || cells[c].owner >= 0); }
    void updateDuelPressure() {
        if (obs->players != 2 || tick < 90)
            return;
        const int enemy = 1 - seat;
        const int landGap = int(obs->total_land[enemy] - obs->total_land[seat]);
        const int armyGap = int(obs->total_army[enemy] - obs->total_army[seat]);
        const int cityGap = std::min(10, economicTowers[enemy]) - int(obs->total_cities[seat]);
        // One 50-tick cycle yields one troop per tile and another 25 per city.
        const int incomeGap = landGap + 25 * cityGap;
        if (armyGap + 2 * incomeGap > 30 && (incomeGap > 3 || armyGap > 70))
            deficitTurns = std::min(16, deficitTurns + 1);
        else
            deficitTurns = std::max(0, deficitTurns - 2);
        if (deficitTurns >= 6)
            pressAttack = true;
        else if (deficitTurns == 0)
            pressAttack = false;

        const int ourMobile = int(obs->total_army[seat] - obs->total_land[seat]);
        const int theirMobile = int(obs->total_army[enemy] - obs->total_land[enemy]);
        if (tick <= attackWindowEnd && ourMobile + 10 > .80f * theirMobile)
            pressAttack = true;
    }
    // The exclusive bound records uninterrupted neutral or friendly control.
    // An enemy observed earlier could not have marched through these tiles.
    std::array<int, capacity> enemyExcludedThrough{};
    std::array<int, capacity> earliestLandLimit{};
    int enemyLosses = 0;
    // A large observed departure and its matching arrival supply a bearing.
    // Compare the two geodesic distances for each possible general location.
    struct Heading {
        int source = -1, target = -1, when = 0;
        std::array<signed char, capacity> side{};
        float confidence = 0;
    };
    std::array<Heading, 10> headings{};
    int numHeadings = 0;
    void readHeading() {
        if (obs->players != 2 || tick < 16 || tick > 150 || !kings.empty() ||
            numHeadings == int(headings.size()))
            return;
        int a = -1, b = -1, weight = 3;
        for (int c : enemies) {
            if (cells[c].seen != tick - 1 || !obs->visible[c] || obs->owner[c] != 1 - seat || obs->army[c] > 2)
                continue;
            int departed = cells[c].army - int(obs->army[c]);
            if (departed <= weight)
                continue;
            for (int z : edges[c]) {
                if (z < 0 || !obs->visible[z] || obs->owner[z] != 1 - seat || cells[z].kind == 2)
                    continue;
                int before = hostile(z) ? cells[z].army : -cells[z].army;
                int balance = int(obs->army[z]) - before;
                if (balance <= 2 || std::abs(balance - departed) > 2)
                    continue;
                bool repeated = false;
                for (int i = 0; i < numHeadings; ++i)
                    if (md(c, headings[i].source) <= 3 && tick - headings[i].when < 38)
                        repeated = true;
                if (!repeated) {
                    a = c;
                    b = z;
                    weight = departed;
                }
            }
        }
        if (a < 0)
            return;
        Heading &row = headings[numHeadings++];
        row.source = a;
        row.target = b;
        row.when = tick;
        row.confidence =
            std::clamp((weight + 2.f) / 12.f, .35f, 1.f) * (homeDist[a] >= homeDist[b] ? 1.f : .46f);
        std::array<int, capacity> from, to;
        auto pass = [&](int c) { return open(c); };
        flood({a}, from, pass);
        flood({b}, to, pass);
        for (int c = 0; c < n; ++c)
            row.side[c] = std::clamp(to[c] - from[c], -1, 1);
    }
    // Early enemy ownership near a candidate reinforces independent bearings.
    float originSupport(int c) const {
        float bearing = 0;
        for (int j = 0; j < numHeadings; ++j) {
            const Heading &row = headings[j];
            bearing += row.confidence * row.side[c] / (1.f + .045f * md(c, row.source));
        }
        int earlyEnemy = 0, empty = 0, examined = 0;
        for (int y = std::max(0, c / w - 4); y <= std::min(h - 1, c / w + 4); ++y)
            for (int x = std::max(0, c % w - 4); x <= std::min(w - 1, c % w + 4); ++x) {
                int z = y * w + x;
                if (md(c, z) > 5 || wall(z) || cells[z].seen < 0)
                    continue;
                ++examined;
                if (cells[z].firstEnemy >= 0 && cells[z].firstEnemy < 140)
                    ++earlyEnemy;
                else if ((cells[z].owner < 0 && cells[z].seen >= 80) || mine(z))
                    ++empty;
            }
        float cover = (earlyEnemy - .8f * empty) / (9.f + examined);
        return std::exp(std::clamp(0.5f * bearing + 1.4f * cover, -2.4f, 2.4f));
    }
    void observe() {
        int oldHome = home;
        home = obs->general_pos;
        tick = int(obs->tick);
        readHeading();
        if (oldHome >= 0 && oldHome != home) {
            if (obs->players == 2) {
                // A mutual capture exchanges capitals and halves transferred
                // garrisons. Our former capital is known even across the fog.
                int defending = cells[oldHome].army;
                if (obs->previous && obs->previous->valid && obs->previous->source == oldHome)
                    defending -= int(obs->previous->moved);
                int arrival = 1;
                for (int e : edges[oldHome])
                    if (hostile(e))
                        arrival = std::max(arrival, cells[e].army - 1 - defending);
                for (int c = 0; c < n; ++c)
                    if (!obs->visible[c]) {
                        if (mine(c)) {
                            int g = cells[c].army;
                            if (obs->previous && obs->previous->valid && obs->previous->source == c)
                                g -= int(obs->previous->moved);
                            cells[c].owner = 1 - seat;
                            cells[c].army = std::max(1, (g + 1) / 2);
                            cells[c].firstEnemy = tick;
                        } else if (hostile(c)) {
                            cells[c].owner = -3;
                            cells[c].army = 0;
                        }
                    }
                auto &king = cells[oldHome];
                king.owner = 1 - seat;
                king.kind = 2;
                king.seen = tick - 1;
                king.army = arrival;
                king.firstEnemy = tick;
            } else {
                for (int c = 0; c < n; ++c)
                    if (cells[c].kind == 2 && !obs->visible[c])
                        cells[c] = Cell{};
            }
            lastGoal = -1;
            useBook = false;
            active = {};
            focal = -1;
        }
        live = 0;
        strongestEnemy = 0;
        if (obs->players == 2 && tick > 60) {
            const int enemy = 1 - seat;
            const int theirLoss = oldArmy[enemy] - int(obs->total_army[enemy]);
            const int ourLoss = oldArmy[seat] - int(obs->total_army[seat]);
            // A large one-sided spend can expose a player who just bought a neutral city.
            if (theirLoss - ourLoss >= 32 && int(obs->total_land[enemy]) + 2 >= int(obs->total_land[seat]) / 2)
                attackWindowEnd = tick + 70;
        }
        for (int p = 0; p < obs->players; ++p) {
            int total = int(obs->total_army[p]);
            if (tick > 0 && tick % 2 == 0 && obs->alive[p]) {
                int produced = total - oldArmy[p] - (tick % 50 == 0 ? int(obs->total_land[p]) : 0) - 1;
                if (produced >= 0 && produced <= n) {
                    if (produced >= economicTowers[p] || tick - lastChange[p] > 18) {
                        economicTowers[p] = produced;
                        lastChange[p] = tick;
                    }
                }
            }
            oldArmy[p] = total;
        }
        for (int p = 0; p < obs->players; ++p)
            if (obs->alive[p]) {
                ++live;
                if (p != seat)
                    strongestEnemy = std::max(strongestEnemy, int(obs->total_army[p]));
            }
        owned.clear();
        mobile.clear();
        enemies.clear();
        kings.clear();
        structures.clear();
        frontierSeeds.clear();
        selectedEnemy = -1;
        int nearest = unreachable;
        for (int c = 0; c < n; ++c) {
            auto &v = cells[c];
            if (obs->visible[c]) {
                if (obs->players == 2) {
                    const int next = obs->owner[c];
                    if (next == seat && hostile(c))
                        ++enemyLosses;
                    if (next < 0 || (next == seat && (tick == 0 || enemyExcludedThrough[c] == tick)))
                        enemyExcludedThrough[c] = tick + 1;
                    if (next == 1 - seat && v.firstEnemy < 0)
                        earliestLandLimit[c] = int(obs->total_land[1 - seat]) + enemyLosses + 1;
                }
                v.owner = obs->owner[c];
                v.army = int(std::clamp<std::int64_t>(obs->army[c], 0, 100000000));
                v.kind = obs->structure[c];
                v.blocked = obs->terrain[c] == 1;
                v.seen = tick;
                if (hostile(c) && v.firstEnemy < 0)
                    v.firstEnemy = tick;
            } else {
                if (v.seen < 0)
                    v.blocked = obs->fog_obstacles[c];
                if (mine(c)) {
                    v.owner = -3;
                    v.army = 0;
                }
                if (hostile(c) && !obs->alive[v.owner]) {
                    if (v.kind == 2)
                        v.kind = 1;
                    v.owner = -3;
                    v.army = v.kind ? std::max(1, v.army / 2) : 0;
                }
            }
            if (mine(c)) {
                owned.push_back(c);
                if (v.army > 1)
                    mobile.push_back(c);
            }
            if (hostile(c) && tick - v.seen < 20) {
                enemies.push_back(c);
                if (md(c, home) < nearest) {
                    nearest = md(c, home);
                    selectedEnemy = v.owner;
                }
            }
            if (hostile(c) && v.kind == 2)
                kings.push_back(c);
            if (!mine(c) && v.kind == 1 && !wall(c))
                structures.push_back(c);
        }
        auto pass = [&](int c) { return open(c); };
        flood({home}, homeDist, pass);
        flood(owned, ownDist, pass);
        flood(enemies, contact, pass);
        flood(kings, kingDist, pass);
        for (int c : owned)
            for (int v : edges[c])
                if (!wall(v) && !mine(v) && open(v)) {
                    frontierSeeds.push_back(c);
                    break;
                }
        flood(frontierSeeds, frontier, [&](int c) { return mine(c); });
        std::fill_n(enemyPower.begin(), n, 0);
        for (int e : enemies)
            if (cells[e].seen == tick)
                for (int v : edges[e])
                    if (v >= 0)
                        enemyPower[v] = std::max(enemyPower[v], cells[e].army - 1);
        // Estimate where another observation is valuable. The first enemy
        // contact and the minimum duel spawn separation constrain the search.
        for (int c = 0; c < n; ++c) {
            int fog = 0, exits = 0;
            for (int v : edges[c])
                if (!wall(v)) {
                    fog += cells[v].seen < 0;
                    exits += !mine(v) && open(v);
                }
            interest[c] = fog;
            float forward = std::min(16, homeDist[c]) * .025f;
            if (!enemies.empty())
                forward = .42f / (1 + .19f * contact[c]);
            if (!kings.empty())
                forward += .70f / (1 + .15f * kingDist[c]);
            float reward = 1.0f + .10f * fog + .075f * std::min(3, exits) + forward;
            if (hostile(c))
                reward += 1.35f;
            reward *= tick < 50 ? 1.f : 1.f + .13f * ((tick % 50) - 23) / 25.f;
            if (cells[c].kind == 1 && hostile(c))
                reward += 7;
            cellReward[c] = reward;
        }
    }
    // Dynamic programming on the shortest route DAG. Enemy stacks can collect
    // their own garrisons and must pay for our defenders at every step.
    // Tree knapsack. Each selected branch costs one move into its parent.
    // Keep the first playable command for each achievable relief deadline.
    struct Relief {
        int power = -1;
        Move first{};
    };
    std::array<std::array<Relief, 12>, capacity> rescueTrees;
    Relief earliestRelief(int &response, int needed, int delay, int allies) {
        int count = 0;
        while (count < allies && distance[queue[count]] <= delay)
            ++count;
        for (int j = 0; j < count; ++j) {
            int c = queue[j];
            for (int t = 0; t <= delay - distance[c]; ++t)
                rescueTrees[c][t] = {};
            rescueTrees[c][0].power = 0;
        }
        for (int j = count - 1; j > 0; --j) {
            int c = queue[j], p = pathNext[c];
            for (int t = delay - distance[p]; t > 0; --t) {
                auto &plan = rescueTrees[p][t];
                for (int branch = 0; branch < t; ++branch) {
                    const auto &subtree = rescueTrees[c][branch];
                    const auto &existing = rescueTrees[p][t - branch - 1];
                    if (subtree.power < 0 || existing.power < 0)
                        continue;
                    int arrival = cells[c].army - 1 + subtree.power;
                    if (arrival < 1 || existing.power + arrival <= plan.power)
                        continue;
                    plan.power = existing.power + arrival;
                    plan.first = subtree.first.from >= 0 ? subtree.first : Move{c, p, false};
                }
            }
        }
        response = unreachable;
        for (int t = 1; t <= delay; ++t)
            if (rescueTrees[home][t].power >= needed) {
                response = t;
                return rescueTrees[home][t];
            }
        return {};
    }
    int reliefSource = -1, reliefNext = -1;
    void defend() {
        reliefSource = reliefNext = -1;
        reserve = 0;
        int count = flood({home}, distance, [&](int c) { return open(c); });
        int danger = -1, delay = 100, deficit = 0;
        float severity = -1;
        std::array<bool, 8> near{};
        for (int e : enemies)
            if (distance[e] <= 14 && cells[e].seen + 2 >= tick && cells[e].army >= 3)
                near[cells[e].owner] = true;
        for (int p = 0; p < obs->players; ++p)
            if (near[p]) {
                pathNeed[home] = 0;
                for (int i = 1; i < count; ++i) {
                    int c = queue[i], least = 100000000;
                    if (distance[c] > 14)
                        break;
                    for (int v : edges[c])
                        if (v >= 0 && distance[v] < distance[c]) {
                            int toll = v == home ? 0 : (cells[v].owner == p ? -guard(v) : guard(v));
                            least = std::min(least, pathNeed[v] + 1 + toll);
                        }
                    pathNeed[c] = least;
                }
                for (int e : enemies)
                    if (cells[e].owner == p && distance[e] <= 14 && cells[e].seen + 2 >= tick &&
                        cells[e].army >= 3) {
                        int steps = distance[e];
                        int arrival = cells[e].army - pathNeed[e] - (steps - 1) / 2;
                        if (steps <= 6)
                            reserve = std::max(reserve, arrival);
                        int gap = arrival - cells[home].army;
                        float risk = (gap + 8.f) / (steps * steps);
                        if (steps <= 11 && gap > 0 && risk > severity) {
                            danger = e;
                            delay = steps;
                            deficit = gap;
                            severity = risk;
                        }
                    }
            }
        // Beyond contact a newly generated garrison and reaction time absorb
        // part of a stack. Never immobilize the general for a remote scout.
        for (int e : edges[home])
            if (hostile(e) && cells[e].army > 2 && cells[home].army > cells[e].army + 1) {
                bool covered = true;
                for (int v : edges[home])
                    if (v != e && hostile(v) && cells[v].army >= 3)
                        covered = false;
                if (covered)
                    bid({home, e, false}, 70 + cells[e].army, home);
            }
        if (danger < 0)
            return;
        int allies = flood({home}, distance, [&](int c) { return mine(c); });
        pathNeed[home] = 0;
        for (int i = 1; i < allies; ++i) {
            int c = queue[i], d = distance[c];
            if (d > delay)
                break;
            int bestGain = -1, parent = -1;
            for (int v : edges[c])
                if (v >= 0 && distance[v] + 1 == d && pathNeed[v] > bestGain) {
                    bestGain = pathNeed[v];
                    parent = v;
                }
            pathNeed[c] = bestGain + std::max(0, cells[c].army - 1);
            pathNext[c] = parent;
        }
        int response = unreachable, needed = deficit + 3 + int(deficit * .10f);
        Relief plan;
        if (obs->players == 2)
            plan = earliestRelief(response, needed, delay, allies);
        bool canDelay = obs->players == 2 && delay >= 4 && response + 2 <= delay;
        if (canDelay) {
            reliefSource = plan.first.from;
            reliefNext = plan.first.to;
        } else if (plan.first.from >= 0) {
            float emergency = 48.0f + 2.6f * std::min(deficit, plan.power) / (1.f + response * .6f);
            bid(plan.first, emergency, home);
        }
        for (int i = 1; i < allies; ++i) {
            int c = queue[i], d = distance[c];
            if (d > delay)
                break;
            int parent = pathNext[c];
            if (cells[c].army > 1 && !canDelay && plan.first.from < 0) {
                float delivered = std::min(deficit + 3, pathNeed[c]);
                float value = 26 + 3.4f * delivered / (.7f + d);
                if (pathNeed[c] >= deficit)
                    value += 12.f / (.5f + d);
                bid({c, parent, false}, value, home);
            }
        }
        for (int c : edges[danger])
            if (mine(c) && c != home && cells[c].army > 1) {
                int hit = std::min(cells[danger].army, cells[c].army - 1);
                bid({c, danger, false}, 24 + 2.8f * std::min(hit, deficit + 5), home);
            }
    }
    // Enumerate short simple paths for the next expansion command. Walking
    // through a friendly stack explicitly credits its troops only once.
    void walk(int c, int force, int depth, int *path, Move first, float sum, float startCost) {
        for (int v : edges[c]) {
            if (wall(v) || std::find(path, path + depth + 1, v) != path + depth + 1)
                continue;
            bool ours = mine(v);
            int troops = force - 1 + (ours ? cells[v].army : -guard(v));
            if (troops < 1)
                continue;
            Move m = depth ? first : Move{c, v, false};
            float gain = ours ? -.25f : cellReward[v];
            if (!ours && cells[v].kind)
                gain += cells[v].kind == 2 ? (live > 2 ? 140.f : 12.f) : 7.f;
            if (!ours)
                gain -= std::min(live > 2 ? 9.f : 4.f, (live > 2 ? .20f : .095f) * guard(v));
            if (obs->players == 2 && !ours)
                gain += hostile(v) ? battleRebate * std::min(36, guard(v)) : -neutralPrice * guard(v);
            if (enemyPower[v] >= troops)
                gain -= std::min(3.f, .15f * troops);
            if (ours && v == lastFrom && c == lastTo)
                gain -= 1.4f;
            float total = sum + gain;
            // A tiny stack that can open the same land leaves the producer
            // available for a later expedition. Reward fresh scouting early.
            float value = (total + std::min(2, interest[v]) * .06f - startCost) / (depth + 1.f);
            if (!ours)
                bid(m, value, v);
            if (depth < 3 && troops > 1) {
                path[depth + 1] = v;
                walk(v, troops, depth + 1, path, m, total, startCost);
            }
        }
    }
    float neutralPrice = 0.f, battleRebate = 0.f;
    void priceTroops() {
        if (obs->players != 2)
            return;
        int them = 1 - seat;
        float ours = std::max(0.f, float(obs->total_army[seat] - obs->total_land[seat]));
        float opponent = std::max(0.f, float(obs->total_army[them] - obs->total_land[them]));
        float ratio = (ours + 30.f) / (opponent + 30.f);
        neutralPrice = std::clamp(0.88f - (ratio - 1.f) * 0.74f, .02f, 1.55f);
        battleRebate = 0.07f * std::clamp(ratio, .6f, 1.6f);
    }
    void expansion() {
        int path[6];
        int scanned = 0;
        for (int c : mobile) {
            if ((++scanned % 32) == 0 && expired())
                break;
            if (frontier[c] > 3)
                continue;
            path[0] = c;
            float price = std::min(1.5f, cells[c].army * .020f);
            if (cells[c].kind && tick >= 50)
                price += .45f;
            if (c == home && reserve > 1)
                price += 2.f;
            walk(c, cells[c].army, 0, path, {}, 0, price);
        }
    }
    // Reverse BFS fixes the number of commands. Among equal length routes,
    // minimize the starting army needed to cross every intermediate square.
    std::array<unsigned char, capacity> exposed{};
    std::array<int, capacity> reliefDistance{};
    void estimateExposure(int target) {
        std::fill_n(exposed.begin(), n, 0);
        for (int k = 0; k < n; ++k)
            if (hostile(k) || k == target) {
                int x = k % w, y = k / w;
                for (int yy = std::max(0, y - 1); yy <= std::min(h - 1, y + 1); ++yy)
                    for (int xx = std::max(0, x - 1); xx <= std::min(w - 1, x + 1); ++xx)
                        exposed[yy * w + xx] = 1;
            }
        flood({target}, reliefDistance, [&](int c) { return open(c) && !mine(c); });
    }
    // Search a second approach that keeps the column outside enemy vision.
    void covertRoutes(int target, int extra) {
        using Entry = std::pair<float, int>;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
        std::array<float, capacity> cost;
        std::fill_n(cost.begin(), n, 1e20f);
        std::fill_n(distance.begin(), n, unreachable);
        cost[target] = 0;
        distance[target] = 0;
        pathNext[target] = -1;
        pathNeed[target] = guard(target) + extra + 1;
        heap.emplace(0.f, target);
        while (!heap.empty()) {
            auto [price, v] = heap.top();
            heap.pop();
            if (price != cost[v] || distance[v] > 28)
                continue;
            float edge = 1.f;
            if (v != target) {
                edge += 1.7f * (exposed[v] ? 1.f : occupancy[v] * .55f);
                edge += 0.06f * (mine(v) ? 0.f : std::min(60, guard(v)));
                if (mine(v))
                    edge -= std::min(.35f, (cells[v].army - 1) * .018f);
            }
            for (int c : edges[v])
                if (open(c) && price + edge < cost[c]) {
                    cost[c] = price + edge;
                    distance[c] = distance[v] + 1;
                    pathNext[c] = v;
                    pathNeed[c] = std::max(2, pathNeed[v] + 1 +
                                                  (v == target ? 0
                                                   : mine(v)   ? -cells[v].army
                                                               : guard(v)));
                    heap.emplace(cost[c], c);
                }
        }
    }
    // Count defenders that can arrive between first sight and impact.
    float readyDefense(int root, int target, const std::vector<int> &march, int prep, int spare,
                       int intercept) {
        int steps = int(march.size()), warning = steps + prep;
        if (!exposed[root]) {
            warning = 1;
            for (int j = 0; j < steps; ++j)
                if (exposed[march[j]] || occupancy[march[j]] > .80f) {
                    warning = steps - j;
                    break;
                }
        }
        int located = 0, first = 0, second = 0, firstDist = 0, secondDist = 0;
        for (int e : enemies)
            if (e != target && cells[e].seen + 3 >= tick) {
                int excess = std::max(0, guard(e) - 1);
                located += excess;
                int d = reliefDistance[e];
                if (d > warning || std::find(march.begin(), march.end(), e) != march.end())
                    continue;
                if (excess > first) {
                    second = first;
                    secondDist = firstDist;
                    first = excess;
                    firstDist = d;
                } else if (excess > second) {
                    second = excess;
                    secondDist = d;
                }
            }
        float next = (firstDist + secondDist <= warning ? .70f : .15f) * second;
        float unlocated =
            std::min(190.f, 0.19f * std::max(0, spare - located) * std::min(1.f, (warning + 1) / 10.f));
        return std::max(.70f * intercept, 0.78f * (first + next) + unlocated);
    }
    void routes(int target, int extra) {
        int count = flood({target}, distance, [&](int c) { return open(c); });
        pathNeed[target] = (mine(target) ? 1 : guard(target) + 1) + extra;
        pathNext[target] = -1;
        for (int i = 1; i < count; ++i) {
            int c = queue[i], need = 100000000, parent = -1;
            for (int v : edges[c])
                if (v >= 0 && distance[v] + 1 == distance[c]) {
                    int value = pathNeed[v] + 1;
                    if (v != target)
                        value = std::max(2, value + (mine(v) ? -cells[v].army : guard(v)));
                    if (value < need || (value == need && mine(v) && !mine(parent))) {
                        need = value;
                        parent = v;
                    }
                }
            pathNeed[c] = need;
            pathNext[c] = parent;
        }
    }
    int focal = -1;
    std::array<int, capacity> focalDist;
    std::array<float, capacity> likelihood{}, information{}, forceValue{}, hiddenKing{};
    std::array<int, capacity> landmarkDistance[3];
    // Estimate the value of revealing a cell from distinct early contacts and
    // the still-unseen spawn region. Observed capitals override the estimate.
    // Recover the duel generator's distribution conditional on the *visible*
    // obstacle mask and our own spawn. Neutral cities were obstacles when
    // the two spawns were drawn. Rejected pairs gradually relax the ratio.
    std::array<float, capacity> shape{}, spawnPrior{};
    std::array<int, capacity> shapeQueue, shapeDist;
    std::array<int, capacity> shapeStamp{};
    int shapeCursor = 0, shapeSerial = 0;
    bool priorReady = false;
    bool spawnBlocked(int v) const { return wall(v) || cells[v].kind == 1; }
    void buildPrior() {
        if (priorReady || obs->players != 2 || n > 1024)
            return;
        static const auto weights = [] {
            std::array<float, 15> out{};
            float value = int(std::pow(1.18, 15));
            for (int d = 1; d <= 14; ++d) {
                out[d] = value;
                value /= d < 7 ? 1.18f : 1.1f;
                if (d < 5)
                    value /= 1.6f;
            }
            return out;
        }();
        for (int end = std::min(n, shapeCursor + 70); shapeCursor < end; ++shapeCursor) {
            int source = shapeCursor;
            if (spawnBlocked(source))
                continue;
            int tail = 1, first = 0;
            ++shapeSerial;
            shapeQueue[0] = source;
            shapeDist[source] = 0;
            shapeStamp[source] = shapeSerial;
            for (int j = 0; j < tail; ++j) {
                int c = shapeQueue[j], d = shapeDist[c] + 1;
                if (d > 14)
                    continue;
                for (int v : edges[c])
                    if (!spawnBlocked(v) && shapeStamp[v] != shapeSerial) {
                        shapeStamp[v] = shapeSerial;
                        shapeQueue[tail++] = v;
                        shapeDist[v] = d;
                        if (d != 1 || ++first <= 3)
                            shape[source] += weights[d];
                    }
            }
        }
        if (shapeCursor < n)
            return;
        std::array<float, 101> thresholds{}, density{}, mass{};
        for (int r = 0; r <= 100; ++r)
            thresholds[r] = 1.f - std::pow(.5f + .005f * r, 4.f) - .02f;
        auto retry = [&](float a, float b) {
            float ratio = std::min(a, b) / std::max(1.f, std::max(a, b));
            return int(std::lower_bound(thresholds.begin(), thresholds.end(), ratio, std::greater<float>()) -
                       thresholds.begin());
        };
        int total = 0;
        for (int a = 0; a < n; ++a)
            if (shape[a] > 0)
                for (int b = a + 1; b < n; ++b)
                    if (shape[b] > 0 && md(a, b) >= 15) {
                        ++total;
                        ++density[retry(shape[a], shape[b])];
                    }
        if (total) {
            float surviving = 1, accepted = 0;
            for (int r = 0; r <= 100; ++r) {
                accepted += density[r] / total;
                mass[r] = surviving;
                surviving *= std::max(0.f, 1 - accepted);
            }
            for (int r = 99; r >= 0; --r)
                mass[r] += mass[r + 1];
            for (int c = 0; c < n; ++c)
                if (shape[c] > 0 && md(home, c) >= 15)
                    spawnPrior[c] = 0.025f + (1.f - 0.025f) * mass[retry(shape[home], shape[c])] / mass[0];
        }
        priorReady = true;
    }

    void locate() {
        auto pass = [&](int c) { return open(c); };
        if (!kings.empty()) {
            float high = -1e20f;
            for (int c : kings) {
                int p = cells[c].owner;
                float value =
                    1.f / (4 + std::min(50, ownDist[c])) *
                    std::clamp(float(obs->total_army[seat] + 8) / (obs->total_army[p] + 8), .40f, 2.5f);
                if (c == focal)
                    value *= 1.15f;
                if (value > high) {
                    focal = c;
                    high = value;
                }
            }
        } else if (tick % 5 == 0 || focal < 0 || cells[focal].seen >= 0) {
            int anchor = -1, earliest = 100000000;
            for (int c = 0; c < n; ++c)
                if (cells[c].firstEnemy >= 0 && cells[c].firstEnemy < earliest) {
                    anchor = c;
                    earliest = cells[c].firstEnemy;
                }
            std::array<int, 3> landmarks{anchor, -1, -1};
            for (int j = 1; j < 3 && anchor >= 0; ++j) {
                int when = 100000000;
                for (int c = 0; c < n; ++c)
                    if (cells[c].firstEnemy >= 0 && cells[c].firstEnemy < when) {
                        bool separate = true;
                        for (int i = 0; i < j; ++i)
                            if (landmarks[i] >= 0 && md(c, landmarks[i]) < 5)
                                separate = false;
                        if (separate) {
                            landmarks[j] = c;
                            when = cells[c].firstEnemy;
                        }
                    }
            }
            for (int j = 0; j < 3; ++j)
                if (obs->players == 2 && landmarks[j] >= 0) {
                    int at = cells[landmarks[j]].firstEnemy;
                    flood({landmarks[j]}, landmarkDistance[j], [&](int x) {
                        if (enemyExcludedThrough[x] > at)
                            return false;
                        return open(x) || (at > 110 && cells[x].seen < 0 && obs->fog_obstacles[x]);
                    });
                } else
                    flood({landmarks[j]}, landmarkDistance[j], pass);
            float total = 0;
            for (int c = 0; c < n; ++c) {
                float prob = 1;
                if (cells[c].seen >= 0 || !open(c) || homeDist[c] >= unreachable)
                    prob = 0;
                if (md(c, home) < (obs->players == 2 ? 15 : std::max(9, 16 - obs->players)))
                    prob = 0;
                for (int j = 0; j < 3 && landmarks[j] >= 0; ++j) {
                    int dist = landmarkDistance[j][c];
                    if (obs->players == 2) {
                        int at = cells[landmarks[j]].firstEnemy;
                        if (dist > earliestLandLimit[landmarks[j]])
                            prob *= 0.01f;
                        float spread = std::clamp(at * .07f, 3.5f, 11.f);
                        prob *= 1.f / (1.f + 0.13f * std::abs(dist - spread));
                    } else
                        prob *= 1.f / (1 + .075f * std::abs(dist - 4));
                }
                if (!enemies.empty())
                    prob *= 1.f / (1 + (obs->players == 2 ? 0.14f : .24f) * contact[c]);
                if (priorReady && prob > 0)
                    prob *= spawnPrior[c];
                if (obs->players == 2 && prob > 0)
                    prob *= originSupport(c);
                likelihood[c] = prob;
                total += prob;
            }
            float scale = total > 0 ? 1.f / total : 0;
            for (int c = 0; c < n; ++c) {
                float mass = 0;
                for (int y = std::max(0, c / w - 1); y <= std::min(h - 1, c / w + 1); ++y)
                    for (int x = std::max(0, c % w - 1); x <= std::min(w - 1, c % w + 1); ++x)
                        mass += likelihood[y * w + x];
                information[c] = mass * scale;
            }
            for (int c = 0; c < n; ++c)
                hiddenKing[c] = likelihood[c] * scale;
            float high = 0;
            for (int c = 0; c < n; ++c)
                if (likelihood[c] > 0) {
                    float mass = 0;
                    for (int y = std::max(0, c / w - 2); y <= std::min(h - 1, c / w + 2); ++y)
                        for (int x = std::max(0, c % w - 2); x <= std::min(w - 1, c % w + 2); ++x)
                            mass += likelihood[y * w + x];
                    float score = (4 * likelihood[c] + mass) / (1 + .045f * ownDist[c]);
                    if (c == focal)
                        score *= 1.08f;
                    if (score > high) {
                        high = score;
                        focal = c;
                    }
                }
            if (high == 0)
                focal = -1;
        }
        flood({focal}, focalDist, pass);
        for (int c = 0; c < n; ++c) {
            if (kings.empty() && !enemies.empty() && !mine(c))
                cellReward[c] += std::min(1.3f, information[c] * 19.f);
            float hunt = 1.f / (1 + .12f * focalDist[c]);
            float fight = 1.f / (1 + .25f * contact[c]);
            int detour = std::max(0, homeDist[c] + focalDist[c] - focalDist[home]);
            float central = (.75f + .38f / (1 + .25f * detour));
            forceValue[c] = central * (0.45f * hunt + .18f * fight) * (tick < 95 ? .12f : 1.f);
        }
    }
    static constexpr int horizon = 15, width = 48;
    struct March {
        std::array<short, horizon + 1> path{};
        int force = 0, initial = 0;
        float loot = 0, gatheredPrice = 0, merit = 0;
    };
    std::array<March, width> layer;
    std::vector<March> nextLayer;
    void campaign() {
        int count = 0;
        for (int depth = 1; depth <= horizon; ++depth) {
            nextLayer.clear();
            auto extend = [&](const March &prev, int v) {
                if (wall(v) || prev.force < 2 ||
                    std::find(prev.path.begin(), prev.path.begin() + depth, v) != prev.path.begin() + depth)
                    return;
                bool ours = mine(v);
                int g = guard(v);
                int force = prev.force - 1 + (ours ? g : -g);
                if (force < 1)
                    return;
                int s = prev.path[0], first = depth == 1 ? v : prev.path[1];
                float loot = prev.loot;
                float price = prev.gatheredPrice;
                if (ours) {
                    loot -= .23f;
                    price += std::min(250, g - 1) * forceValue[v] * (g < 10 ? .50f : .99f);
                } else {
                    loot += cellReward[v] + (hostile(v) ? .85f : 0.f) - std::min(16.f, g * .13f);
                    if (cells[v].kind == 1)
                        loot += hostile(v) ? 26.f : std::clamp(4.f + .025f * (tick - 100), 4.f, 16.f);
                    if (cells[v].kind == 2)
                        loot += live > 2 ? 160.f : 12.f;
                    if (enemyPower[v] > force && depth <= 3)
                        loot -= std::min(20.f, .50f * std::min(force, enemyPower[v]));
                    if (obs->players == 2)
                        loot += hostile(v) ? battleRebate * std::min(48, g) : -neutralPrice * g;
                    if (cells[v].seen < 0)
                        loot += .25f;
                    if (kings.empty() && cells[v].seen < 0 && focalDist[v] <= 6)
                        loot += .65f;
                }
                if (obs->players == 2 && kings.empty() && tick >= 110 && depth <= 8 &&
                    information[v] > .002f) {
                    int dx = v % w - prev.path[depth - 1] % w, dy = v / w - prev.path[depth - 1] / w;
                    int vx = v % w, vy = v / w;
                    float discovery = 0;
                    for (int offset = -1; offset <= 1; ++offset) {
                        int x = vx + dx + (dx ? 0 : offset), y = vy + dy + (dy ? 0 : offset);
                        if (x < 0 || x >= w || y < 0 || y >= h)
                            continue;
                        int z = y * w + x;
                        if (cells[z].seen >= 0 || !hiddenKing[z])
                            continue;
                        bool newSight = true;
                        for (int j = 0; j < depth - 1; ++j)
                            if (std::abs(x - prev.path[j] % w) <= 1 && std::abs(y - prev.path[j] / w) <= 1) {
                                newSight = false;
                                break;
                            }
                        if (newSight)
                            discovery += hiddenKing[z];
                    }
                    loot += 65.0f * discovery * (1.f + 0.85f * std::min(45, force) / 45.f);
                }
                float delivery =
                    std::min(250, force) * forceValue[v] - std::min(250, prev.initial) * forceValue[s] - price;
                if (pressAttack && !kings.empty() && hostile(v) && cells[v].kind == 2)
                    loot += 120.f;
                float advance = std::clamp(focalDist[s] - focalDist[v], -15, 15) * 0.18f;
                float rhythm = tick < 60 ? 1.f : 1.32f - .58f * (tick % 50) / 49.f;
                float merit = loot + rhythm * delivery + advance;
                float score = merit / (depth + 2.5f);
                if (depth == 1 && cells[v].kind == 2 && hostile(v))
                    score = 10000;
                bid({s, first, false}, score, v);
                if (depth == horizon || force < 2)
                    return;
                March next = prev;
                next.path[depth] = short(v);
                next.force = force;
                next.loot = loot;
                next.gatheredPrice = price;
                next.merit = merit + .04f * std::min(120, force);
                nextLayer.push_back(next);
            };
            if (depth == 1) {
                for (int s : mobile) {
                    if (s == home && reserve > 1)
                        continue;
                    March start;
                    start.path[0] = short(s);
                    start.force = start.initial = cells[s].army;
                    for (int v : edges[s])
                        extend(start, v);
                }
            } else
                for (int i = 0; i < count; ++i) {
                    const auto &p = layer[i];
                    int c = p.path[depth - 1];
                    for (int v : edges[c])
                        extend(p, v);
                }
            if (nextLayer.empty() || expired())
                break;
            auto better = [](const March &a, const March &b) { return a.merit > b.merit; };
            // Keep at most one route per endpoint, preventing a single
            // prosperous corner from exhausting all of the search width.
            std::fill_n(mark.begin(), n, -1);
            int unique = 0;
            for (int i = 0; i < int(nextLayer.size()); ++i) {
                int end = nextLayer[i].path[depth], &slot = mark[end];
                if (slot < 0) {
                    slot = unique++;
                    nextLayer[slot] = nextLayer[i];
                } else if (better(nextLayer[i], nextLayer[slot]))
                    nextLayer[slot] = nextLayer[i];
            }
            count = std::min(width, unique);
            if (unique > count)
                std::nth_element(nextLayer.begin(), nextLayer.begin() + count, nextLayer.begin() + unique,
                                 better);
            std::copy_n(nextLayer.begin(), count, layer.begin());
        }
    }
    struct Order {
        std::vector<Move> moves;
        float quality = 0;
        int target = -1, started = 0;
    };
    Order active, proposed;
    std::array<int, capacity> nodeDepth, treeParent, treeOrder, treeArmy;
    std::array<bool, capacity> selected;
    std::array<float, capacity> treeYield;
    int tree(int root) {
        std::fill_n(nodeDepth.begin(), n, unreachable);
        treeOrder[0] = root;
        nodeDepth[root] = 0;
        treeParent[root] = -1;
        treeYield[root] = 0;
        int tail = 1;
        for (int i = 0; i < tail; ++i) {
            int c = treeOrder[i], d = nodeDepth[c];
            if (d >= 14)
                continue;
            for (int v : edges[c])
                if (mine(v) && nodeDepth[v] > d) {
                    float gain = treeYield[c] + std::min(300, available(c));
                    if (nodeDepth[v] == unreachable) {
                        nodeDepth[v] = d + 1;
                        treeOrder[tail++] = v;
                    } else if (gain <= treeYield[v])
                        continue;
                    treeParent[v] = c;
                    treeYield[v] = gain;
                }
        }
        return tail;
    }
    // Evaluate the prefixes of a connected gathering tree. Each selected edge
    // costs exactly one command, including bridges occupied by a single army.
    // A shared trunk is charged once. Issue orders in postorder before marching.
    enum class Objective { City, Scout, Capital };
    void assemble(int root, int target, const std::vector<int> &march, float worth, Objective style) {
        int tail = tree(root);
        for (int i = 0; i < tail; ++i) {
            int c = treeOrder[i];
            selected[c] = false;
            treeArmy[c] = available(c);
            if (c == home && reserve > 1)
                treeArmy[c] = cells[c].army / 2 >= reserve ? cells[c].army / 2 : 0;
        }
        if (treeArmy[root] == 0 && tail == 1)
            return;
        selected[root] = true;
        int power = 1 + treeArmy[root], moves = 0;
        int enemy = hostile(target) ? cells[target].owner : selectedEnemy;
        int spare = enemy < 0 ? std::max(0, strongestEnemy - 40)
                              : int(obs->total_army[enemy] - obs->total_land[enemy]);
        int cost = 0, intercept = 0, worst = 0, loot = 0;
        for (int c : march) {
            cost += mine(c) ? 1 - cells[c].army : 1 + guard(c);
            worst = std::max(worst, cost);
            if (!mine(c))
                loot += hostile(c) ? 2 : 1;
            if (cells[c].seen >= 0)
                for (int v : edges[c])
                    if (hostile(v) && v != target && std::find(march.begin(), march.end(), v) == march.end())
                        intercept = std::max(intercept, guard(v) - 1);
        }
        if (style == Objective::Capital)
            spare = std::max(0, spare - guard(target));
        int steps = int(march.size());
        for (int round = 0; round <= 14 && !expired(); ++round) {
            int turns = moves + steps;
            int growth = style == Objective::Capital && steps > 1 ? turns / 2 : 0;
            int surplus = power - cost - growth;
            if (power > worst && surplus > 0) {
                float standing = 1;
                if (style != Objective::City && steps > 1) {
                    float need = style == Objective::Capital
                                     ? std::max(float(intercept) * .8f, std::min(220.f, spare * .26f))
                                     : std::max(float(intercept) * .70f, std::min(120.f, spare * .25f));
                    if (obs->players == 2 && style == Objective::Capital)
                        need = readyDefense(root, target, march, moves, spare, intercept);
                    standing = std::clamp((surplus + 3.f) / (need + 6.f), .01f, 1.30f);
                    standing *= standing;
                    if (pressAttack && style == Objective::Capital && surplus > 4)
                        standing = std::max(standing, .38f);
                } else if (style == Objective::City && contact[target] <= 3)
                    standing = std::clamp((surplus + 6.f) / (intercept + 12.f), .10f, 1.f);
                float value = (worth * standing + loot) / (turns + 4.f);
                if (style == Objective::City && !hostile(target) && tick < 350) {
                    float usable = std::max(0.f, float(obs->total_army[seat] - obs->total_land[seat]));
                    value *= std::clamp((usable - guard(target)) / (.70f * guard(target) + 10.f), .35f, 1.4f);
                }
                if (live > 2 && style == Objective::Capital && enemy >= 0) {
                    float odds = (obs->total_army[seat] + 22.f) / (obs->total_army[enemy] + 22.f);
                    value *= std::clamp(odds, .5f, 1.8f);
                }
                if (target == active.target)
                    value *= 1.10f;
                if (value > proposed.quality) {
                    proposed = {};
                    proposed.quality = value;
                    proposed.target = target;
                    proposed.started = tick;
                    for (int j = tail - 1; j >= 1; --j) {
                        int c = treeOrder[j];
                        if (selected[c])
                            proposed.moves.push_back({c, treeParent[c], c == home && reserve > 1});
                    }
                    int c = root;
                    for (int v : march) {
                        proposed.moves.push_back({c, v, c == home && reserve > 1});
                        c = v;
                    }
                }
            }
            int bestLeaf = -1, add = 0, amount = 0;
            float high = 0;
            for (int j = 1; j < tail; ++j) {
                int c = treeOrder[j];
                if (selected[c] || treeArmy[c] == 0)
                    continue;
                int length = 0, gain = 0;
                for (int v = c; !selected[v]; v = treeParent[v]) {
                    ++length;
                    gain += treeArmy[v];
                }
                if (length + moves > 32 || gain == 0)
                    continue;
                float density = gain / (length + .5f);
                if (density > high) {
                    high = density;
                    bestLeaf = c;
                    add = length;
                    amount = gain;
                }
            }
            if (bestLeaf < 0)
                break;
            for (int c = bestLeaf; !selected[c]; c = treeParent[c])
                selected[c] = true;
            moves += add;
            power += amount;
        }
    }
    void expedition(int target, float worth, Objective style) {
        // Staging at the last friendly cell lets the assault gather several
        // distant reserves before stepping onto an exposed enemy square.
        bool covert = obs->players == 2 && style == Objective::Capital;
        if (covert)
            estimateExposure(target);
        int variants = covert ? 2 : 1;
        for (int lane = 0; lane < variants && !expired(); ++lane) {
            if (covert && lane == 1)
                covertRoutes(target, std::clamp(strongestEnemy / 5, 12, 200));
            else
                routes(target, std::clamp(strongestEnemy / 5, 12, 200));

            std::array<Target, 6> gates;
            for (auto &g : gates)
                g = {-1, -1e20f};
            for (int c : owned)
                if (distance[c] >= 1 && distance[c] <= 25 && !mine(pathNext[c])) {
                    float mass = std::min(100, available(c));
                    for (int v : edges[c])
                        if (mine(v))
                            mass += .50f * std::min(100, available(v));
                    float score =
                        -distance[c] + .04f * mass - .035f * std::max(0, pathNeed[c] - (strongestEnemy / 5));
                    Target g{c, score};
                    for (auto &slot : gates)
                        if (g.value > slot.value)
                            std::swap(slot, g);
                }
            std::vector<int> march;
            for (auto gate : gates)
                if (gate.cell >= 0 && !expired()) {
                    march.clear();
                    for (int v = pathNext[gate.cell]; v >= 0; v = pathNext[v])
                        march.push_back(v);
                    assemble(gate.cell, target, march, worth, style);
                }
        }
    }
    void combat() {
        for (int e : enemies)
            if (cells[e].seen == tick && cells[e].army >= 3) {
                int power = cells[e].army;
                for (int c : edges[e])
                    if (mine(c) && cells[c].army >= 3) {
                        int sent = cells[c].army - 1;
                        if (c == home && reserve > 1 && cells[c].army <= power + 1)
                            continue;
                        float menace = std::max(0.f, 1.f - homeDist[e] / 12.f);
                        float worth = (.03f + menace * .23f) * std::min(power, sent);
                        if (sent > power)
                            worth += 1.0f + .055f * std::min(80, power);
                        if (cells[e].kind && sent <= power)
                            worth *= .4f;
                        if (sent < 4 || (sent < power && power < 8))
                            continue;
                        bid({c, e, false}, worth, e);
                    }
            }
    }
    int closestProducerDistance(int c) const {
        int best = md(home, c);
        for (int v : owned)
            if (cells[v].kind)
                best = std::min(best, md(v, c));
        return best;
    }
    float cityCommitment(int c) const {
        if (obs->players != 2 || hostile(c))
            return 1.f;
        int enemy = 1 - seat;
        int margin = int(obs->total_army[seat] - obs->total_army[enemy]) - guard(c);
        int cityGap = economicTowers[enemy] - int(obs->total_cities[seat]);
        int territory = int(obs->total_land[seat] - obs->total_land[enemy]);
        // Price in future production as well as troops that can reach the buyer.
        float balance = margin + .68f * territory + 14.f * cityGap;
        float safety = std::clamp((balance + 36.0f) / 50.f, 0.1f, 1.6f);
        if (kings.empty() && contact[c] >= 5 && reserve < 4)
            safety *= 2.3f;
        if (contact[c] < 4 && enemyPower[c] > 3)
            safety *= .45f;
        return safety;
    }
    float investment(int c) const {
        if (hostile(c))
            return live > 2 ? 73.f : 57.f;
        int land = 0, cities = 0;
        for (int p = 0; p < obs->players; ++p)
            if (p != seat && obs->alive[p]) {
                land = std::max(land, int(obs->total_land[p]));
                cities = std::max(cities, economicTowers[p]);
            }
        float ours = float(obs->total_army[seat]);
        float remainingArmyRatio = (ours - guard(c)) / (strongestEnemy + 12.f);
        float value = std::clamp(19.f + 15.f * (remainingArmyRatio - .80f) + .022f * (tick - 120), 11.f, 38.f);
        value += std::clamp((land - int(obs->total_land[seat])) * .15f +
                                4.f * (cities - int(obs->total_cities[seat])),
                            -5.f, 15.f);
        if (contact[c] < 3)
            value *= .72f;
        int producerDistance = closestProducerDistance(c);
        if (live > 2) {
            value *= contact[c] >= 4 ? 2.7f : .95f;
            if (producerDistance <= 4 && obs->total_cities[seat] < 2)
                value *= 1.5f;
        }
        if (producerDistance > 5 && contact[c] > 5)
            value *= .85f;
        return value;
    }
    void operations() {
        bool valid = false;
        if (!active.moves.empty() && !mine(active.target)) {
            Move m = active.moves.front();
            int send = mine(m.from) ? (m.half ? cells[m.from].army / 2 : cells[m.from].army - 1) : 0;
            valid = send > 0 && !wall(m.to) && (mine(m.to) || send > guard(m.to));
            if (valid && !mine(m.to) && enemyPower[m.from] > send + 4)
                valid = false;
        }
        if (!valid)
            active = {};
        if (!valid || tick % 5 == 0) {
            proposed = {};
            const float capitalValue = pressAttack ? 350.f : 80.f;
            const float scoutValue = pressAttack ? 125.f : 26.f;
            for (int c : kings)
                if (!expired())
                    expedition(c, live > 2 ? 210.f : capitalValue, Objective::Capital);
            if (kings.empty() && focal >= 0 && !expired() && tick >= (live > 2 || pressAttack ? 75 : 150))
                expedition(focal, live > 2 ? 52.f : scoutValue, Objective::Scout);
            std::vector<Target> cities;
            for (int c : structures) {
                int d = unreachable;
                for (int v : edges[c])
                    if (v >= 0)
                        d = std::min(d, ownDist[v] + 1);
                if (d <= 8)
                    cities.push_back({c, (hostile(c) ? 2.f : 1.f) / (2 + d)});
            }
            std::sort(cities.begin(), cities.end(), [](Target a, Target b) { return a.value > b.value; });
            for (int i = 0; i < std::min(4, int(cities.size())) && !expired(); ++i) {
                int c = cities[i].cell;
                float value = investment(c) * cityCommitment(c);
                if (pressAttack)
                    value *= hostile(c) ? 2.3f : 0.15f;
                if (obs->players == 2 && !hostile(c))
                    value *= std::clamp(1.15f - neutralPrice * 1.3f, .04f, 1.6f);
                expedition(c, value, Objective::City);
            }
            if (!proposed.moves.empty() &&
                (!valid || proposed.quality > active.quality * 1.20f || tick - active.started > 24))
                active = proposed;
        }
        if (!active.moves.empty())
            bid(active.moves.front(), active.quality, active.target);
    }
    std::uint64_t stream = 0;
    unsigned draw() {
        stream ^= stream << 13;
        stream ^= stream >> 7;
        stream ^= stream << 17;
        return unsigned(stream >> 24);
    }
    std::array<Move, 50> book;
    float bookScore = -1e20f;
    std::array<int, capacity> simulated{}, visited{};
    int serial = 0, visitId = 0;
    bool useBook = true;
    // Search complete dispatch schedules before the first land pulse. All
    // sampled boards treat unrevealed obstacles identically to mountains.
    // Economic ties are broken by scouting and by next-round frontier access.
    void openingSample() {
        ++serial;
        std::array<Move, 50> plan;
        std::array<int, 52> land;
        int count = 1, active = home, power = 1, garrison = 1, transit = -1;
        int delay = 14 + draw() % 14;
        int bias = int(draw() % 5);
        land[0] = home;
        simulated[home] = serial;
        for (int t = 0; t < 50; ++t) {
            if (t >= delay) {
                if (active == home || power < 2) {
                    active = home;
                    power = garrison;
                    transit = -1;
                }
                if (power >= 2) {
                    int chosen = -1;
                    if (transit >= 0) {
                        for (int v : edges[active])
                            if (v == transit)
                                chosen = v;
                        transit = -1;
                    }
                    float high = -1e20f;
                    if (chosen < 0)
                        for (int v : edges[active])
                            if (open(v) && simulated[v] != serial) {
                                int exits = 0;
                                for (int u : edges[v])
                                    exits += open(u) && simulated[u] != serial;
                                float score =
                                    (draw() % 1024) / 300.f + .16f * std::min(20, homeDist[v]) + exits * .21f;
                                if (bias < 3)
                                    score += .16f * (focalDist[home] - focalDist[v]);
                                if (power > 2 && !exits)
                                    score -= 3;
                                if (score > high) {
                                    high = score;
                                    chosen = v;
                                }
                            }
                    if (chosen < 0) {
                        // Seek the nearest unused exit through our occupied
                        // tiles. A different equal-cost exit gives a new wave.
                        ++visitId;
                        std::array<int, 52> openCells, steps, first;
                        int tail = 1;
                        openCells[0] = active;
                        steps[0] = 0;
                        first[0] = -1;
                        visited[active] = visitId;
                        for (int j = 0; j < tail; ++j) {
                            int c = openCells[j], d = steps[j];
                            if (d >= 6)
                                continue;
                            for (int v : edges[c])
                                if (open(v) && visited[v] != visitId) {
                                    int f = d ? first[j] : v;
                                    if (simulated[v] != serial) {
                                        float value = -1.4f * d + (draw() % 1024) / 700.f;
                                        if (value > high) {
                                            high = value;
                                            chosen = f;
                                            if (d == 1)
                                                transit = v;
                                            else
                                                transit = -1;
                                        }
                                    } else {
                                        visited[v] = visitId;
                                        first[tail] = f;
                                        steps[tail] = d + 1;
                                        openCells[tail++] = v;
                                    }
                                }
                        }
                    }
                    if (chosen >= 0) {
                        plan[t] = {active, chosen, false};
                        if (active == home)
                            garrison = 1;
                        --power;
                        if (simulated[chosen] != serial) {
                            simulated[chosen] = serial;
                            land[count++] = chosen;
                        } else
                            power += chosen == home ? garrison : 1;
                        active = chosen;
                        if (active == home)
                            garrison = power;
                    }
                }
            }
            if (t % 2)
                ++garrison;
        }
        int frontier = 0, extent = 0, forward = 0;
        float pos = 0;
        for (int j = 0; j < count; ++j) {
            int c = land[j], exits = 0;
            for (int v : edges[c])
                exits += open(v) && simulated[v] != serial;
            frontier += exits > 0;
            extent = std::max(extent, std::min(18, homeDist[c]));
            int progress = std::clamp(focalDist[home] - focalDist[c], -20, 20);
            forward = std::max(forward, progress);
            pos += progress;
        }
        float score = 12.f * count + .65f * frontier + .55f * extent + 1.1f * forward + .045f * pos;
        if (score > bookScore) {
            bookScore = score;
            book = plan;
        }
    }
    std::array<int, capacity> onRoute{};
    int stamp = 0;
    // Check short capital attacks against visible garrisons and nearby relief.
    // Fog makes the success margin an estimate even for a short route.
    void finishPath(int target, int *path, int depth) {
        if (depth == 3 && expired())
            return;
        int c = path[depth];
        if (depth > 0 && mine(c) && cells[c].army >= 3) {
            int force = cells[c].army;
            float resistance = 0;
            ++stamp;
            for (int j = 0; j <= depth; ++j)
                onRoute[path[j]] = stamp;
            for (int j = depth - 1; j >= 0; --j) {
                int v = path[j], t = depth - j;
                int growth = cells[v].owner < 0 ? 0
                                                : (tick + t - 1) / 50 - tick / 50 +
                                                      (cells[v].kind ? ((tick + t - 1) / 2 - tick / 2) : 0);
                force += mine(v) ? cells[v].army + growth - 1 : -guard(v) - growth - 1;
                if (force <= 0)
                    break;
                for (int e : edges[v])
                    if (hostile(e) && onRoute[e] != stamp && cells[e].owner == cells[target].owner)
                        resistance = std::max(resistance, float(guard(e) - 1));
            }
            if (force > 0) {
                float backup = 0;
                for (int e : enemies)
                    if (onRoute[e] != stamp && cells[e].owner == cells[target].owner &&
                        md(e, target) <= depth && cells[e].army >= 3)
                        backup = std::max(backup, float(guard(e) - 1));
                resistance = std::max(resistance, backup) + std::max(0, depth - 2) * 2;
                float certainty = (force + 2.f) / (resistance + 5.f);
                if (certainty > 1.35f) {
                    float score = (live > 2 ? 125.f : 55.f) * std::min(1.6f, certainty) / (depth + 1);
                    bid({c, path[depth - 1], false}, score, target);
                }
            }
        }
        if (depth == 6)
            return;
        for (int v : edges[c]) {
            if (wall(v) || std::find(path, path + depth + 1, v) != path + depth + 1)
                continue;
            if (ownDist[v] > 6 - depth && !mine(v))
                continue;
            path[depth + 1] = v;
            finishPath(target, path, depth + 1);
        }
    }
    struct Change {
        int c = -1, owner = -3, army = 0;
    };
    struct Exchange {
        std::array<Change, 4> changes{};
        int count = 0, winner = -1;
    };
    Change stateAt(const Exchange &board, int c, int after = 0) const {
        Change r{c, cells[c].owner, guard(c)};
        for (int j = 0; j < board.count; ++j)
            if (board.changes[j].c == c) {
                r = board.changes[j];
                break;
            }
        if (after && r.owner >= 0) {
            int t = tick + 1;
            r.army += int(t % 50 == 0) + int(cells[c].kind != 0 && t % 2 == 0);
        }
        return r;
    }
    static void writeChange(Exchange &b, Change r) {
        for (int i = 0; i < b.count; ++i)
            if (b.changes[i].c == r.c) {
                b.changes[i] = r;
                return;
            }
        b.changes[b.count++] = r;
    }
    void resolveLocal(Exchange &b, Move m, int who) const {
        if (m.from < 0 || b.winner >= 0)
            return;
        auto a = stateAt(b, m.from), z = stateAt(b, m.to);
        if (a.owner != who || a.army < 2)
            return;
        int sent = m.half ? a.army / 2 : a.army - 1;
        a.army -= sent;
        if (z.owner == who)
            z.army += sent;
        else if (sent > z.army) {
            if (cells[z.c].kind == 2)
                b.winner = who;
            z.owner = who;
            z.army = sent - z.army;
        } else
            z.army -= sent;
        writeChange(b, a);
        writeChange(b, z);
    }
    Exchange clash(Move a, int attacker, Move d) const {
        Exchange b;
        if (a.from < 0) {
            resolveLocal(b, d, 1 - attacker);
            return b;
        }
        if (d.from < 0) {
            resolveLocal(b, a, attacker);
            return b;
        }
        bool first;
        if (a.to == d.from && d.to != a.from)
            first = true;
        else if (d.to == a.from && a.to != d.from)
            first = false;
        else if ((cells[a.to].owner == attacker) != (cells[d.to].owner == 1 - attacker))
            first = cells[a.to].owner == attacker;
        else if ((cells[a.to].kind == 2) != (cells[d.to].kind == 2))
            first = cells[d.to].kind == 2;
        else if (guard(a.from) != guard(d.from))
            first = guard(a.from) > guard(d.from);
        else
            first = attacker == tick % 2;
        if (first) {
            resolveLocal(b, a, attacker);
            resolveLocal(b, d, 1 - attacker);
        } else {
            resolveLocal(b, d, 1 - attacker);
            resolveLocal(b, a, attacker);
        }
        return b;
    }
    int oneTurnMargin(const Exchange &b, int king, int attacker) const {
        if (b.winner >= 0)
            return b.winner == attacker ? 100000 : -100000;
        int g = stateAt(b, king, 1).army, backup = 0, high = -100000;
        for (int z : edges[king])
            if (z >= 0) {
                auto u = stateAt(b, z, 1);
                if (u.owner == 1 - attacker)
                    backup = std::max(backup, u.army - 1);
            }
        for (int v : edges[king])
            if (v >= 0) {
                auto u = stateAt(b, v, 1);
                if (u.owner != attacker || u.army <= 1)
                    continue;
                int cut = backup;
                for (int z : edges[v])
                    if (z >= 0 && z != king) {
                        auto inter = stateAt(b, z, 1);
                        if (inter.owner == 1 - attacker)
                            cut = std::max(cut, inter.army - 1);
                    }
                high = std::max(high, u.army - 1 - g - cut);
            }
        return high;
    }
    void localCommands(int center, int who, bool attacking, std::vector<Move> &out) {
        out.clear();
        out.push_back({});
        for (int y = std::max(0, center / w - 3); y <= std::min(h - 1, center / w + 3); ++y)
            for (int x = std::max(0, center % w - 3); x <= std::min(w - 1, center % w + 3); ++x) {
                int c = y * w + x;
                if (md(c, center) > 3 || cells[c].owner != who || guard(c) < 2)
                    continue;
                if (who != seat && cells[c].seen + 2 < tick)
                    continue;
                for (int v : edges[c])
                    if (open(v) && md(v, center) <= 2 &&
                        (!attacking || md(v, center) <= 1 || cells[v].owner != who)) {
                        out.push_back({c, v, false});
                        if (c == center && cells[c].army >= 4)
                            out.push_back({c, v, true});
                    }
            }
    }
    std::vector<Move> attackers, replies;
    // Test two simultaneous move pairs near a capital. A second-turn attack
    // must survive both an immediate reinforcement and a hit on its source.
    void checkTwoTurns(int king, bool attack) {
        int who = attack ? seat : 1 - seat;
        bool nearby = false;
        for (int y = std::max(0, king / w - 2); y <= std::min(h - 1, king / w + 2); ++y)
            for (int x = std::max(0, king % w - 2); x <= std::min(w - 1, king % w + 2); ++x) {
                int c = y * w + x;
                if (md(c, king) <= 2 && cells[c].owner == who && guard(c) >= 3 &&
                    guard(c) + 6 >= guard(king) / 2)
                    nearby = true;
            }
        if (!nearby || expired())
            return;
        localCommands(king, who, true, attackers);
        localCommands(king, 1 - who, false, replies);
        if (attack) {
            int high = 2;
            Move choice;
            for (Move m : attackers)
                if (m.from >= 0) {
                    int worst = oneTurnMargin(clash(m, who, {}), king, who);
                    if (worst <= high)
                        continue;
                    for (Move d : replies) {
                        worst = std::min(worst, oneTurnMargin(clash(m, who, d), king, who));
                        if (worst <= high)
                            break;
                    }
                    if (worst > high) {
                        high = worst;
                        choice = m;
                    }
                }
            if (choice.from >= 0)
                bid(choice, 9950.f, king);
            return;
        }
        auto risk = [&](Move d, int limit) {
            int worst = -100000;
            for (Move m : attackers) {
                worst = std::max(worst, oneTurnMargin(clash(m, who, d), king, who));
                if (worst >= limit)
                    break;
            }
            return worst;
        };
        Move original = best.move;
        if (original.from >= 0 && md(original.from, king) > 3 && md(original.to, king) > 2)
            original = {};
        int worst = risk(original, 100001);
        if (worst <= 0 || best.value >= 9900.f)
            return;
        Move choice;
        int low = worst;
        for (Move d : replies)
            if (d.from >= 0) {
                int value = risk(d, low);
                if (value < low) {
                    low = value;
                    choice = d;
                    if (low <= 0)
                        break;
                }
            }
        if (choice.from >= 0 && low < worst)
            bid(choice, 9800.f, king);
    }
    void finish() {
        for (int k : kings) {
            if (ownDist[k] > 6 || expired())
                continue;
            int path[7] = {k};
            finishPath(k, path, 0);
        }
    }

  public:
    explicit Player(int p, std::uint64_t seed) : seat(p), stream(seed ? seed : 1) {}
    std::int64_t act(const ArenaObservation *o) {
        obs = o;
        if (!o || !o->alive[seat] || o->width <= 0 || o->height <= 0 ||
            std::int64_t(o->width) * o->height > capacity || o->general_pos < 0 ||
            o->general_pos >= o->width * o->height)
            return 0;
        deadline = Clock::now() + std::chrono::milliseconds(4);
        if (!n)
            geometry();
        observe();
        updateDuelPressure();
        if (obs->players == 2 && tick >= 210)
            pressAttack = true;
        buildPrior();
        if (useBook && tick < 50) {
            if (!enemies.empty())
                useBook = false;
            else if (tick < 14) {
                if (tick == 0 || (tick % 5 == 0 && priorReady))
                    locate();
                for (int i = 0; i < 30 && (i == 0 || !expired()); ++i)
                    openingSample();
                return 0;
            } else {
                Move m = book[tick];
                if (m.from < 0)
                    return 0;
                if (mine(m.from) && cells[m.from].army > 1 && open(m.to) &&
                    (mine(m.to) || cells[m.from].army - 1 > guard(m.to)))
                    return arena_move(m.from, dir(m));
                useBook = false;
            }
        }
        best = {};
        defend();
        locate();
        inferTerritory();
        if (obs->players == 2 && tick >= 70)
            for (int c = 0; c < n; ++c)
                if (!obs->visible[c]) {
                    cellReward[c] += 1.9f * occupancy[c];
                    forceValue[c] += 0.085f * occupancy[c];
                }
        savingForCity = live > 2 && tick >= 50 && tick < 260 && obs->total_cities[seat] < 3 &&
                        !structures.empty() && contact[home] >= 5 && reserve < 7;
        priceTroops();
        expansion();
        if (pressAttack)
            best.value *= 0.55f;
        campaign();
        combat();
        finish();
        if (tick >= 45 && !expired())
            operations();
        for (int k : kings)
            for (int c : edges[k])
                if (mine(c) && cells[c].army - 1 > cells[k].army)
                    bid({c, k, false}, 10000, k);
        if (obs->players == 2 && !kings.empty())
            checkTwoTurns(kings.front(), true);
        if (obs->players == 2)
            checkTwoTurns(home, false);
        auto m = best.move;
        if (m.from < 0)
            return 0;
        if (!active.moves.empty()) {
            Move a = active.moves.front();
            if (a.from == m.from && a.to == m.to && a.half == m.half)
                active.moves.erase(active.moves.begin());
            else if (tick - active.started > 14)
                active = {};
        }
        lastFrom = m.from;
        lastTo = m.to;
        lastGoal = best.objective;
        return arena_move(m.from, dir(m), m.half);
    }
};
} // namespace meridian
extern "C" void *arena_create(int p, std::uint64_t seed, const char *) {
    return new meridian::Player(p, seed);
}
extern "C" std::int64_t arena_act(void *state, const ArenaObservation *o) {
    return static_cast<meridian::Player *>(state)->act(o);
}
extern "C" void arena_destroy(void *state) { delete static_cast<meridian::Player *>(state); }
