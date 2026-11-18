// Meridian v1 -- a single-threaded, observation-only Generals policy.
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
    int available(int c) const {
        return std::max(0, cells[c].army - (c == home ? std::max(1, reserve) : 1));
    }
    bool wall(int c) const { return c < 0 || cells[c].blocked; }
    int guard(int c) const {
        auto &v = cells[c];
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
        if (savingForCity && cells[m.from].kind && cells[m.from].army < 54 && target != home &&
            target >= 0 && cells[target].kind == 0 && !(hostile(m.to) && cells[m.to].kind))
            value -= 3.4f;
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
    void observe() {
        int oldHome = home;
        home = obs->general_pos;
        tick = int(obs->tick);
        if (oldHome >= 0 && oldHome != home) {
            for (int c = 0; c < n; ++c)
                if (cells[c].kind == 2 && !obs->visible[c])
                    cells[c] = Cell{};
            lastGoal = -1;
            useBook = false;
            active = {};
            focal = -1;
        }
        live = 0;
        strongestEnemy = 0;
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
    void defend() {
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
            if (cells[c].army > 1) {
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
    std::array<float, capacity> likelihood{}, information{}, forceValue;
    std::array<int, capacity> landmarkDistance[3];
    // Estimate the value of revealing a cell from distinct early contacts and
    // the still-unseen spawn region. Observed capitals override the estimate.
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
                    prob *= 1.f / (1 + .075f * std::abs(dist - 4));
                }
                if (!enemies.empty())
                    prob *= 1.f / (1 + .24f * contact[c]);
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
                    if (cells[v].seen < 0)
                        loot += .25f;
                    if (kings.empty() && cells[v].seen < 0 && focalDist[v] <= 6)
                        loot += .65f;
                }
                float delivery = std::min(250, force) * forceValue[v] -
                                 std::min(250, prev.initial) * forceValue[s] - price;
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
                    if (hostile(v) && v != target &&
                        std::find(march.begin(), march.end(), v) == march.end())
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
                    standing = std::clamp((surplus + 3.f) / (need + 6.f), .01f, 1.30f);
                    standing *= standing;
                } else if (style == Objective::City && contact[target] <= 3)
                    standing = std::clamp((surplus + 6.f) / (intercept + 12.f), .10f, 1.f);
                float value = (worth * standing + loot) / (turns + 4.f);
                if (style == Objective::City && !hostile(target) && tick < 350) {
                    float usable = std::max(0.f, float(obs->total_army[seat] - obs->total_land[seat]));
                    value *=
                        std::clamp((usable - guard(target)) / (.70f * guard(target) + 10.f), .35f, 1.4f);
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
        float value =
            std::clamp(19.f + 15.f * (remainingArmyRatio - .80f) + .022f * (tick - 120), 11.f, 38.f);
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
            for (int c : kings)
                if (!expired())
                    expedition(c, live > 2 ? 210.f : 80.f, Objective::Capital);
            if (kings.empty() && focal >= 0 && !expired() && tick >= (live > 2 ? 75 : 150))
                expedition(focal, live > 2 ? 52.f : 26.f, Objective::Scout);
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
                expedition(c, investment(c), Objective::City);
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
                                float score = (draw() % 1024) / 300.f + .16f * std::min(20, homeDist[v]) +
                                              exits * .21f;
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
        if (useBook && tick < 50) {
            if (!enemies.empty())
                useBook = false;
            else if (tick < 14) {
                if (tick == 0)
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
        savingForCity = live > 2 && tick >= 50 && tick < 260 && obs->total_cities[seat] < 3 &&
                        !structures.empty() && contact[home] >= 5 && reserve < 7;
        expansion();
        campaign();
        combat();
        finish();
        if (tick >= 45 && !expired())
            operations();
        for (int k : kings)
            for (int c : edges[k])
                if (mine(c) && cells[c].army - 1 > cells[k].army)
                    bid({c, k, false}, 10000, k);
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
