// Candidate ONLY: deterministic original 9x9 AggressiveAI candidate ranking.
// Random selection, remembered state, path planning and move feedback stay Python.
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <vector>

struct Candidate { int source, target, strategy; double score; };
static int distance(int a, int b) {
    return std::abs(a / 9 - b / 9) + std::abs(a % 9 - b % 9);
}
static int adjacent(int cell, int direction) {
    const int y = cell / 9, x = cell % 9;
    if (direction == 0) return y ? cell - 9 : -1;
    if (direction == 1) return x < 8 ? cell + 1 : -1;
    if (direction == 2) return y < 8 ? cell + 9 : -1;
    return x ? cell - 1 : -1;
}

extern "C" int prepare_sources(const std::int16_t *owner, const std::int64_t *army,
        const std::int16_t *memory_owner, const std::uint8_t *terrain, int pid,
        int *sources, std::int64_t *frontiers, std::int64_t *threats) {
    std::vector<std::pair<int, double>> ranked;
    for (int cell = 0; cell < 81; ++cell) {
        frontiers[cell] = threats[cell] = 0;
        if (owner[cell] != pid) continue;
        for (int direction = 0; direction < 4; ++direction) {
            const int target = adjacent(cell, direction);
            if (target < 0 || terrain[target] == 1) continue;
            frontiers[cell] += memory_owner[target] != pid;
            if (owner[target] >= 0 && owner[target] != pid)
                threats[cell] += std::max<std::int64_t>(0, army[target]);
        }
        if (army[cell] > 1) {
            const double score = static_cast<double>(army[cell])
                                 * (1.0 + 0.25 * static_cast<bool>(frontiers[cell]));
            ranked.emplace_back(cell, score);
        }
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    const int count = std::min<int>(6, ranked.size());
    for (int index = 0; index < count; ++index) sources[index] = ranked[index].first;
    return count;
}

extern "C" int build_candidates(const std::int16_t *owner, const std::int64_t *army,
        const std::int16_t *memory_owner, const std::int64_t *memory_army,
        const std::uint8_t *terrain, const std::uint8_t *structure, int pid,
        const int *sources, int source_count, const std::int64_t *frontiers,
        const std::int64_t *threats, Candidate *output) {
    int count = 0;
    for (int source_index = 0; source_index < source_count; ++source_index) {
        const int source = sources[source_index];
        // Original ordered weights: merge, explore, attack.
        for (int strategy = 0; strategy < 3; ++strategy) {
            std::vector<Candidate> local;
            if (strategy == 0) {
                for (int target = 0; target < 81; ++target) {
                    if (owner[target] != pid || target == source) continue;
                    if (army[target] < army[source]
                            || (army[target] == army[source] && target >= source)) continue;
                    if (structure[source] == 2 && threats[source] > army[source] / 2) continue;
                    double value = 3.0 + std::min(static_cast<double>(army[target]), 100.0) * 0.06
                                   + frontiers[target];
                    if (threats[target]) {
                        value += std::min<std::int64_t>(threats[target], 100) * 0.15;
                        if (structure[target] == 2) value += 20.0;
                    }
                    local.push_back({source, target, strategy, value / (distance(source, target) + 1)});
                }
            } else if (strategy == 1) {
                std::vector<int> nearest;
                for (int target = 0; target < 81; ++target)
                    if ((memory_owner[target] == -1 || memory_owner[target] == -3)
                            && terrain[target] != 1) nearest.push_back(target);
                std::sort(nearest.begin(), nearest.end(), [source](int a, int b) {
                    const int da = distance(source, a), db = distance(source, b);
                    return da != db ? da < db : a < b;
                });
                if (nearest.size() > 16) nearest.resize(16);
                for (int target : nearest) {
                    const int d = distance(source, target);
                    if (d >= army[source]) continue;
                    int unknown = 0;
                    for (int direction = 0; direction < 4; ++direction) {
                        const int neighbor = adjacent(target, direction);
                        if (neighbor >= 0 && terrain[neighbor] != 1) unknown += terrain[neighbor] == 255;
                    }
                    const double score = (5.0 + 0.4 * unknown) / (d + 1);
                    local.push_back({source, target, strategy, score});
                }
            } else {
                for (int target = 0; target < 81; ++target) {
                    if (memory_owner[target] < 0 || memory_owner[target] == pid) continue;
                    const int d = distance(source, target);
                    const auto defense = std::max<std::int64_t>(0, memory_army[target]);
                    if (army[source] - d <= defense) continue;
                    const double value = structure[target] == 2 ? 50.0 : structure[target] == 1 ? 18.0 : 8.0;
                    const double score = value / (d + 1)
                        + std::min(static_cast<double>(army[source] - defense), 100.0) * 0.01;
                    local.push_back({source, target, strategy, score});
                }
            }
            std::sort(local.begin(), local.end(), [](const Candidate &a, const Candidate &b) {
                return a.score != b.score ? a.score > b.score : a.target < b.target;
            });
            const int keep = std::min<int>(4, local.size());
            for (int i = 0; i < keep; ++i) output[count++] = local[i];
        }
    }
    return count;
}
