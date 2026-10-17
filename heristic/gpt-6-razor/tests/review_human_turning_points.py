"""Audit the recorded v7 losses, with score changes separated from outcomes.

Offline legal-observation review only. Neither opponent intent nor an
alternative win is inferred from a frozen historical continuation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from review_recorded_campaigns import load_frames, describe_order


def row(obs, actions):
    p = obs.player_id
    enemy = 1 - p
    columns = sorted(
        (c for c in range(obs.size)
         if obs.visible.flat[c] and obs.owner.flat[c] == enemy),
        key=lambda c: -int(obs.army.flat[c]))
    return dict(tick=obs.tick, own_army=int(obs.stats.army[p]),
                enemy_army=int(obs.stats.army[enemy]),
                own_land=int(obs.stats.territory[p]),
                enemy_land=int(obs.stats.territory[enemy]),
                own_cities=int(obs.stats.cities[p]),
                home_army=int(obs.army.flat[obs.general_pos]),
                visible_enemy_columns=[dict(cell=c, army=int(obs.army.flat[c]))
                                       for c in columns[:3]],
                submitted_orders=[describe_order(obs, a) for a in actions])


def cities_and_discovery(frames):
    previous, events = {}, []
    first_king = None
    for obs, _ in frames:
        for c in range(obs.size):
            if not obs.visible.flat[c]:
                continue
            owner, structure = int(obs.owner.flat[c]), int(obs.structure.flat[c])
            if structure == 2 and owner >= 0 and owner != obs.player_id:
                if first_king is None:
                    first_king = obs.tick
            if structure != 1:
                continue
            if c in previous and previous[c][0] != owner:
                old_owner, seen_tick = previous[c]
                events.append(dict(tick=obs.tick, city=c, previous_owner=old_owner,
                                   owner=owner, army=int(obs.army.flat[c]),
                                   previous_seen_tick=seen_tick,
                                   consecutive_observations=seen_tick + 1 == obs.tick))
            previous[c] = (owner, obs.tick)
    return dict(first_enemy_general_visible=first_king,
                observed_city_owner_changes=events)


def decomposition(frames, start, end):
    by_tick = {o.tick: o for o, _ in frames}
    p = by_tick[start].player_id
    gap = lambda o: int(o.stats.army[p] - o.stats.army[1-p])
    totals, hist, neutral = Counter(), Counter(), []
    for (a, _), (b, _) in zip(frames, frames[1:]):
        if not start < b.tick <= end:
            continue
        phase = ('observation_gap' if b.tick-a.tick != 1 else
                 'land_growth_boundary' if b.tick % 50 == 0 else
                 'ordinary_growth_boundary' if b.tick % 2 == 0 else
                 'between_growth_boundaries')
        delta = gap(b) - gap(a)
        totals[phase] += delta
        hist[f'{phase}:{delta}'] += 1
        for c in range(b.size):
            if a.owner.flat[c] == -2 and b.owner.flat[c] == p:
                neutral.append(dict(tick=b.tick, city=c,
                                    previous_visible_garrison=int(a.army.flat[c])))
    change = gap(by_tick[end]) - gap(by_tick[start])
    assert sum(totals.values()) == change
    return dict(start=start, end=end, own_minus_enemy_before=gap(by_tick[start]),
                own_minus_enemy_after=gap(by_tick[end]), changes_by_phase=totals,
                histogram=hist, observed_neutral_city_captures=neutral)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path(
        'outputs/online/razor-v7-main-10-20261015T161905Z/report.json'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    games = []
    for entry in report['games']:
        if entry.get('result', {}).get('reason') != 'lost':
            continue
        record = args.report.parent / Path(entry['record']).name
        frames = load_frames(record)
        item = dict(game=entry['attempt'], record=str(record),
                    record_sha256=hashlib.sha256(record.read_bytes()).hexdigest(),
                    players=entry['usernames'], seat=entry['player_id'],
                    replay_url=entry['result']['replay_url'],
                    observation_count=len(frames), last_tick=frames[-1][0].tick,
                    **cities_and_discovery(frames))
        if entry['attempt'] == 6:
            item['score_windows'] = [decomposition(frames, a, b) for a, b in
                [(400, 442), (442, 550), (550, 800), (600, 650),
                 (650, 700), (700, 750), (750, 800)]]
            item['selected_actual_frames'] = [row(o, actions) for o, actions in frames
                if o.tick in (400,404,414,421,427,432,436,442,444,463,489,499,550,800)
                or 690 <= o.tick <= 722]
        games.append(item)
    result = dict(scope='Recorded legal views and submitted orders. Opponent human '
                  'identity is not independently verified for main-site usernames. '
                  'City changes with observation gaps are not exact capture times. '
                  'Growth-frame changes can include investment; no hidden cities are read. '
                  'No policy is run and no counterfactual outcome is claimed.',
                  policy=report['policy'], games=games)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(games=len(games),
        losses_without_discovering_general=sum(g['first_enemy_general_visible'] is None
                                              for g in games), output=str(args.output))))


if __name__ == '__main__':
    main()
