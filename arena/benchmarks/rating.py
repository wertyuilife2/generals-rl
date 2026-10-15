"""Compare chronological prediction and late volatility on identical outcomes.

Read-only: no simulations, database writes, or saved copies of old scores.
"""
import argparse
from collections import Counter
import json
import math

import numpy as np

from generals_arena.league.rating import expected
from generals_arena.league.mode_rating import initial_rating, probability
from generals_arena.league.modes import rating_mode
from generals_arena.league.rating_policy import CUMULATIVE_POLICY, LEGACY_POLICY, CALIBRATED_POLICY, FIXED_GOR_POLICY, POLICY, score
from generals_arena.league.rank_model import comparisons, rotation_key
from generals_arena.league.store import LeagueStore
from generals_arena.util import write_json


def anchor_calibration(ratings, evidence, anchor, predict=expected):
    """Check the absolute origin separately from relative prediction accuracy."""
    pairs = [(right if left == anchor else left, total, total-wins if left == anchor else wins)
             for (left, right), (total, wins) in evidence.items() if anchor in (left,right)]
    weight = sum(total for _,total,_ in pairs)
    if not weight:
        return None
    observed = sum(wins for _,_,wins in pairs)/weight

    def predicted(offset):
        return sum(total*predict(ratings.get(key,0)+offset,0) for key,total,_ in pairs)/weight

    offset = None
    if 0 < observed < 1:
        lower, upper = -10000., 10000.
        for _ in range(60):
            mid = (lower+upper)/2
            if predicted(mid) < observed:
                lower = mid
            else:
                upper = mid
        offset = (lower+upper)/2
    return dict(effective_comparisons=weight, observed_pair_score=observed,
                predicted_pair_score=predicted(0), required_offset_elo=offset,
                opponents=[dict(entrant_id=key, weight=total, observed_pair_score=wins/total,
                                predicted_pair_score=predict(ratings.get(key,0),0))
                           for key,total,wins in pairs])


def evaluate(records, policy, anchor):
    ratings, counts, evidence = {}, Counter(), {}
    changes, loss, weight, violations = [], 0., 0., 0
    mode = rating_mode(records[0]) if records else '1v1'
    modern = policy in (FIXED_GOR_POLICY, POLICY)
    relative = policy == POLICY and mode == '1v1'
    initial = initial_rating(mode) if modern else 0.
    predict = (lambda a,b: probability(mode,a,b)) if modern else expected
    for result in records:
        for left, right, amount, outcome in comparisons(result):
            p = max(1e-12, min(1-1e-12, predict(ratings.get(left,initial), ratings.get(right,initial))))
            loss -= amount*(outcome*math.log(p)+(1-outcome)*math.log1p(-p))
            weight += amount
        update = score(ratings, {**result,'rating_policy':policy}, anchor, 32, counts, evidence)
        if len(result['entrant_ids']) == len(set(result['entrant_ids'])) == 2:
            for left, right, _, outcome in comparisons(result):
                if outcome in (0., 1.):
                    winner, loser = (left,right) if outcome == 1 else (right,left)
                    violations += int(update[winner]['delta'] < -1e-8 or update[loser]['delta'] > 1e-8)
        for key, change in update.items():
            if (key != anchor or relative or (modern and mode == 'ffa')) and counts[key] >= 200 and change.get('distinct_opponents',0):
                changes.append(abs(change['delta']))
            ratings[key], counts[key] = change['after'], change['rated_games']
    offset = ratings.get(anchor,0.) if relative else 0.
    return dict(prequential_log_loss=loss/weight if weight else None,
                late_update_mean=float(np.mean(changes)) if changes else None,
                late_update_p95=float(np.percentile(changes,95)) if changes else None,
                late_update_max=max(changes, default=None), late_updates=len(changes),
                duel_direction_violations=violations, ratings={key:value-offset for key,value in ratings.items()},
                raw_ratings=ratings, rating_offset=offset,
                anchor_calibration=None if relative or (modern and mode == 'ffa') else anchor_calibration(ratings,evidence,anchor,predict))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--output')
    args = parser.parse_args()
    store = LeagueStore(args.directory)
    with store.connect() as db:
        rows = [(row[0],json.loads(row[1])) for row in db.execute(
            "SELECT job_id,result FROM games WHERE status='complete' ORDER BY sequence")]
    sizes = Counter(rotation_key(result,job) for job,result in rows)
    records = [{**result,'rating_group_size':result.get('rating_group_size',sizes[rotation_key(result,job)])}
               for job,result in rows]
    from generals_arena.league.modes import MODES, rating_mode
    report = {'games':len(records), 'modes':{}}
    for mode in MODES:
        matches = [result for result in records if rating_mode(result) == mode]
        report['modes'][mode] = dict(games=len(matches), provisional_cutoff=200,
                      legacy=evaluate(matches,LEGACY_POLICY,store.anchor),
                      cumulative=evaluate(matches,CUMULATIVE_POLICY,store.anchor),
                      previous=evaluate(matches,CALIBRATED_POLICY,store.anchor),
                      fixed_gor=evaluate(matches,FIXED_GOR_POLICY,store.anchor),
                      current=evaluate(matches,POLICY,store.anchor))
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
