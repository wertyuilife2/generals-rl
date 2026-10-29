"""Shared rating reports: online league updates or jointly fitted GoR evaluation."""
from generals_arena.competitions.rating import POLICY
from generals_arena.competitions.rating import FFA_MEAN
from generals_arena.competitions.rating import initial_rating
from generals_arena.competitions.rating import rating_mode
from generals_arena.competitions.rating import ranks
from generals_arena.competitions.rating import score
from generals_arena.competitions.rating import validate_mode
from generals_arena.matches.results import aggregate
from generals_arena.matches.results import win_matrix
from generals_arena.competitions.uncertainty import RatingUncertainty


def score_results(entrants, results, *, mode, reference=None, fit=False, fixed_ratings=None):
    """Rate a schedule online, or fit all duel results on a fixed reference scale.

    Online mode starts from equal ratings and updates every participant.
    Fitted mode holds the supplied calibration fixed. Display offsets never enter the probability
    model. FFA retains its original mean of 1000 without an anchored policy.
    """
    mode = validate_mode(mode)
    fit = fit or fixed_ratings is not None
    if fit and mode != '1v1':
        raise ValueError('calibrated GoR evaluation requires a 1v1 profile')
    ids = [entry['entrant_id'] if isinstance(entry, dict) else entry for entry in entrants]
    if len(set(ids)) != len(ids) or len(ids) < 2:
        raise ValueError('provide at least two distinct rating participants')
    if reference is not None and (reference not in ids or mode != '1v1'):
        raise ValueError('a display reference must be a participating 1v1 policy')
    ratings = dict.fromkeys(ids, initial_rating(mode))
    counts = dict.fromkeys(ids, 0)
    uncertainty = RatingUncertainty(mode)
    seen = set()
    rows = list(results)
    for result in rows:
        if result.get('policy_error') or result.get('reason') == 'policy_error':
            raise ValueError('failed matches cannot become rating evidence')
        if result['game_id'] in seen:
            raise ValueError('duplicate game cannot become additional rating evidence')
        if rating_mode(result) != mode or set(result['entrant_ids']) - ratings.keys():
            raise ValueError('match does not belong to this rating pool and roster')
        seen.add(result['game_id'])
        result.update(rating_policy=POLICY, rating_mode=mode, ranks=ranks(result))
        if not fit:
            changes = score(ratings, result, counts)
            result['rating_changes'] = changes
            for key, change in changes.items():
                ratings[key], counts[key] = change['after'], change['rated_games']
        else:
            result.pop('rating_changes', None)
        uncertainty.add(result)
    fit_info = {}
    if fit:
        from generals_arena.competitions.calibration import fit_ratings
        ratings, intervals, fit_info = fit_ratings(ids, rows, reference, fixed_ratings)
    offset = ratings[reference] if reference is not None else 0.
    ranking = [{'entrant_id': key, 'raw_elo': ratings[key], 'elo': ratings[key] - offset,
                'anchor': key == reference, **aggregate(rows, key)} for key in ids]
    report = {'rating_policy': POLICY, 'rating_mode': mode,
              'rating_model': 'egf-gor' if mode == '1v1' else 'codeforces',
              'anchor': reference, 'rating_offset': offset,
              'rating_mean': FFA_MEAN if mode == 'ffa' else None,
              'ranking': ranking, 'games': len(rows), 'win_matrix': win_matrix(rows), **fit_info}
    if fit:
        for row in ranking:
            row.update(elo_ci=intervals[row['entrant_id']], ci_blocks=uncertainty.blocks[row['entrant_id']])
    else:
        uncertainty.annotate(report)
    ranking.sort(key=lambda row: (-row['elo'], row['entrant_id']))
    return {**report, 'raw_ratings': ratings}
