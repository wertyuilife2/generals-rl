"""All schedule modes replay the same GoR / Codeforces rating updates."""
from generals_arena.competitions.rating import POLICY
from generals_arena.competitions.rating import FFA_MEAN
from generals_arena.competitions.rating import initial_rating
from generals_arena.competitions.rating import rating_mode
from generals_arena.competitions.rating import ranks
from generals_arena.competitions.rating import score
from generals_arena.competitions.rating import validate_mode
from generals_arena.matches.results import aggregate
from generals_arena.competitions.uncertainty import RatingUncertainty


def score_results(entrants, results, *, mode, reference=None):
    """Rate a deterministic schedule from fresh, equal initial raw ratings.

    Opponent policies and schedule may be fixed; every participating strategy's
    rating evolves normally. GoR display offsets never enter its probability
    model. FFA retains its original mean of 1000 without an anchored policy.
    """
    mode = validate_mode(mode)
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
        changes = score(ratings, result, counts)
        result.update(rating_policy=POLICY, rating_mode=mode, rating_changes=changes, ranks=ranks(result))
        for key, change in changes.items():
            ratings[key], counts[key] = change['after'], change['rated_games']
        uncertainty.add(result)
    offset = ratings[reference] if reference is not None else 0.
    ranking = [{'entrant_id': key, 'raw_elo': ratings[key], 'elo': ratings[key] - offset,
                'anchor': key == reference, **aggregate(rows, key)} for key in ids]
    report = {'rating_policy': POLICY, 'rating_mode': mode,
              'rating_model': 'egf-gor' if mode == '1v1' else 'codeforces',
              'anchor': reference, 'rating_offset': offset,
              'rating_mean': FFA_MEAN if mode == 'ffa' else None,
              'ranking': ranking, 'games': len(rows)}
    uncertainty.annotate(report)
    ranking.sort(key=lambda row: (-row['elo'], row['entrant_id']))
    return {**report, 'raw_ratings': ratings}
