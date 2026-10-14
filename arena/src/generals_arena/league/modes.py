"""Rating pools are determined by seats, never by a caller-supplied label."""
MODES = ('1v1', 'ffa')


def rating_mode(result):
    return '1v1' if len(result['entrant_ids']) == 2 else 'ffa'


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError('rating mode must be 1v1 or ffa')
    return mode


def mode_predicate(mode):
    validate_mode(mode)
    return "json_array_length(plan, '$.entrant_ids')" + ('=2' if mode == '1v1' else '>2')


def split_budget(games, counts):
    if 2 not in counts or not any(p > 2 for p in counts):
        raise ValueError('commit requires both 1v1 and FFA player counts')
    if games < 2:
        raise ValueError('split evaluation needs at least two games')
    return {'1v1': games // 2, 'ffa': games - games // 2}
