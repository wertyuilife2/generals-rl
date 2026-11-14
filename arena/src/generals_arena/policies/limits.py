"""Per-policy execution limits, independent of policy identity and ratings."""
from generals_arena.util import integer


def execution_limits(entry, budget=None):
    limits = dict(decision_budget_ms=50 if entry.get('kind') == 'package' else 10,
                  decision_timeout_ms=None if entry.get('kind') == 'baseline' else 100)
    supplied = entry.get('execution', {})
    if not isinstance(supplied, dict) or set(supplied) - limits.keys():
        raise ValueError('execution requires decision_budget_ms and/or decision_timeout_ms')
    limits.update(supplied)
    if budget is not None:
        limits['decision_budget_ms'] = budget
    integer('decision_budget_ms', limits['decision_budget_ms'], 1)
    timeout = limits['decision_timeout_ms']
    if timeout is not None:
        integer('decision_timeout_ms', timeout, 1)
        if timeout < limits['decision_budget_ms']:
            raise ValueError('hard timeout must be at least the decision budget, or off')
    return limits


def duration(value, *, allow_off=False):
    if allow_off and value == 'off':
        return None
    value = str(value).removesuffix('ms')
    if not value.isdigit() or int(value) < 1:
        raise ValueError('duration must be positive integer milliseconds' + (' or off' if allow_off else ''))
    return int(value)


def configure(store, selector, changes):
    entries = store.entrants()
    matches = [e for e in entries.values() if e['entrant_id'] == selector or e['name'] == selector.lstrip('@')]
    if len(matches) != 1 or matches[0]['kind'] == 'human':
        raise ValueError('select exactly one registered AI by name or entrant ID')
    entry = matches[0]
    entry['execution'] = execution_limits({**entry, 'execution': {**entry.get('execution', {}), **changes}})
    from generals_arena.util import canonical
    with store.connect(write=True) as db:
        db.execute('UPDATE entrants SET data=? WHERE id=?', (canonical(entry), entry['entrant_id']))
    return dict(name=entry['name'], entrant_id=entry['entrant_id'], execution=entry['execution'])


def freeze_plan(plan, entries, budget=None):
    plan = dict(plan)
    values = plan.get('decision_limits')
    if values is None:
        values = [execution_limits(entries[key], budget if entries[key].get('kind') != 'package' else None) for key in plan['entrant_ids']]
    if len(values) != len(plan['entrant_ids']):
        raise ValueError('each seat needs its own decision limits')
    plan['decision_limits'] = [execution_limits({'execution': value}) for value in values]
    return plan
