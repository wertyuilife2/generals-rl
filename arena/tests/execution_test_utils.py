"""Deterministic game evidence excludes wall-clock measurements."""
def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if k != 'decision_timing'}
    if isinstance(value, (list, tuple)):
        return [without_timing(v) for v in value]
    return value
