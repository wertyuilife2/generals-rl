"""Official-server games using the same policy packages as local matches."""
from pathlib import Path

from generals import MapConfig
from generals.backends.online import OnlineBackend
from generals.runtime.online import run_online
from generals_arena.matches.config import MatchConfig
from generals_arena.policies.entries import prepare_entrant, load_controller
from generals_arena.util import write_json


def play_online(spec, options, *, games=1, output=None, seed=42, max_decisions=None,
                backend_factory=OnlineBackend):
    """Record completed and failed attempts without modifying offline ratings.

    A failure stops this explicit series; no hidden retries join more games.
    Source/bundle selection is handled by the shared Arena policy registry.
    """
    if type(games) is not int or games < 1:
        raise ValueError('games must be positive')
    if games > 1 and output is None:
        raise ValueError('multiple online games require an output directory')
    entrant = prepare_entrant(spec)
    if entrant['kind'] == 'baseline':
        raise ValueError('path-following baselines require authoritative move receipts unavailable online')
    def factory(obs):
        if entrant['kind'] == 'package':
            from generals_arena.policies.artifacts import validate_bundle, capabilities_supports
            caps = validate_bundle(entrant['factory_kwargs']['bundle'])['manifest']['capabilities']
            if not capabilities_supports(caps, obs.width, obs.height, len(obs.alive)):
                raise ValueError('online map/player count is outside this policy package capabilities')
        return load_controller(entrant, MatchConfig(MapConfig(obs.width, obs.height, len(obs.alive))))

    output = None if output is None else Path(output)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    report = {'policy': entrant['name'], 'version': entrant['version'], 'requested_games': games,
              'completed_games': 0, 'wins': 0, 'losses': 0, 'status': 'running', 'games': []}
    def save():
        if output is not None:
            write_json(output / 'report.json', report)
    save()
    try:
        for index in range(games):
            if output is not None and (output / 'stop-after-game').exists():
                report['status'] = 'paused'
                break
            record = output / f'game-{index + 1:03d}.jsonl' if output else None
            row = {'attempt': index + 1, 'seed': seed + index}
            report['games'].append(row)
            save()
            try:
                result = run_online(backend_factory(**options, record=record), factory,
                                    seed=seed + index, max_decisions=max_decisions)
                row.update(result)
            except BaseException as exc:
                # Credentials can appear in a third-party transport exception.
                row['error'] = type(exc).__name__
                report['status'] = 'failed'
                raise
            if not result['done']:
                report['status'] = 'stopped'
                break
            report['completed_games'] += 1
            report['wins'] += int(result['reason'] == 'won')
            report['losses'] += int(result['reason'] == 'lost')
            save()
        else:
            report['status'] = 'completed'
    finally:
        save()
    return report
