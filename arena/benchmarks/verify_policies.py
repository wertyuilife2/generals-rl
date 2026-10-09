"""Re-simulate archived baseline matches with current native policies and engine.

Unlike verify_league (recorded actions only), this also verifies every policy's
decision. The original database is read only; source identity changes are
intentional in this comparison, never accepted for continuing an old league.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

from generals_arena.league.game import RecordedGame
from generals_arena.league.store import LeagueStore
from generals_baselines import create_controller


def verify(item):
    directory, key = item
    store = LeagueStore(directory)
    result, blob = store.replay(key)
    entries = store.entrants()
    policies = []
    for entrant_id in result['entrant_ids']:
        entrant = entries[entrant_id]
        if entrant['kind'] != 'baseline':
            raise ValueError('this comparison requires baseline-only matches')
        policies.append(create_controller(entrant['factory_kwargs']['kind']))
    game = RecordedGame(result, store.config, policies)
    try:
        while not game.done:
            game.advance()
        expected = hashlib.sha256(gzip.decompress(blob)).hexdigest()
        actual = hashlib.sha256(gzip.decompress(game.replay_blob())).hexdigest()
        if actual != expected:
            raise AssertionError(f'action/state trajectory diverged: {key}')
        current = game.result()
        for field in ('final_stats', 'elimination_ticks', 'invalid_actions', 'winner_seat', 'ticks'):
            assert current[field] == result[field], (key, field)
        return game.state.tick
    finally:
        game.engine.close()
        for controller in policies:
            controller.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    store = LeagueStore(args.directory)
    with store.connect() as db:
        ids = [row[0] for row in db.execute("SELECT id FROM games WHERE status='complete' AND json_extract(plan,'$.human')=0 ORDER BY sequence")]
    # Compile before timing; each worker gets independent native policy/RNG state.
    from generals.backends.cpu.env import _library
    _library()
    probe = create_controller('random'); probe.close()
    start = perf_counter()
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        ticks = sum(pool.map(verify, ((args.directory, key) for key in ids)))
    report = {'verified_games': len(ids), 'verified_ticks': ticks, 'workers': args.workers,
              'seconds': perf_counter()-start, 'all_actions_and_states_identical': True,
              'engine': 'env native CPU', 'policies': 'C++ with NumPy BitGenerator'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
