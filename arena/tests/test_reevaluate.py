"""A fresh evaluation retains identities but no previous official scores."""
import importlib.util
from pathlib import Path

from generals_arena.league.config import LeagueConfig
from generals_arena.league.service import run_jobs
from generals_arena.league.store import LeagueStore


def test_reset_without_backup_and_resume_reuses_registered_jobs(tmp_path):
    spec = importlib.util.spec_from_file_location('reevaluate',
        Path(__file__).parents[1]/'benchmarks'/'reevaluate_league.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    store = LeagueStore.create(tmp_path/'league', LeagueConfig(max_ticks=2))
    bot, job = store.add({'name':'bot','kind':'baseline','policy':'aggressive'}, games=2)
    account = store.account('human')
    assert run_jobs(store) == 2
    report = store.directory/'commits'/'dummy'/'publication.json'
    report.parent.mkdir(parents=True)
    report.write_text('{}')
    module.reset(store, 3)
    assert not report.exists()
    assert store.leaderboard()['completed_games'] == 0
    assert store.ratings() == {}
    assert store.authenticate(account['token']) == account['entrant']['entrant_id']
    assert all(j['budget'] == 3 and j['done'] == 0 for j in store.jobs())
    assert {j['name'] for j in store.jobs()} == {'bot','random'}
    assert next(j for j in store.jobs() if j['name']=='bot')['id'] == job
    assert not (store.directory/'archives').exists()
    module.evaluate(store, 2)
    assert sum(b['completed_games'] for b in store.leaderboards().values()) == 6
    # Job identities survive, but this is a new evaluation; there are no old
    # records or ratings carried into its completed prefix.
    assert len(store.matches()['matches']) == 6
    board = store.leaderboard()
    module.evaluate(store, 2)
    assert store.leaderboard() == board
