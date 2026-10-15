import importlib.util
import json
from pathlib import Path

import pytest

from generals_arena.league.config import LeagueConfig
from generals_arena.league.service import run_jobs
from generals_arena.league.store import LeagueStore

spec = importlib.util.spec_from_file_location('brawl_league',Path(__file__).parents[1]/'benchmarks'/'brawl_league.py')
brawl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(brawl)


def test_brawl_keeps_commit_and_prepends_exact_modes_without_old_data(tmp_path):
    official = LeagueStore.create(tmp_path/'league',LeagueConfig(max_ticks=2,player_counts=(2,3,4)))
    official.add({'name':'old','kind':'baseline','policy':'random'},games=2)
    assert run_jobs(official) == 2
    target,job_id = official.add({'name':'keep-v5','kind':'baseline','policy':'aggressive'},
                               games=4,mode_games={'1v1':2,'ffa':2})
    assert run_jobs(official) == 4
    account = official.account('human')
    wait = official.entrants()[official.anchor]
    metadata = official.directory/'commits'/wait['version']/'commit.json'
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps(dict(entrant_id=wait['entrant_id'],job_id='obsolete',name='WAIT')))
    old_ids = {m['game_id'] for m in official.matches()['matches'] if target['entrant_id'] not in m['entrant_ids']}
    with official.connect() as db:
        keep_ids = {r[0] for r in db.execute('SELECT id FROM games WHERE job_id=?',(job_id,))}
        original = brawl.preserved_digest(db,job_id)
    stage,descriptor = brawl.prepare(official,'keep-v5',7,9)
    assert target['entrant_id'] not in stage.entrants()
    assert sum(j['budget'] for j in stage.jobs()) == 16
    with pytest.raises(ValueError,match='finish the complete brawl'):
        brawl.install(official,stage,descriptor)
    brawl.evaluate(stage,2)
    assert brawl.prepare(official,'keep-v5',7,9)[0].directory == stage.directory
    brawl.install(official,stage,descriptor)
    assert not stage.directory.exists()
    official.verify_execution()
    assert official.authenticate(account['token']) == account['entrant']['entrant_id']
    record = json.loads(metadata.read_text())
    assert record['job_id'] in {job['id'] for job in official.jobs()}
    assert record['evaluation'] == 'brawl'
    assert json.loads(metadata.with_name('publication.json').read_text())['evaluation'] == 'brawl'
    with official.connect() as db:
        records = db.execute('SELECT id,job_id,sequence,result FROM games ORDER BY sequence').fetchall()
        assert [r['sequence'] for r in records] == list(range(20))
        assert {r['id'] for r in records[16:]} == keep_ids
        assert not old_ids.intersection(r['id'] for r in records)
        assert brawl.preserved_digest(db,job_id) == original
        assert all(target['entrant_id'] not in json.loads(r['result'])['entrant_ids'] for r in records[:16])
    assert official.leaderboard('1v1')['completed_games'] == 9
    assert official.leaderboard('ffa')['completed_games'] == 11
    assert {r['game_id'] for r in official.matches(limit=4)['matches']} == keep_ids
    before = official.leaderboards()
    official.rebuild()
    assert official.leaderboards() == before
