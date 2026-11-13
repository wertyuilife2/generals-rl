"""Derived scheduling/history metadata must follow transactions and old writers."""
import json
import random
import sqlite3

import pytest

from generals_arena.competitions import indexes
from generals_arena.competitions.config import LeagueConfig
from generals_arena.competitions.store import CompetitionStore


def populate(store, count=80):
    rng = random.Random(42)
    with store.connect(write=True) as db:
        for i in range(count):
            status = rng.choice(['complete', 'complete', 'pending', 'live', 'aborted', 'failed'])
            plan = {'game_id': str(i), 'entrant_ids': rng.choices('abcde', k=rng.choice([2, 3, 4, 8]))}
            db.execute('INSERT INTO games(id,status,plan,sequence,created,finished) VALUES (?,?,?,?,?,?)',
                       (str(i), status, json.dumps(plan), i if status == 'complete' else None, i, i+.5))


def check_coverage(store):
    with store.connect() as db:
        complete = [json.loads(row[0])['entrant_ids']
                    for row in db.execute("SELECT plan FROM games WHERE status='complete'")]
        assert db.execute('SELECT min(games) FROM matchup_coverage').fetchone()[0] != 0
    for mode in ('1v1', 'ffa'):
        for target in 'abcdef':
            expected = {}
            for ids in complete:
                if (len(ids) == 2 if mode == '1v1' else len(ids) > 2) and target in ids:
                    for opponent in set(ids) - {target}:
                        expected[opponent] = expected.get(opponent, 0) + 1
            assert store.pair_counts(target, mode) == expected


def check_history(store):
    with store.connect() as db:
        rows = [dict(row) for row in db.execute("SELECT * FROM games WHERE status IN "
                "('complete','aborted','failed') ORDER BY sequence DESC,finished DESC")]
    options = [{}, {'mode': '1v1'}, {'mode': 'ffa'}, {'mode': 'multiplayer'}, {'players': 8},
               {'include': ['a', 'b']}, {'include': ['a', 'a'], 'exclude': ['d', 'e']},
               {'entrant': 'a', 'opponent': 'a'}, {'entrant': 'a', 'opponent': 'b'},
               {'include': ['a'], 'exclude': ['a']}, {'exclude': ['missing', 'a']}]
    for filters in options:
        expected = []
        for row in rows:
            ids = json.loads(row['plan'])['entrant_ids']
            if filters.get('mode') == '1v1' and len(ids) != 2:
                continue
            if filters.get('mode') in ('ffa', 'multiplayer') and len(ids) <= 2:
                continue
            if 'players' in filters and filters['players'] != len(ids):
                continue
            if not set(filters.get('include', ())).issubset(ids) or set(filters.get('exclude', ())) & set(ids):
                continue
            if any(ids.count(key) < (2 if filters.get('entrant') == filters.get('opponent') else 1)
                   for key in (filters.get('entrant'), filters.get('opponent')) if key):
                continue
            expected.append(row['id'])
        for offset in (0, 5):
            actual = store.matches(limit=7, offset=offset, **filters)
            assert actual['total'] == len(expected)
            assert [row['game_id'] for row in actual['matches']] == expected[offset:offset+7]


def test_incremental_indexes_follow_status_plan_delete_and_rollback(tmp_path):
    store = CompetitionStore.create(tmp_path, LeagueConfig(player_counts=(2, 4)), reference='random')
    populate(store)
    check_coverage(store)
    check_history(store)
    # A separate writer needs no knowledge of the new projections.
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE games SET status='complete' WHERE id IN ('0','1','2')")
        db.execute("UPDATE games SET status='pending' WHERE id='3'")
        db.execute("UPDATE games SET plan=? WHERE id='0'", (json.dumps({'game_id': '0', 'entrant_ids': ['a', 'a', 'b']}),))
        db.execute("DELETE FROM games WHERE id IN ('4','5','6')")
        db.execute("UPDATE games SET finished=1000 WHERE id='1'")
    check_coverage(store)
    check_history(store)
    with pytest.raises(RuntimeError):
        with store.connect(write=True) as db:
            db.execute("DELETE FROM games WHERE status='complete'")
            raise RuntimeError('rollback results and projections together')
    check_coverage(store)
    check_history(store)
    with store.connect(write=True) as db:
        db.execute("UPDATE games SET status=status,plan=plan")
    check_coverage(store)  # Repeated updates cannot count the same game twice.
    check_history(store)
    with sqlite3.connect(store.path) as db:
        db.execute('VACUUM')
    check_coverage(store)
    check_history(store)


def test_legacy_backfill_is_atomic_and_idempotent(tmp_path, monkeypatch):
    store = CompetitionStore.create(tmp_path, LeagueConfig(player_counts=(2, 4)), reference='random')
    with store.connect(write=True) as db:
        for suffix in ('insert', 'update', 'delete'):
            db.execute(f'DROP TRIGGER match_index_{suffix}')
        for table in ('match_history', 'match_participants', 'matchup_coverage'):
            db.execute(f'DROP TABLE {table}')
        db.execute("DELETE FROM meta WHERE key='query_indexes'")
    populate(store)
    install = indexes.install
    def interrupted(db):
        install(db)
        raise RuntimeError('interrupted migration')
    monkeypatch.setattr(indexes, 'install', interrupted)
    with pytest.raises(RuntimeError):
        CompetitionStore(tmp_path)
    with store.connect() as db:
        assert db.execute("SELECT 1 FROM meta WHERE key='query_indexes'").fetchone() is None
        assert db.execute("SELECT 1 FROM sqlite_master WHERE name='match_history'").fetchone() is None
        original = list(map(tuple, db.execute('SELECT * FROM games')))
    monkeypatch.setattr(indexes, 'install', install)
    migrated = CompetitionStore(tmp_path)
    check_coverage(migrated)
    check_history(migrated)
    check_coverage(CompetitionStore(tmp_path))
    with migrated.connect(write=True) as db:
        install(db)
        assert list(map(tuple, db.execute('SELECT * FROM games'))) == original
