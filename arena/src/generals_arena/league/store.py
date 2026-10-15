"""SQLite transactions keep results, compressed traces, and ratings consistent."""
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import secrets
import sqlite3
import time
import uuid

from ..entrants import prepare_entrant
from ..execution import decision_timeout
from ..util import canonical, integer
from .config import LeagueConfig
from .rating import SCALE, PROVISIONAL_GAMES, ranks
from .rating_policy import EVIDENCE_POLICIES, POLICY, score
from .mode_rating import CF_SCALE, FFA_MEAN, initial_rating
from .rank_model import comparisons, rotation_key
from .modes import MODES, rating_mode, validate_mode, mode_predicate
from .titles import rating_title, title_bands

PAIR_SCHEMA = """CREATE TABLE IF NOT EXISTS rating_pairs(
 mode TEXT NOT NULL,
 left_id TEXT NOT NULL, right_id TEXT NOT NULL, weight REAL NOT NULL, score REAL NOT NULL,
 PRIMARY KEY(mode,left_id,right_id))"""

RATING_SCHEMA = """CREATE TABLE ratings(mode TEXT NOT NULL,
 entrant_id TEXT NOT NULL REFERENCES entrants(id), elo REAL NOT NULL,
 games INTEGER NOT NULL, rated_games INTEGER NOT NULL, appearances INTEGER NOT NULL,
 wins INTEGER NOT NULL, losses INTEGER NOT NULL, draws INTEGER NOT NULL,
 PRIMARY KEY(mode,entrant_id))"""

JOB_MODE_SCHEMA = """CREATE TABLE IF NOT EXISTS job_modes(
 job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE, mode TEXT NOT NULL,
 budget INTEGER NOT NULL, done INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(job_id,mode))"""

SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE entrants(id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE accounts(token_hash TEXT PRIMARY KEY, entrant_id TEXT UNIQUE REFERENCES entrants(id));
CREATE TABLE jobs(id TEXT PRIMARY KEY, entrant_id TEXT NOT NULL REFERENCES entrants(id),
 budget INTEGER NOT NULL, done INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'queued',
 error TEXT, created REAL NOT NULL);
CREATE TABLE games(id TEXT PRIMARY KEY, job_id TEXT REFERENCES jobs(id), status TEXT NOT NULL,
 plan TEXT NOT NULL, result TEXT, replay BLOB, sequence INTEGER UNIQUE, created REAL NOT NULL, finished REAL);
CREATE INDEX games_status ON games(status, created);
""" + RATING_SCHEMA + ';' + PAIR_SCHEMA + ';' + JOB_MODE_SCHEMA + ';'


class LeagueStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / "league.sqlite3"
        if not self.path.is_file():
            raise ValueError("league does not exist; run generals-arena league init first")
        with self.connect() as db:
            meta = {row["key"]: json.loads(row["value"]) for row in db.execute("SELECT * FROM meta")}
        self.manifest = meta["manifest"]
        if self.manifest.get("schema") not in (3, 4):
            raise ValueError("this is an archived league schema; initialize a new rank-rated league directory")
        self.config = LeagueConfig(**{"map_generation": None, **self.manifest["config"]})
        self.anchor = meta["anchor"]
        # A pool count should scan a compact index, not parse every saved plan.
        # Install for existing leagues as well as newly created ones.
        with self.connect(write=True) as db:
            db.execute("CREATE INDEX IF NOT EXISTS games_completed_players ON games "
                       "(json_array_length(plan, '$.entrant_ids'), sequence) WHERE status='complete'")
            # Filter/count/order history without reading the large result and
            # replay rows. Only the requested page is fetched from the table.
            db.execute("CREATE INDEX IF NOT EXISTS games_history ON games "
                       "(sequence DESC, finished DESC, plan) "
                       "WHERE status IN ('complete','aborted','failed')")

    @property
    def reference(self):
        with self.connect() as db:
            row = db.execute("SELECT value FROM meta WHERE key='rating_reference'").fetchone()
        return json.loads(row[0]) if row else self.anchor

    @classmethod
    def create(cls, directory, config=None, *, reference="wait"):
        if reference not in ("wait", "random"):
            raise ValueError("rating reference must be wait or random")
        config = config or LeagueConfig()
        manifest = config.manifest()
        anchor = prepare_entrant({"name": "random", "kind": "baseline", "policy": "random"})
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "league.sqlite3"
        # Exclusive creation: never overwrite another league.
        with path.open("xb"):
            pass
        try:
            with sqlite3.connect(path) as db:
                db.executescript(SCHEMA)
                db.executemany("INSERT INTO meta VALUES (?,?)",
                               [("manifest", canonical(manifest)), ("anchor", canonical(anchor["entrant_id"]))])
                db.execute("INSERT INTO entrants VALUES (?,?,?,?)",
                           (anchor["entrant_id"], anchor["name"], anchor["kind"], canonical(anchor)))
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        store = cls(directory)
        if reference == "wait":
            from ..reference import register_wait_reference
            register_wait_reference(store)
        return store

    def set_rating_reference(self, entrant_id):
        """Choose the fixed numerical anchor before the first rated game."""
        with self.connect(write=True) as db:
            if not db.execute('SELECT 1 FROM entrants WHERE id=?', (entrant_id,)).fetchone():
                raise ValueError('rating reference must be a registered entrant')
            if entrant_id != self.anchor and db.execute("SELECT 1 FROM games WHERE status='complete' LIMIT 1").fetchone():
                raise ValueError('choose a fixed rating anchor before rated games; start a new league to change it')
            db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',
                       ('rating_reference', canonical(entrant_id)))
            db.execute("UPDATE meta SET value=? WHERE key='anchor'", (canonical(entrant_id),))
        self.anchor = entrant_id

    @contextmanager
    def connect(self, write=False):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA journal_mode=WAL")
            if write:
                db.execute("BEGIN IMMEDIATE")
            else:
                db.execute("BEGIN")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def verify_execution(self):
        if self.config.manifest() != self.manifest:
            # JSON arrays vs tuples are normalized by canonical serialization.
            if canonical(self.config.manifest()) != canonical(self.manifest):
                raise ValueError("league rules/source changed; use its original code or initialize a new league")

    def resize_maps(self, min_side, max_side, side_cap):
        """Explicit map revision; preserve accounts, results, and rating history.

        Only distribution/presentation may change here. A different engine,
        scoring algorithm or scheduler still requires its own league.
        """
        from .service import process_lock
        config = replace(self.config, map_generation=None, min_side=min_side, max_side=max_side, side_cap=side_cap)
        manifest = config.manifest()
        previous = self.manifest
        for key in ("schema", "engine_sha256", "rating", "decision_timing"):
            if previous.get(key) != manifest[key]:
                raise ValueError("map resize cannot change the engine or rating rules")
        for name in ("game.py", "observation.py", "rating.py", "scheduler.py"):
            if previous["implementation"].get(name) != manifest["implementation"][name]:
                raise ValueError("map resize cannot change simulation, scoring or scheduling code")
        with process_lock(self, ".server.lock"), process_lock(self, ".worker.lock"):
            with self.connect(write=True) as db:
                current = json.loads(db.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()[0])
                if canonical(current) != canonical(previous):
                    raise ValueError("league changed; reopen it before resizing")
                if db.execute("SELECT 1 FROM games WHERE status IN ('live','pending') LIMIT 1").fetchone():
                    raise ValueError("finish active and pending games before resizing maps")
                if canonical(previous) != canonical(manifest):
                    row = db.execute("SELECT value FROM meta WHERE key='rules_history'").fetchone()
                    history = json.loads(row[0]) if row else []
                    history.append({"manifest": previous, "replaced_at": time.time(),
                                    "next_sequence": db.execute("SELECT COALESCE(MAX(sequence),-1)+1 FROM games").fetchone()[0],
                                    "reason": "map bounds and human fog presentation updated"})
                    db.execute("INSERT OR REPLACE INTO meta VALUES ('rules_history',?)", (canonical(history),))
                    db.execute("UPDATE meta SET value=? WHERE key='manifest'", (canonical(manifest),))
        self.config, self.manifest = config, manifest
        return {p: config.side_range(p) for p in config.player_counts}

    def configure_maps(self, options=None):
        """Explicitly adopt the installed generator for future games.

        This accepts the new environment source revision as part of the map
        upgrade. Recorded plans, replays, ratings and entrants are untouched.
        Finish evaluations and stop servers/workers before changing rules.
        """
        from generals import MapGenerationConfig
        from .service import process_lock
        options = MapGenerationConfig() if options is None else options
        config = replace(self.config, map_generation=options)
        manifest, previous = config.manifest(), self.manifest
        for key in ('schema', 'rating', 'decision_timing', 'simulation_backend', 'human_observation'):
            if previous.get(key) != manifest[key]:
                raise ValueError('map configuration cannot change simulation or rating rules')
        for name in ('game.py', 'observation.py', 'rating.py', 'scheduler.py', 'views.py'):
            if previous['implementation'].get(name) != manifest['implementation'][name]:
                raise ValueError('map configuration cannot change simulation, scoring or scheduling code')
        with process_lock(self, '.server.lock'), process_lock(self, '.worker.lock'), process_lock(self, '.registry.lock'):
            with self.connect(write=True) as db:
                current = json.loads(db.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()[0])
                if canonical(current) != canonical(previous):
                    raise ValueError('league changed; reopen it before configuring maps')
                if (db.execute("SELECT 1 FROM games WHERE status IN ('live','pending') LIMIT 1").fetchone() or
                    db.execute("SELECT 1 FROM jobs WHERE done < budget AND status='queued' LIMIT 1").fetchone()):
                    raise ValueError('finish active and queued evaluations before configuring maps')
                if canonical(previous) != canonical(manifest):
                    row = db.execute("SELECT value FROM meta WHERE key='rules_history'").fetchone()
                    history = json.loads(row[0]) if row else []
                    history.append({'manifest': previous, 'replaced_at': time.time(),
                                    'next_sequence': db.execute('SELECT COALESCE(MAX(sequence),-1)+1 FROM games').fetchone()[0],
                                    'reason': 'configured random-map generation'})
                    db.execute("INSERT OR REPLACE INTO meta VALUES ('rules_history',?)", (canonical(history),))
                    db.execute("UPDATE meta SET value=? WHERE key='manifest'", (canonical(manifest),))
        self.config, self.manifest = config, manifest
        return {p: config.side_range(p) for p in config.player_counts}

    def entrants(self):
        with self.connect() as db:
            return {r["id"]: json.loads(r["data"]) for r in db.execute("SELECT * FROM entrants ORDER BY name")}

    def retired_entrants(self):
        """Return immutable entrant IDs that are hidden from the active roster."""
        with self.connect() as db:
            row = db.execute("SELECT value FROM meta WHERE key='retired_entrants'").fetchone()
        return json.loads(row[0]) if row else {}

    def retired_entrant_ids(self):
        return set(self.retired_entrants())

    def set_retired(self, entrant, retired=True, *, reason=None):
        """Retire or restore an entrant without deleting results or artifacts.

        Retirement is deliberately restricted to entrants with no scheduled or
        live games. This makes the boundary auditable: every existing plan runs
        under the roster that selected it, and all later plans use the new one.
        """
        if not isinstance(entrant, str) or not entrant.strip():
            raise ValueError("entrant must be a name or entrant ID")
        selector = entrant.strip()
        if selector.startswith('@'):
            selector = selector[1:]
        if reason is not None and (not isinstance(reason, str) or len(reason) > 500):
            raise ValueError("retirement reason must be a string of at most 500 characters")
        with self.connect(write=True) as db:
            row = db.execute("SELECT id,name,kind FROM entrants WHERE id=? OR name=?",
                             (selector, selector)).fetchone()
            if row is None:
                raise ValueError(f"entrant not found: {entrant}")
            key = row['id']
            current_row = db.execute("SELECT value FROM meta WHERE key='retired_entrants'").fetchone()
            current = json.loads(current_row[0]) if current_row else {}
            history_row = db.execute("SELECT value FROM meta WHERE key='entrant_status_history'").fetchone()
            history = json.loads(history_row[0]) if history_row else []
            if retired:
                if key in (self.anchor, self.reference):
                    raise ValueError("the rating anchor/reference cannot be retired")
                if key in current:
                    return {'entrant_id': key, 'name': row['name'], 'retired': True, **current[key]}
                active = db.execute("""SELECT 1 FROM games,json_each(games.plan,'$.entrant_ids') participant
                    WHERE games.status IN ('pending','live') AND participant.value=? LIMIT 1""", (key,)).fetchone()
                if active:
                    raise ValueError("entrant has pending or live games; finish them before retirement")
                record = {'retired_at': time.time(), 'reason': reason.strip() if reason else None}
                current[key] = record
            else:
                if key not in current:
                    return {'entrant_id': key, 'name': row['name'], 'retired': False}
                record = current.pop(key)
            history.append({'entrant_id': key, 'name': row['name'],
                            'action': 'retire' if retired else 'restore',
                            'at': time.time(), 'reason': reason.strip() if reason else None})
            db.execute("INSERT OR REPLACE INTO meta VALUES ('retired_entrants',?)", (canonical(current),))
            db.execute("INSERT OR REPLACE INTO meta VALUES ('entrant_status_history',?)", (canonical(history),))
        result = {'entrant_id': key, 'name': row['name'], 'retired': bool(retired)}
        if retired:
            result.update(record)
        return result

    def add(self, spec, *, base_dir=Path("."), games=1000, mode_games=None):
        self.verify_execution()
        integer("games", games, 1)
        if mode_games is not None:
            for mode, amount in mode_games.items():
                validate_mode(mode)
                integer('mode games', amount, 1)
                if not any((count == 2) == (mode == '1v1') for count in self.config.player_counts):
                    raise ValueError('evaluation mode is not enabled')
            if sum(mode_games.values()) != games:
                raise ValueError('mode budgets must sum to games')
        if not isinstance(spec, dict):
            raise ValueError("entrant specification must be a JSON object")
        if spec.get("kind") == "rl":
            raise ValueError("RL evaluation is deferred: fixed-size checkpoints cannot enter the random-map league")
        entrant = prepare_entrant(spec, base_dir=base_dir)
        with self.connect(write=True) as db:
            old = db.execute("SELECT * FROM entrants WHERE name=?", (entrant["name"],)).fetchone()
            if old:
                if old["id"] != entrant["entrant_id"]:
                    raise ValueError("name already identifies a different version; give the new algorithm a new name")
                job = db.execute("SELECT id FROM jobs WHERE entrant_id=? ORDER BY created LIMIT 1",
                                 (old["id"],)).fetchone()
                if job:
                    return entrant, job['id']  # Idempotent registration.
            else:
                db.execute("INSERT INTO entrants VALUES (?,?,?,?)",
                       (entrant["entrant_id"], entrant["name"], entrant["kind"], canonical(entrant)))
            job_id = uuid.uuid4().hex
            db.execute("INSERT INTO jobs(id,entrant_id,budget,created) VALUES (?,?,?,?)",
                       (job_id, entrant["entrant_id"], games, time.time()))
            if mode_games:
                db.executemany('INSERT INTO job_modes(job_id,mode,budget) VALUES (?,?,?)',
                               [(job_id, mode, amount) for mode, amount in mode_games.items()])
        return entrant, job_id

    def register(self, spec, *, base_dir=Path('.')):
        """Register a private-evaluation opponent without scheduling its own job."""
        self.verify_execution()
        if spec.get('kind') == 'rl':
            raise ValueError('RL evaluation is deferred')
        entrant = prepare_entrant(spec, base_dir=base_dir)
        with self.connect(write=True) as db:
            old = db.execute('SELECT id FROM entrants WHERE name=?', (entrant['name'],)).fetchone()
            if old and old['id'] != entrant['entrant_id']:
                raise ValueError('name already identifies a different version')
            if not old:
                db.execute('INSERT INTO entrants VALUES (?,?,?,?)',
                    (entrant['entrant_id'], entrant['name'], entrant['kind'], canonical(entrant)))
        return entrant

    def account(self, name):
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 48:
            raise ValueError("name must contain 1–48 characters")
        key, token = uuid.uuid4().hex, secrets.token_urlsafe(32)
        entrant = {"entrant_id": key, "name": name.strip(), "kind": "human", "version": "human"}
        try:
            with self.connect(write=True) as db:
                db.execute("INSERT INTO entrants VALUES (?,?,?,?)", (key, entrant["name"], "human", canonical(entrant)))
                db.execute("INSERT INTO accounts VALUES (?,?)", (sha256(token.encode()).hexdigest(), key))
        except sqlite3.IntegrityError as exc:
            raise ValueError("this name is already registered; restore its browser token or use another name") from exc
        return {"token": token, "entrant": entrant}

    def authenticate(self, token):
        with self.connect() as db:
            row = db.execute("SELECT entrant_id FROM accounts WHERE token_hash=?",
                             (sha256(token.encode()).hexdigest(),)).fetchone()
        if row is None:
            raise PermissionError("invalid player token")
        return row[0]

    def jobs(self):
        with self.connect() as db:
            jobs = [dict(r) for r in db.execute("SELECT jobs.*, entrants.name FROM jobs JOIN entrants ON entrants.id=jobs.entrant_id ORDER BY created")]
            modes = {}
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='job_modes'").fetchone():
                for row in db.execute('SELECT * FROM job_modes'):
                    modes.setdefault(row['job_id'], {})[row['mode']] = dict(budget=row['budget'], done=row['done'])
            for job in jobs:
                job['modes'] = modes.get(job['id'], {})
            return jobs

    def retry(self, job_id=None):
        with self.connect(write=True) as db:
            jobs = db.execute("SELECT j.id,e.kind FROM jobs j JOIN entrants e ON e.id=j.entrant_id "
                              "WHERE j.status='failed'").fetchall()
            for job in jobs:
                if job_id is not None and job['id'] != job_id:
                    continue
                failures = [json.loads(r[0] or '{}') for r in db.execute(
                    "SELECT result FROM games WHERE job_id=? AND status='failed'", (job['id'],))]
                infrastructure = failures and all(
                    r.get('error_detail', {}).get('player') is None for r in failures)
                if infrastructure:
                    # Keep the result and replay for audit, but distinguish an
                    # unattributed host failure from a policy violation.  It
                    # remains unrated and a replacement game will be planned.
                    db.execute("UPDATE games SET status='aborted' WHERE job_id=? AND status='failed'",
                               (job['id'],))
                if job['kind'] != 'cpp' or infrastructure or (failures and all(
                        decision_timeout(r.get('error_detail', {})) for r in failures)):
                    db.execute("UPDATE jobs SET status='queued',error=NULL WHERE id=?", (job['id'],))

    def fail(self, job_id, error):
        with self.connect(write=True) as db:
            db.execute("UPDATE jobs SET status='failed',error=? WHERE id=?", (str(error), job_id))

    def insert_plans(self, plans, job_id=None, status="pending"):
        with self.connect(write=True) as db:
            for plan in plans:
                db.execute("INSERT INTO games(id,job_id,status,plan,created) VALUES (?,?,?,?,?)",
                           (plan["game_id"], job_id, status, canonical(plan), time.time()))

    def pending(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT plan FROM games WHERE job_id=? AND status='pending' ORDER BY rowid LIMIT 1", (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def default_mode(self):
        return '1v1' if 2 in self.config.player_counts else 'ffa'

    def ratings(self, mode=None):
        mode = validate_mode(mode or self.default_mode())
        with self.connect() as db:
            return {r["entrant_id"]: r["elo"] for r in db.execute("SELECT * FROM ratings WHERE mode=?", (mode,))}

    def pair_counts(self, target, mode=None):
        counts = {}
        where = ' AND ' + mode_predicate(mode) if mode else ''
        with self.connect() as db:
            for row in db.execute("SELECT plan FROM games WHERE status='complete'" + where):
                ids = json.loads(row[0])["entrant_ids"]
                if target in ids:
                    for key in set(ids):
                        if key != target:
                            counts[key] = counts.get(key, 0) + 1
        return counts

    def _score(self, db, result):
        ids = result["entrant_ids"]
        mode = rating_mode(result)
        result['rating_mode'] = mode
        rows = list(db.execute("SELECT * FROM ratings WHERE mode=?", (mode,)))
        old = {r["entrant_id"]: r["elo"] for r in rows}
        games = {r["entrant_id"]: r["rated_games"] for r in rows}
        evidence = None
        if result.get('rating_policy') in EVIDENCE_POLICIES:
            db.execute(PAIR_SCHEMA)
            evidence = {(row[0], row[1]): (row[2], row[3])
                        for row in db.execute('SELECT left_id,right_id,weight,score FROM rating_pairs WHERE mode=?', (mode,))}
        changes = score(old, result, self.anchor, self.config.k_factor, games, evidence)
        if evidence is not None:
            db.executemany('INSERT OR REPLACE INTO rating_pairs VALUES (?,?,?,?,?)',
                           [(mode, left, right, *evidence[left, right])
                            for left, right, _, _ in comparisons(result)])
        result["ranks"] = ranks(result)
        result["rating_changes"] = changes
        for key in dict.fromkeys(ids):
            seats = [seat for seat, entrant in enumerate(ids) if entrant == key]
            winner = result["winner_seat"]
            appearances = len(seats)
            win = int(winner is not None and winner in seats)
            loss = appearances - win if winner is not None else 0
            draw = appearances if winner is None else 0
            db.execute("""INSERT INTO ratings VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(mode,entrant_id)
                DO UPDATE SET elo=excluded.elo,games=games+1,rated_games=excluded.rated_games,appearances=appearances+excluded.appearances,
                wins=wins+excluded.wins,losses=losses+excluded.losses,draws=draws+excluded.draws""",
                       (mode, key, changes[key]["after"], 1, changes[key]["rated_games"], appearances, win, loss, draw))

    def complete(self, game):
        from ..execution import result
        return self.complete_result(result(game), game.replay_blob())

    def fail_game(self, result, blob):
        """Preserve a policy violation and its partial replay without awarding rating."""
        with self.connect(write=True) as db:
            row = db.execute('SELECT status,plan FROM games WHERE id=?', (result['game_id'],)).fetchone()
            if row is None or row['status'] not in ('pending', 'live'):
                raise ValueError('failed game was not pending/live')
            if any(result.get(k) != v for k, v in json.loads(row['plan']).items()):
                raise ValueError('failed game plan mismatch')
            db.execute("UPDATE games SET status='failed',result=?,replay=?,finished=? WHERE id=?",
                (canonical(result), blob, time.time(), result['game_id']))

    def disabled_policy_ids(self):
        """Fatal C++ errors disqualify; decision timeouts only surrender a seat."""
        with self.connect() as db:
            native = {r[0] for r in db.execute("SELECT id FROM entrants WHERE kind='cpp'")}
            rows = db.execute("SELECT result FROM games WHERE status='failed' AND result IS NOT NULL").fetchall()
        disabled = set()
        for row in rows:
            result = json.loads(row[0])
            if decision_timeout(result.get('error_detail', {})):
                continue
            seat = result.get('error_detail', {}).get('player')
            if seat is not None and result['entrant_ids'][seat] in native:
                disabled.add(result['entrant_ids'][seat])
        return disabled

    def complete_result(self, result, blob):
        with self.connect(write=True) as db:
            row = db.execute("SELECT * FROM games WHERE id=?", (result["game_id"],)).fetchone()
            if row is None:
                raise ValueError("game was never scheduled")
            if row["status"] == "complete":
                return json.loads(row["result"])
            plan = json.loads(row["plan"])
            if row["status"] not in ("pending", "live") or any(result.get(key) != value for key, value in plan.items()):
                raise ValueError("game plan/status mismatch")
            sequence = db.execute("SELECT COALESCE(MAX(sequence),-1)+1 FROM games").fetchone()[0]
            result = {**result, "sequence": sequence}
            self._score(db, result)
            db.execute("UPDATE games SET status='complete',result=?,replay=?,sequence=?,finished=? WHERE id=?",
                       (canonical(result), blob, sequence, time.time(), result["game_id"]))
            if row["job_id"]:
                mode = rating_mode(result)
                quota = db.execute('SELECT budget,done FROM job_modes WHERE job_id=? AND mode=?', (row['job_id'], mode)).fetchone()
                if ((quota and quota['done'] >= quota['budget']) or
                    (quota is None and db.execute('SELECT 1 FROM job_modes WHERE job_id=?', (row['job_id'],)).fetchone())):
                    raise ValueError('evaluation mode budget exhausted')
                db.execute('UPDATE job_modes SET done=done+1 WHERE job_id=? AND mode=?', (row['job_id'], mode))
                db.execute("UPDATE jobs SET done=done+1,status=CASE WHEN done+1>=budget THEN 'complete' ELSE 'queued' END WHERE id=?", (row["job_id"],))
        return result

    def abort_live(self, game_id=None):
        with self.connect(write=True) as db:
            if game_id is None:
                db.execute("UPDATE games SET status='aborted',finished=? WHERE status='live'", (time.time(),))
            else:
                db.execute("UPDATE games SET status='aborted',finished=? WHERE status='live' AND id=?", (time.time(), game_id))

    def leaderboard(self, mode=None, *, include_retired=False):
        mode = validate_mode(mode or self.default_mode())
        current = self.manifest['rating'] in ('independent-gor-codeforces-v2', 'independent-relative-gor-codeforces-v3')
        relative = self.manifest['rating'] == 'independent-relative-gor-codeforces-v3' and mode == '1v1'
        mean_pool = current and mode == 'ffa'
        with self.connect() as db:
            rows = [dict(row) for row in db.execute("""SELECT e.id entrant_id,e.name,e.kind,e.rowid registration_order,
                COALESCE(r.elo,?) elo,COALESCE(r.games,0) games,COALESCE(r.rated_games,0) rated_games,COALESCE(r.appearances,0) appearances,
                COALESCE(r.wins,0) wins,COALESCE(r.losses,0) losses,COALESCE(r.draws,0) draws FROM entrants e
                LEFT JOIN ratings r ON r.entrant_id=e.id AND r.mode=? ORDER BY elo DESC,e.name""", (initial_rating(mode) if current else 0., mode))]
            completed = db.execute("SELECT count(*) FROM games INDEXED BY games_completed_players "
                                   "WHERE status='complete' AND " + mode_predicate(mode)).fetchone()[0]
        retired = self.retired_entrants()
        offset = next(row['elo'] for row in rows if row['entrant_id'] == self.anchor) if relative else 0.
        for row in rows:
            if relative:
                row['raw_elo'] = row['elo']
                row['elo'] -= offset
            row["anchor"] = not mean_pool and row["entrant_id"] == self.anchor
            row["provisional"] = row["rated_games"] < PROVISIONAL_GAMES
            if mode == '1v1':
                row['rating_title'] = rating_title(row['elo'],row['rated_games'])
            if row['entrant_id'] in retired:
                row['retired'] = True
        disabled = self.disabled_policy_ids()
        for row in rows:
            if row['entrant_id'] in disabled:
                row['disqualified'] = True
        if not include_retired:
            rows = [row for row in rows if not row.get('retired')]
        return {"mode": mode, "player_counts": tuple(p for p in self.config.player_counts if (p == 2) == (mode == '1v1')), "ranking": rows,
                "anchor": None if mean_pool else self.anchor,
                "rating_model": ('codeforces' if mean_pool else 'egf-gor') if current else 'rank',
                "rating_offset": offset,
                "title_bands": title_bands() if mode == '1v1' else [],
                "rating_mean": FFA_MEAN if mean_pool else None,
                "scale": (CF_SCALE if mean_pool else None) if current else SCALE,
                "completed_games": completed,
                "anchor_games": next((row['games'] for row in rows if row['anchor']), 0)}

    def leaderboards(self, *, include_retired=False):
        return {mode: self.leaderboard(mode, include_retired=include_retired) for mode in MODES}

    def matches(self, limit=50, offset=0, players=None, entrant=None, opponent=None,
                *, include=None, exclude=None, mode=None, include_retired=True):
        integer("limit", limit, 1)
        integer("offset", offset)
        if limit > 100:
            raise ValueError("limit must be <= 100")
        if mode not in (None, 'all', '1v1', 'multiplayer', 'ffa'):
            raise ValueError('mode must be all, 1v1 or ffa (multiplayer)')
        def ids(value):
            if value is None:
                return set()
            if not isinstance(value, (list, tuple, set)) or any(not isinstance(key, str) or not key for key in value):
                raise ValueError('include and exclude must be lists of entrant IDs')
            return set(value)
        included, excluded = ids(include), ids(exclude)
        if not include_retired:
            excluded.update(self.retired_entrant_ids())
        where, values = "status IN ('complete','aborted','failed')", []
        if mode == '1v1':
            where += " AND json_array_length(plan, '$.entrant_ids')=2"
        elif mode in ('multiplayer', 'ffa'):
            where += " AND json_array_length(plan, '$.entrant_ids')>2"
        if included:
            where += """ AND (SELECT count(DISTINCT value) FROM json_each(plan, '$.entrant_ids')
                WHERE value IN (SELECT value FROM json_each(?)))=?"""
            values.extend((canonical(sorted(included)), len(included)))
        if excluded:
            where += """ AND NOT EXISTS (SELECT 1 FROM json_each(plan, '$.entrant_ids')
                WHERE value IN (SELECT value FROM json_each(?)))"""
            values.append(canonical(sorted(excluded)))
        if players is not None:
            where += " AND json_array_length(plan, '$.entrant_ids')=?"
            values.append(players)
        for key in dict.fromkeys(key for key in (entrant, opponent) if key):
            if not isinstance(key, str):
                raise ValueError('entrant and opponent must be entrant IDs')
            # Selecting the same strategy twice means two distinct seats.
            minimum = 2 if entrant == opponent == key else 1
            where += " AND (SELECT count(*) FROM json_each(plan, '$.entrant_ids') WHERE value=?)>=?"
            values.extend((key, minimum))
        with self.connect() as db:
            source = f"FROM games INDEXED BY games_history WHERE {where}"
            total = db.execute(f"SELECT count(*) {source}", values).fetchone()[0]
            rows = db.execute(f"""SELECT id,status,plan,result,finished FROM games WHERE rowid IN (
                SELECT rowid {source} ORDER BY sequence DESC,finished DESC LIMIT ? OFFSET ?
                ) ORDER BY sequence DESC,finished DESC""", (*values, limit, offset)).fetchall()
        return {"total": total, "matches": [{**json.loads(r["result"] or r["plan"]), "status": r["status"], "finished": r["finished"]} for r in rows]}

    def match_is_visible(self, game_id):
        """Web visibility is evaluated on every request, including cached replays."""
        with self.connect() as db:
            row = db.execute('SELECT plan FROM games WHERE id=?', (game_id,)).fetchone()
        return row is not None and not self.retired_entrant_ids().intersection(json.loads(row['plan'])['entrant_ids'])

    def replay(self, game_id):
        with self.connect() as db:
            row = db.execute("SELECT result,replay FROM games WHERE id=? AND status IN ('complete','failed')", (game_id,)).fetchone()
        if row is None:
            raise ValueError("completed replay not found")
        return json.loads(row[0]), row[1]

    def rebuild(self, *, upgrade=False, progress=None):
        """Re-score recorded outcomes atomically; optionally adopt current rules.

        Migration rewrites scoring metadata only, never maps, actions or replay
        blobs. No archived copy of the old scores is retained.
        """
        from collections import Counter
        from .service import process_lock
        next_manifest = self.config.manifest()
        if upgrade:
            # Rating/scheduler upgrades cannot silently adopt a different engine.
            for key in ('engine_sha256', 'config', 'simulation_backend', 'human_observation', 'decision_timing'):
                if canonical(self.manifest.get(key)) != canonical(next_manifest[key]):
                    raise ValueError('rating upgrade cannot change engine or game configuration')
            for name in ('game.py', 'observation.py', 'rating.py', 'views.py'):
                if self.manifest['implementation'].get(name) != next_manifest['implementation'][name]:
                    raise ValueError('rating upgrade cannot change simulation code')
        else:
            self.verify_execution()
        with process_lock(self, '.server.lock'), process_lock(self, '.worker.lock'), process_lock(self, '.registry.lock'):
            with self.connect(write=True) as db:
                if db.execute("SELECT 1 FROM games WHERE status='live' LIMIT 1").fetchone():
                    raise ValueError('finish or abort live games before rebuilding ratings')
                if 'mode' not in {row['name'] for row in db.execute('PRAGMA table_info(ratings)')}:
                    db.execute('DROP TABLE ratings')
                    db.execute('DROP TABLE IF EXISTS rating_pairs')
                    db.execute(RATING_SCHEMA)
                db.execute(JOB_MODE_SCHEMA)
                if upgrade:
                    rows = db.execute('SELECT id,job_id,plan,result FROM games ORDER BY rowid').fetchall()
                    plans = {row['id']: json.loads(row['plan']) for row in rows}
                    sizes = Counter(rotation_key(plans[row['id']], row['job_id']) for row in rows)
                    for row in rows:
                        plan = plans[row['id']]
                        metadata = dict(rating_policy=POLICY,
                                        rating_group_size=plan.get('rating_group_size', sizes[rotation_key(plan, row['job_id'])]))
                        if 'entrant_ids' in plan:
                            metadata['rating_mode'] = rating_mode(plan)
                        plan.update(metadata)
                        result = json.loads(row['result']) if row['result'] else None
                        if result is not None:
                            result.update(metadata)
                        db.execute('UPDATE games SET plan=?,result=? WHERE id=?',
                                   (canonical(plan), canonical(result) if result is not None else None, row['id']))
                db.execute(PAIR_SCHEMA)
                db.execute('DELETE FROM rating_pairs')
                db.execute("DELETE FROM ratings")
                completed = db.execute("SELECT result FROM games WHERE status='complete' ORDER BY sequence").fetchall()
                for index, row in enumerate(completed, 1):
                    result = json.loads(row[0])
                    self._score(db, result)
                    db.execute("UPDATE games SET result=? WHERE id=?", (canonical(result), result["game_id"]))
                    if progress and (index % 500 == 0 or index == len(completed)):
                        progress(index, len(completed))
                db.execute("INSERT OR REPLACE INTO meta VALUES ('rating_revision',?)", (canonical(uuid.uuid4().hex),))
                if upgrade:
                    db.execute("UPDATE meta SET value=? WHERE key='manifest'", (canonical(next_manifest),))
            if upgrade:
                self.manifest = next_manifest
                from ..research import refresh_publications
                refresh_publications(self)
        return self.leaderboard()
