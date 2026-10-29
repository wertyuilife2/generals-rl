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

from generals_arena.policies.entries import prepare_entrant
from generals_arena.policies.native import decision_timeout
from generals_arena.matches.results import seat_outcome
from generals_arena.util import canonical, integer, process_lock
from generals_arena.competitions.config import LeagueConfig
from generals_arena.competitions.rating import (CF_SCALE, FFA_MEAN, MODES, POLICY, PROVISIONAL_GAMES,
                                                initial_rating, ranks, rating_mode, score, validate_mode)

SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE entrants(id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE accounts(token_hash TEXT PRIMARY KEY, entrant_id TEXT UNIQUE REFERENCES entrants(id));
CREATE TABLE jobs(id TEXT PRIMARY KEY, entrant_id TEXT NOT NULL REFERENCES entrants(id),
 budget INTEGER NOT NULL, done INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'queued',
 error TEXT, created REAL NOT NULL);
CREATE TABLE games(id TEXT PRIMARY KEY, job_id TEXT REFERENCES jobs(id), status TEXT NOT NULL,
 plan TEXT NOT NULL, result TEXT, replay BLOB, sequence INTEGER UNIQUE, created REAL NOT NULL, finished REAL);
CREATE TABLE ratings(mode TEXT NOT NULL, entrant_id TEXT NOT NULL REFERENCES entrants(id), elo REAL NOT NULL,
 games INTEGER NOT NULL, rated_games INTEGER NOT NULL, appearances INTEGER NOT NULL, wins INTEGER NOT NULL, losses INTEGER NOT NULL, draws INTEGER NOT NULL,
 PRIMARY KEY(mode,entrant_id));
CREATE INDEX games_status ON games(status, created);
CREATE INDEX games_completed_players ON games(status,json_array_length(plan, '$.entrant_ids'),sequence);
"""


class CompetitionStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / "league.sqlite3"
        if not self.path.is_file():
            raise ValueError("league does not exist; run generals-arena league init first")
        with self.connect() as db:
            meta = {row["key"]: json.loads(row["value"]) for row in db.execute("SELECT * FROM meta")}
        self.manifest = meta["manifest"]
        from generals.protocol.rules import RULES_VERSION
        if self.manifest.get("schema") != 7 or self.manifest.get('rules') != RULES_VERSION:
            raise ValueError("this league uses archived rules; initialize a new league directory")
        self.config = LeagueConfig(**self.manifest["config"])

    @property
    def reference(self):
        with self.connect() as db:
            row = db.execute("SELECT value FROM meta WHERE key='rating_reference'").fetchone()
        return json.loads(row[0])

    @classmethod
    def create(cls, directory, config=None, *, reference="wait"):
        if reference not in ("wait", "random"):
            raise ValueError("rating reference must be wait or random")
        config = config or LeagueConfig()
        manifest = config.manifest()
        random = prepare_entrant({"name": "random", "kind": "baseline", "policy": "random"})
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
                               [("manifest", canonical(manifest)), ("rating_reference", canonical(random["entrant_id"]))])
                db.execute("INSERT INTO entrants VALUES (?,?,?,?)",
                           (random["entrant_id"], random["name"], random["kind"], canonical(random)))
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        store = cls(directory)
        if reference == "wait":
            from generals_arena.policies.builtin import wait_spec
            entry = store.register(wait_spec())
            store.set_rating_reference(entry["entrant_id"])
        return store

    def set_rating_reference(self, entrant_id):
        """Choose the displayed 1v1 reference before the first rated game."""
        with self.connect(write=True) as db:
            if not db.execute('SELECT 1 FROM entrants WHERE id=?', (entrant_id,)).fetchone():
                raise ValueError('rating reference must be a registered entrant')
            reference = json.loads(db.execute("SELECT value FROM meta WHERE key='rating_reference'").fetchone()[0])
            if entrant_id != reference and db.execute("SELECT 1 FROM games WHERE status='complete' LIMIT 1").fetchone():
                raise ValueError('choose the 1v1 display reference before rated games; start a new league to change it')
            db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',
                       ('rating_reference', canonical(entrant_id)))

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
        if canonical(self.config.manifest()) != canonical(self.manifest):
            raise ValueError("league rules/source changed; use its original code or initialize a new league")

    def resize_maps(self, min_side, max_side, side_cap):
        """Explicit map revision; preserve accounts, results, and rating history.

        Only distribution/presentation may change here. A different engine,
        scoring algorithm or scheduler still requires its own league.
        """
        config = replace(self.config, min_side=min_side, max_side=max_side, side_cap=side_cap)
        manifest = config.manifest()
        previous = self.manifest
        for key in ("schema", "engine_sha256", "rating", "decision_timing", "matches_sha256", "competitions_sha256", "policies_sha256"):
            if previous.get(key) != manifest[key]:
                raise ValueError("map resize cannot change the engine or rating rules")
        for name in ("matches/game.py", "matches/observation.py", "competitions/rating.py", "competitions/matchmaking.py"):
            if previous["implementation"].get(name) != manifest["implementation"][name]:
                raise ValueError("map resize cannot change simulation, scoring or scheduling code")
        with process_lock(self.directory, ".server.lock"), process_lock(self.directory, ".worker.lock"):
            with self.connect(write=True) as db:
                current = json.loads(db.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()[0])
                if canonical(current) != canonical(previous):
                    raise ValueError("league changed; reopen it before resizing")
                if db.execute("SELECT 1 FROM games WHERE status IN ('live','pending') LIMIT 1").fetchone():
                    raise ValueError("finish active and pending games before resizing maps")
                for row in db.execute("SELECT data FROM entrants"):
                    self._validate_package_profile(json.loads(row["data"]), config)
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
                if key == self.reference:
                    raise ValueError("the rating anchor/reference cannot be retired")
                if key in current:
                    return {'entrant_id': key, 'name': row['name'], 'retired': True, **current[key]}
                active = db.execute("""SELECT 1 FROM games,json_each(games.plan,'$.entrant_ids') participant
                    WHERE games.status IN ('pending','live') AND participant.value=? LIMIT 1""", (key,)).fetchone()
                queued = db.execute("SELECT 1 FROM jobs WHERE entrant_id=? AND status IN ('queued','running') LIMIT 1", (key,)).fetchone()
                if active or queued:
                    raise ValueError("entrant has queued jobs or pending/live games; finish them before retirement")
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

    @staticmethod
    def _validate_package_profile(entrant, config):
        """Require the complete league map distribution to fit the policy."""
        if entrant["kind"] != "package":
            return
        from generals_arena.policies.artifacts import capabilities_supports
        from generals_arena.policies.artifacts import validate_bundle
        info = validate_bundle(entrant["factory_kwargs"]["bundle"])
        capabilities = info["manifest"]["capabilities"]
        for players in config.player_counts:
            if players not in capabilities["player_counts"]:
                raise ValueError(f"policy package does not support league player count {players}")
            low, high = config.side_range(players)
            compatible = all(capabilities_supports(capabilities, width, height, players)
                             for width in range(low, high + 1)
                             for height in range(low, high + 1))
            if not compatible:
                raise ValueError(f"policy package does not support the complete league map range "
                                 f"{low}..{high} for {players} players")

    def validate_entrant_rules(self, spec, *, base_dir=Path(".")):
        """Preflight package registration without creating an entrant or job.

        The returned prepared package entrant can be reused by registration. Existing
        other policy paths stay unchanged and need no optional inference dependencies.
        """
        if not isinstance(spec, dict):
            raise ValueError("entrant specification must be a JSON object")
        if spec.get("kind") != "package":
            return None
        self.verify_execution()
        entrant = prepare_entrant(spec, base_dir=base_dir)
        self._validate_package_profile(entrant, self.config)
        return entrant

    def _verify_package_registration_manifest(self, db, entrant):
        if entrant["kind"] == "package":
            current = json.loads(db.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()[0])
            if canonical(current) != canonical(self.manifest):
                raise ValueError("league changed; reopen it before registering the policy package")

    def add(self, spec, *, base_dir=Path("."), games=1000):
        self.verify_execution()
        integer("games", games, 1)
        if not isinstance(spec, dict):
            raise ValueError("entrant specification must be a JSON object")
        entrant = self.validate_entrant_rules(spec, base_dir=base_dir)
        if entrant is None:
            entrant = prepare_entrant(spec, base_dir=base_dir)
        with self.connect(write=True) as db:
            self._verify_package_registration_manifest(db, entrant)
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
        return entrant, job_id

    def register(self, spec, *, base_dir=Path('.')):
        """Register a private-evaluation opponent without scheduling its own job."""
        self.verify_execution()
        entrant = self.validate_entrant_rules(spec, base_dir=base_dir)
        if entrant is None:
            entrant = prepare_entrant(spec, base_dir=base_dir)
        with self.connect(write=True) as db:
            self._verify_package_registration_manifest(db, entrant)
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
            return [dict(r) for r in db.execute("SELECT jobs.*, entrants.name FROM jobs JOIN entrants ON entrants.id=jobs.entrant_id ORDER BY created")]

    def retry(self, job_id=None):
        with self.connect(write=True) as db:
            row = db.execute("SELECT value FROM meta WHERE key='retired_entrants'").fetchone()
            retired = set(json.loads(row[0])) if row else set()
            jobs = db.execute("SELECT j.id,j.entrant_id,e.kind FROM jobs j JOIN entrants e ON e.id=j.entrant_id "
                              "WHERE j.status='failed'").fetchall()
            for job in jobs:
                if job['entrant_id'] in retired or (job_id is not None and job['id'] != job_id):
                    continue
                failed = db.execute("SELECT g.plan,g.result FROM games g WHERE g.job_id=? AND g.status='failed' "
                                    "AND NOT EXISTS (SELECT 1 FROM games r WHERE json_extract(r.plan,'$.retry_of')=g.id)",
                                    (job['id'],)).fetchall()
                if any(retired.intersection(json.loads(row['plan'])['entrant_ids']) for row in failed):
                    continue
                failures = [json.loads(row['result'] or '{}') for row in failed]
                if job['kind'] != 'cpp' or (failures and all(
                        r.get('error_detail', {}).get('player') is None or decision_timeout(r.get('error_detail', {})) for r in failures)):
                    db.execute("UPDATE jobs SET status='queued',error=NULL WHERE id=?", (job['id'],))
                    for row in failed:
                        # Keep the failed attempt and its replay as evidence.
                        # Retry the same seeds/seats with a distinct attempt ID.
                        plan = json.loads(row['plan'])
                        plan.update(retry_of=plan['game_id'], game_id=uuid.uuid4().hex)
                        db.execute("INSERT INTO games(id,job_id,status,plan,created) VALUES (?,?,'pending',?,?)",
                                   (plan['game_id'], job['id'], canonical(plan), time.time()))

    def fail(self, job_id, error):
        with self.connect(write=True) as db:
            db.execute("UPDATE jobs SET status='failed',error=? WHERE id=?", (str(error), job_id))

    def insert_plans(self, plans, job_id=None, status="pending"):
        with self.connect(write=True) as db:
            row = db.execute("SELECT value FROM meta WHERE key='retired_entrants'").fetchone()
            retired = set(json.loads(row[0])) if row else set()
            for plan in plans:
                if retired.intersection(plan['entrant_ids']):
                    raise ValueError('entrant was retired before scheduling; restore it or regenerate the schedule')
                db.execute("INSERT INTO games(id,job_id,status,plan,created) VALUES (?,?,?,?,?)",
                           (plan["game_id"], job_id, status, canonical(plan), time.time()))

    def rating_modes(self):
        return tuple(mode for mode in MODES if any((count == 2) == (mode == '1v1')
                                                   for count in self.config.player_counts))

    def default_mode(self):
        return self.rating_modes()[0]

    @staticmethod
    def _mode_predicate(mode):
        validate_mode(mode)
        return "json_array_length(plan, '$.entrant_ids')" + ("=2" if mode == '1v1' else ">2")

    def ratings(self, mode=None):
        """Raw model coordinates; never feed normalized GoR display values back."""
        mode = validate_mode(mode or self.default_mode())
        with self.connect() as db:
            return {r["entrant_id"]: r["elo"] for r in db.execute("SELECT * FROM ratings WHERE mode=?", (mode,))}

    def pair_counts(self, target, mode=None):
        mode = validate_mode(mode or self.default_mode())
        counts = {}
        with self.connect() as db:
            for row in db.execute("SELECT plan FROM games WHERE status='complete' AND " + self._mode_predicate(mode)):
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
        result.setdefault('rating_policy', POLICY)
        rows = list(db.execute("SELECT * FROM ratings WHERE mode=?", (mode,)))
        old = {r["entrant_id"]: r["elo"] for r in rows}
        games = {r["entrant_id"]: r["rated_games"] for r in rows}
        changes = score(old, result, games)
        result["ranks"] = ranks(result)
        result["rating_changes"] = changes
        for key in dict.fromkeys(ids):
            seats = [seat for seat, entrant in enumerate(ids) if entrant == key]
            appearances = len(seats)
            outcomes = [seat_outcome(result, seat) for seat in seats]
            win, loss, draw = (outcomes.count(outcome) for outcome in ('win', 'loss', 'draw'))
            db.execute("""INSERT INTO ratings VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(mode,entrant_id)
                DO UPDATE SET elo=excluded.elo,games=games+1,rated_games=excluded.rated_games,appearances=appearances+excluded.appearances,
                wins=wins+excluded.wins,losses=losses+excluded.losses,draws=draws+excluded.draws""",
                       (mode, key, changes[key]["after"], 1, changes[key]["rated_games"], appearances, win, loss, draw))

    def complete(self, game):
        return self.complete_result(game.result(), game.replay_blob())

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
        with self.connect() as db:
            rows = [dict(row) for row in db.execute("""SELECT e.id entrant_id,e.name,e.kind,
                COALESCE(r.elo,?) raw_elo,COALESCE(r.games,0) games,COALESCE(r.rated_games,0) rated_games,COALESCE(r.appearances,0) appearances,
                COALESCE(r.wins,0) wins,COALESCE(r.losses,0) losses,COALESCE(r.draws,0) draws FROM entrants e
                LEFT JOIN ratings r ON r.entrant_id=e.id AND r.mode=? ORDER BY raw_elo DESC,e.name""", (initial_rating(mode), mode))]
            completed = db.execute("SELECT count(*) FROM games WHERE status='complete' AND " + self._mode_predicate(mode)).fetchone()[0]
        retired = self.retired_entrants()
        reference = self.reference
        offset = next(row['raw_elo'] for row in rows if row['entrant_id'] == reference) if mode == '1v1' else 0.
        disabled = self.disabled_policy_ids()
        for row in rows:
            row['elo'] = row['raw_elo'] - offset
            if row['entrant_id'] in retired:
                row['retired'] = True
            row['elo_ci'] = None
            row['ci_blocks'] = 0
            row["anchor"] = mode == '1v1' and row["entrant_id"] == reference
            row["provisional"] = row["rated_games"] < PROVISIONAL_GAMES
            if row['entrant_id'] in disabled:
                row['disqualified'] = True
        if not include_retired:
            rows = [row for row in rows if not row.get("retired")]
        return {"mode": mode, "rating_mode": mode,
                "player_counts": tuple(p for p in self.config.player_counts if (p == 2) == (mode == '1v1')),
                "ranking": rows, "anchor": reference if mode == '1v1' else None,
                "rating_model": 'egf-gor' if mode == '1v1' else 'codeforces',
                "rating_offset": offset, "rating_mean": FFA_MEAN if mode == 'ffa' else None,
                "scale": CF_SCALE if mode == 'ffa' else None, "completed_games": completed}

    def leaderboards(self, *, include_retired=False):
        return {mode: self.leaderboard(mode, include_retired=include_retired) for mode in self.rating_modes()}

    def matches(self, limit=50, offset=0, players=None, entrant=None, opponent=None, mode=None):
        integer("limit", limit, 1)
        integer("offset", offset)
        if limit > 100:
            raise ValueError("limit must be <= 100")
        where, values = "status IN ('complete','aborted','failed')", []
        if mode is not None:
            where += " AND " + self._mode_predicate(mode)
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
            total = db.execute(f"SELECT count(*) FROM games WHERE {where}", values).fetchone()[0]
            rows = db.execute(f"SELECT id,status,plan,result,finished FROM games WHERE {where} ORDER BY finished DESC LIMIT ? OFFSET ?", (*values, limit, offset)).fetchall()
        return {"total": total, "matches": [{**json.loads(r["result"] or r["plan"]), "status": r["status"], "finished": r["finished"]} for r in rows]}

    def replay(self, game_id):
        with self.connect() as db:
            row = db.execute("SELECT result,replay FROM games WHERE id=? AND status IN ('complete','failed')", (game_id,)).fetchone()
        if row is None:
            raise ValueError("completed replay not found")
        return json.loads(row[0]), row[1]

    def rebuild(self):
        """Recompute both current rating pools atomically from recorded games."""
        self.verify_execution()
        with process_lock(self.directory, '.server.lock'), process_lock(self.directory, '.worker.lock'), process_lock(self.directory, '.registry.lock'):
            with self.connect(write=True) as db:
                if db.execute("SELECT 1 FROM games WHERE status='live' LIMIT 1").fetchone():
                    raise ValueError('finish or abort live games before rebuilding ratings')
                db.execute("DELETE FROM ratings")
                for row in db.execute("SELECT result FROM games WHERE status='complete' ORDER BY sequence").fetchall():
                    result = json.loads(row[0])
                    self._score(db, result)
                    db.execute("UPDATE games SET result=? WHERE id=?", (canonical(result), result["game_id"]))
                db.execute("INSERT OR REPLACE INTO meta VALUES ('rating_revision',?)", (canonical(uuid.uuid4().hex),))
        return self.leaderboard()
