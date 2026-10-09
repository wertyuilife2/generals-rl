"""SQLite transactions keep results, compressed traces, and ratings consistent."""
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import secrets
import sqlite3
import time
import uuid

from ..entrants import prepare_entrant
from ..util import canonical, integer
from .config import LeagueConfig
from .rating import SCALE, update

SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE entrants(id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE accounts(token_hash TEXT PRIMARY KEY, entrant_id TEXT UNIQUE REFERENCES entrants(id));
CREATE TABLE jobs(id TEXT PRIMARY KEY, entrant_id TEXT NOT NULL REFERENCES entrants(id),
 budget INTEGER NOT NULL, done INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'queued',
 error TEXT, created REAL NOT NULL);
CREATE TABLE games(id TEXT PRIMARY KEY, job_id TEXT REFERENCES jobs(id), status TEXT NOT NULL,
 plan TEXT NOT NULL, result TEXT, replay BLOB, sequence INTEGER UNIQUE, created REAL NOT NULL, finished REAL);
CREATE TABLE ratings(players INTEGER, entrant_id TEXT REFERENCES entrants(id), elo REAL NOT NULL,
 games INTEGER NOT NULL, wins INTEGER NOT NULL, losses INTEGER NOT NULL, draws INTEGER NOT NULL,
 PRIMARY KEY(players, entrant_id));
CREATE INDEX games_status ON games(status, created);
"""


class LeagueStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / "league.sqlite3"
        if not self.path.is_file():
            raise ValueError("league does not exist; run generals-arena league init first")
        with self.connect() as db:
            meta = {row["key"]: json.loads(row["value"]) for row in db.execute("SELECT * FROM meta")}
        self.manifest = meta["manifest"]
        self.config = LeagueConfig(**self.manifest["config"])
        self.anchor = meta["anchor"]

    @classmethod
    def create(cls, directory, config=None):
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
        return cls(directory)

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

    def entrants(self):
        with self.connect() as db:
            return {r["id"]: json.loads(r["data"]) for r in db.execute("SELECT * FROM entrants ORDER BY name")}

    def add(self, spec, *, base_dir=Path("."), games=1000):
        self.verify_execution()
        integer("games", games, 1)
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
                return entrant, job["id"] if job else None  # Idempotent registration.
            db.execute("INSERT INTO entrants VALUES (?,?,?,?)",
                       (entrant["entrant_id"], entrant["name"], entrant["kind"], canonical(entrant)))
            job_id = uuid.uuid4().hex
            db.execute("INSERT INTO jobs(id,entrant_id,budget,created) VALUES (?,?,?,?)",
                       (job_id, entrant["entrant_id"], games, time.time()))
        return entrant, job_id

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

    def retry(self):
        with self.connect(write=True) as db:
            db.execute("UPDATE jobs SET status='queued',error=NULL WHERE status='failed'")

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

    def ratings(self, players):
        with self.connect() as db:
            return {r["entrant_id"]: r["elo"] for r in db.execute("SELECT * FROM ratings WHERE players=?", (players,))}

    def pair_counts(self, target, players):
        counts = {}
        with self.connect() as db:
            for row in db.execute("SELECT plan FROM games WHERE status='complete'"):
                ids = json.loads(row[0])["entrant_ids"]
                if len(ids) == players and target in ids:
                    for key in ids:
                        if key != target:
                            counts[key] = counts.get(key, 0) + 1
        return counts

    def _score(self, db, result):
        players = len(result["entrant_ids"])
        old = {r["entrant_id"]: r["elo"] for r in db.execute("SELECT * FROM ratings WHERE players=?", (players,))}
        changed = update(old, result, self.anchor, self.config.k_factor)
        for seat, key in enumerate(result["entrant_ids"]):
            winner = result["winner_seat"]
            win, loss, draw = int(winner == seat), int(winner is not None and winner != seat), int(winner is None)
            db.execute("""INSERT INTO ratings VALUES (?,?,?,?,?,?,?) ON CONFLICT(players,entrant_id)
                DO UPDATE SET elo=excluded.elo,games=games+1,wins=wins+excluded.wins,
                losses=losses+excluded.losses,draws=draws+excluded.draws""",
                       (players, key, changed[key], 1, win, loss, draw))

    def complete(self, game):
        result, blob = game.result(), game.replay_blob()
        with self.connect(write=True) as db:
            row = db.execute("SELECT * FROM games WHERE id=?", (result["game_id"],)).fetchone()
            if row is None:
                raise ValueError("game was never scheduled")
            if row["status"] == "complete":
                return json.loads(row["result"])
            if row["status"] not in ("pending", "live") or json.loads(row["plan"]) != game.plan:
                raise ValueError("game plan/status mismatch")
            sequence = db.execute("SELECT COALESCE(MAX(sequence),-1)+1 FROM games").fetchone()[0]
            result["sequence"] = sequence
            db.execute("UPDATE games SET status='complete',result=?,replay=?,sequence=?,finished=? WHERE id=?",
                       (canonical(result), blob, sequence, time.time(), result["game_id"]))
            self._score(db, result)
            if row["job_id"]:
                db.execute("UPDATE jobs SET done=done+1,status=CASE WHEN done+1>=budget THEN 'complete' ELSE 'queued' END WHERE id=?", (row["job_id"],))
        return result

    def abort_live(self, game_id=None):
        with self.connect(write=True) as db:
            if game_id is None:
                db.execute("UPDATE games SET status='aborted',finished=? WHERE status='live'", (time.time(),))
            else:
                db.execute("UPDATE games SET status='aborted',finished=? WHERE status='live' AND id=?", (time.time(), game_id))

    def leaderboard(self, players):
        if players not in self.config.player_counts:
            raise ValueError("player count is not enabled")
        with self.connect() as db:
            rows = [dict(row) for row in db.execute("""SELECT e.id entrant_id,e.name,e.kind,
                COALESCE(r.elo,0) elo,COALESCE(r.games,0) games,COALESCE(r.wins,0) wins,
                COALESCE(r.losses,0) losses,COALESCE(r.draws,0) draws FROM entrants e
                LEFT JOIN ratings r ON r.entrant_id=e.id AND r.players=? ORDER BY elo DESC,e.name""", (players,))]
        for row in rows:
            row["anchor"] = row["entrant_id"] == self.anchor
            row["provisional"] = row["games"] < 30
        return {"players": players, "ranking": rows, "anchor": self.anchor, "scale": SCALE,
                "completed_games": sum(row["games"] for row in rows) // players}

    def matches(self, limit=50, offset=0, players=None):
        integer("limit", limit, 1)
        integer("offset", offset)
        if limit > 100:
            raise ValueError("limit must be <= 100")
        where, values = "status IN ('complete','aborted')", []
        if players is not None:
            where += " AND json_array_length(plan, '$.entrant_ids')=?"
            values.append(players)
        with self.connect() as db:
            total = db.execute(f"SELECT count(*) FROM games WHERE {where}", values).fetchone()[0]
            rows = db.execute(f"SELECT id,status,plan,result,finished FROM games WHERE {where} ORDER BY finished DESC LIMIT ? OFFSET ?", (*values, limit, offset)).fetchall()
        return {"total": total, "matches": [{**json.loads(r["result"] or r["plan"]), "status": r["status"], "finished": r["finished"]} for r in rows]}

    def replay(self, game_id):
        with self.connect() as db:
            row = db.execute("SELECT result,replay FROM games WHERE id=? AND status='complete'", (game_id,)).fetchone()
        if row is None:
            raise ValueError("completed replay not found")
        return json.loads(row[0]), row[1]

    def rebuild(self):
        with self.connect(write=True) as db:
            db.execute("DELETE FROM ratings")
            for row in db.execute("SELECT result FROM games WHERE status='complete' ORDER BY sequence").fetchall():
                self._score(db, json.loads(row[0]))
        return {p: self.leaderboard(p) for p in self.config.player_counts}
