"""Transactional projections for scheduling coverage and match-history queries.

These contain only match metadata, never simulation state. Triggers also keep
old writers and administrative result/status repairs consistent with readers.
"""

VERSION = 1
TERMINAL = "('complete','aborted','failed')"


def install(db):
    """Install/backfill once inside the caller's write transaction."""
    row = db.execute("SELECT value FROM meta WHERE key='query_indexes'").fetchone()
    if row and int(row[0]) == VERSION:
        return
    statements = [
        """CREATE TABLE match_history(game_rowid INTEGER PRIMARY KEY, status TEXT NOT NULL,
            players INTEGER NOT NULL, mode TEXT NOT NULL, sequence INTEGER, finished REAL)""",
        """CREATE TABLE match_participants(entrant_id TEXT NOT NULL, game_rowid INTEGER NOT NULL,
            copies INTEGER NOT NULL, PRIMARY KEY(entrant_id,game_rowid)) WITHOUT ROWID""",
        "CREATE INDEX match_participants_game ON match_participants(game_rowid,entrant_id)",
        "CREATE INDEX match_history_order ON match_history(sequence DESC,finished DESC)",
        "CREATE INDEX match_history_mode ON match_history(mode,sequence DESC,finished DESC)",
        "CREATE INDEX match_history_players ON match_history(players,sequence DESC,finished DESC)",
        """CREATE TABLE matchup_coverage(mode TEXT NOT NULL, left_id TEXT NOT NULL,
            right_id TEXT NOT NULL, games INTEGER NOT NULL,
            PRIMARY KEY(mode,left_id,right_id)) WITHOUT ROWID""",
        "CREATE INDEX matchup_coverage_right ON matchup_coverage(mode,right_id,left_id)",
        f"""INSERT INTO match_history SELECT rowid,status,json_array_length(plan,'$.entrant_ids'),
            CASE WHEN json_array_length(plan,'$.entrant_ids')=2 THEN '1v1' WHEN json_array_length(plan,'$.entrant_ids')>2 THEN 'ffa' ELSE '' END,
            sequence,finished FROM games WHERE status IN {TERMINAL}""",
        """INSERT INTO match_participants SELECT j.value,g.rowid,count(*) FROM games g
            JOIN match_history h ON h.game_rowid=g.rowid JOIN json_each(g.plan,'$.entrant_ids') j
            GROUP BY g.rowid,j.value""",
        """INSERT INTO matchup_coverage SELECT h.mode,a.entrant_id,b.entrant_id,count(*)
            FROM match_history h JOIN match_participants a ON a.game_rowid=h.game_rowid
            JOIN match_participants b ON b.game_rowid=h.game_rowid AND a.entrant_id<b.entrant_id
            WHERE h.status='complete' GROUP BY h.mode,a.entrant_id,b.entrant_id""",
    ]
    for statement in statements:
        db.execute(statement)
    remove = """
        UPDATE matchup_coverage SET games=games-1
        WHERE OLD.status='complete'
          AND mode=CASE WHEN json_array_length(OLD.plan,'$.entrant_ids')=2 THEN '1v1' WHEN json_array_length(OLD.plan,'$.entrant_ids')>2 THEN 'ffa' ELSE '' END
          AND left_id IN (SELECT entrant_id FROM match_participants WHERE game_rowid=OLD.rowid)
          AND right_id IN (SELECT entrant_id FROM match_participants WHERE game_rowid=OLD.rowid);
        DELETE FROM matchup_coverage WHERE OLD.status='complete' AND games=0
          AND mode=CASE WHEN json_array_length(OLD.plan,'$.entrant_ids')=2 THEN '1v1' WHEN json_array_length(OLD.plan,'$.entrant_ids')>2 THEN 'ffa' ELSE '' END
          AND left_id IN (SELECT entrant_id FROM match_participants WHERE game_rowid=OLD.rowid)
          AND right_id IN (SELECT entrant_id FROM match_participants WHERE game_rowid=OLD.rowid);
        DELETE FROM match_participants WHERE game_rowid=OLD.rowid;
        DELETE FROM match_history WHERE game_rowid=OLD.rowid;
    """
    add = f"""
        INSERT INTO match_history SELECT NEW.rowid,NEW.status,json_array_length(NEW.plan,'$.entrant_ids'),
            CASE WHEN json_array_length(NEW.plan,'$.entrant_ids')=2 THEN '1v1' WHEN json_array_length(NEW.plan,'$.entrant_ids')>2 THEN 'ffa' ELSE '' END,
            NEW.sequence,NEW.finished WHERE NEW.status IN {TERMINAL};
        INSERT INTO match_participants SELECT value,NEW.rowid,count(*)
            FROM json_each(NEW.plan,'$.entrant_ids') WHERE NEW.status IN {TERMINAL} GROUP BY value;
        INSERT INTO matchup_coverage SELECT h.mode,a.entrant_id,b.entrant_id,1
            FROM match_history h JOIN match_participants a ON a.game_rowid=h.game_rowid
            JOIN match_participants b ON b.game_rowid=h.game_rowid AND a.entrant_id<b.entrant_id
            WHERE h.game_rowid=NEW.rowid AND h.status='complete'
            ON CONFLICT(mode,left_id,right_id) DO UPDATE SET games=games+1;
    """
    # execute(), not executescript(): DDL/backfill/marker must commit atomically.
    db.execute(f"""CREATE TRIGGER match_index_insert AFTER INSERT ON games
        WHEN NEW.status IN {TERMINAL} BEGIN {add} END""")
    db.execute(f"""CREATE TRIGGER match_index_update AFTER UPDATE OF status,plan,sequence,finished ON games
        WHEN OLD.status IN {TERMINAL} OR NEW.status IN {TERMINAL}
        BEGIN {remove} {add} END""")
    db.execute(f"""CREATE TRIGGER match_index_delete AFTER DELETE ON games
        WHEN OLD.status IN {TERMINAL} BEGIN {remove} END""")
    db.execute("INSERT OR REPLACE INTO meta VALUES ('query_indexes',?)", (str(VERSION),))
