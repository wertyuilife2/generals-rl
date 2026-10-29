"""Local web UI and JSON API. No web framework or build tool is required."""
from dataclasses import asdict
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlsplit

from generals_arena.web.replays import ReplayResponses
from generals_arena.web.live import LiveArena
from generals_arena.util import process_lock
from generals_arena.competitions.store import CompetitionStore
from generals_arena.competitions.leaderboard import LeaderboardStatistics
from generals_arena.competitions.rating import validate_mode
from generals_arena.util import digest

STATIC = Path(__file__).parent


def make_server(store, host="127.0.0.1", port=8086):
    arena = LiveArena(store)
    statistics = {mode: LeaderboardStatistics(store, mode) for mode in store.rating_modes()}
    for stats in statistics.values():
        stats.leaderboard()

    replays = ReplayResponses(store)

    class Handler(BaseHTTPRequestHandler):
        server_version = "GeneralsArena/1"

        def log_message(self, *args):
            pass

        def log_error(self, message, *args):
            print(message % args, file=sys.stderr)

        def respond(self, value, status=200, content_type="application/json; charset=utf-8", *, compressed=False):
            data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            accepts_gzip = "gzip" in self.headers.get("Accept-Encoding", "")
            if compressed or len(data) > 2048:
                self.send_header("Vary", "Accept-Encoding")
                if accepts_gzip:
                    data = data if compressed else gzip.compress(data, compresslevel=3)
                    self.send_header("Content-Encoding", "gzip")
                elif compressed:
                    data = gzip.decompress(data)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def human(self):
            auth = self.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                raise PermissionError("player token required")
            return store.authenticate(auth[7:])

        def request_data(self):
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("send application/json")
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 8192:
                raise ValueError("request body must be 1–8192 bytes")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("request must be a JSON object")
            return value

        def dispatch(self, post=False):
            try:
                # Loopback service: reject DNS rebinding and cross-origin writes.
                host_header = self.headers.get("Host", "")
                if host_header not in self.server.allowed_hosts:
                    raise PermissionError("unrecognized Host header")
                origin = self.headers.get("Origin")
                if origin is not None and origin != "http://" + host_header:
                    raise PermissionError("cross-origin requests are not allowed")
                url = urlsplit(self.path)
                path = url.path.rstrip("/") or "/"
                query = parse_qs(url.query)
                players = int(query["players"][0]) if "players" in query else None
                if post:
                    data = self.request_data()
                    if path == "/api/players":
                        store.verify_execution()
                        self.respond(store.account(data["name"]), 201)
                    elif path == "/api/play":
                        self.respond(arena.create_game(self.human(), data.get("players")), 201)
                    elif path.startswith("/api/play/") and path.endswith("/commands"):
                        self.respond(arena.command(path.split("/")[3], self.human(), data))
                    else:
                        self.respond({"error": "not found"}, 404)
                elif path == "/api/meta":
                    entries = store.entrants()
                    self.respond({"config": asdict(store.config), "anchor": store.reference,
                                  "rating_modes": store.rating_modes(), "default_rating_mode": store.default_mode(),
                                  "league_id": digest(str(store.directory.resolve())),
                                  "rules_id": store.manifest["rules_id"],
                                  "entrants": [{k: e[k] for k in ("entrant_id", "name", "kind", "version")} for e in entries.values()],
                                  "map_ranges": {p: store.config.side_range(p) for p in store.config.player_counts},
                                  "jobs": store.jobs(), "worker_error": arena.worker_error})
                elif path == "/api/leaderboard":
                    mode = validate_mode(query.get("mode", [store.default_mode()])[0])
                    if mode not in statistics:
                        raise ValueError("rating pool is not enabled for this league")
                    self.respond(statistics[mode].leaderboard())
                elif path == "/api/matches":
                    self.respond(store.matches(limit=int(query.get("limit", ["20"])[0]),
                                               offset=int(query.get("offset", ["0"])[0]), players=players,
                                               entrant=query.get('entrant', [None])[0],
                                               opponent=query.get('opponent', [None])[0]))
                elif path.startswith("/api/replay/"):
                    self.respond(replays.get(path.split("/")[-1]), compressed=True)
                elif path == "/api/me":
                    self.respond(store.entrants()[self.human()])
                elif path == "/api/play/active":
                    self.respond(arena.active(self.human()))
                elif path.startswith("/api/play/"):
                    self.respond(arena.view(path.split("/")[-1], self.human(), int(query["after"][0]) if "after" in query else None))
                elif path in ("/", "/app.js", "/board.js", "/playback.js", "/input.js", "/svgIcons.js", "/style.css"):
                    filename = "index.html" if path == "/" else path[1:]
                    mime = "text/html" if filename.endswith(".html") else "text/css" if filename.endswith(".css") else "text/javascript"
                    self.respond((STATIC / filename).read_bytes(), content_type=mime + "; charset=utf-8")
                else:
                    self.respond({"error": "not found"}, 404)
            except PermissionError as exc:
                self.respond({"error": str(exc)}, 403)
            except (ValueError, TypeError, KeyError) as exc:
                self.respond({"error": str(exc)}, 400)
            except Exception as exc:
                self.log_error("request failed: %s", exc)
                self.respond({"error": "server error; check league data and worker status"}, 500)

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.dispatch(True)

    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("this local arena supports --host 127.0.0.1 or localhost")
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    bound_port = server.server_address[1]
    server.allowed_hosts = {f"127.0.0.1:{bound_port}", f"localhost:{bound_port}"}
    server.arena = arena
    return server


def serve(directory, host="127.0.0.1", port=8086, *, run_worker=True):
    store = CompetitionStore(directory)
    with process_lock(store.directory, ".server.lock"):
        server = make_server(store, host, port)
        try:
            server.arena.start(run_worker=run_worker)
            print(f"Arena: http://{host}:{server.server_address[1]} (Ctrl-C to stop)", flush=True)
            server.serve_forever(poll_interval=.2)
        except KeyboardInterrupt:
            pass
        finally:
            server.arena.close()
            server.server_close()
