"""One serial match process: direct C++ policy calls and the existing env engine.

Only match results (or 2 Hz human turns) cross the process boundary. A shared
action journal preserves partial replays if a native watchdog exits the child.
There are no per-policy processes, worker threads or per-decision messages.
"""
import mmap
import multiprocessing
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace

import numpy as np
from generals import Action
from generals.core.recording import state_to_dict, state_from_dict

from . import native
from .execution import close_game, timing, result, failure
from .league.game import RecordedGame


class Journal:
    def __init__(self, path, ticks, players, create=False):
        self.path, self.players = Path(path), players
        size = (4+8*8+ticks*players)*8
        self.file = self.path.open('w+b' if create else 'r+b')
        if create:
            self.file.truncate(size)
        self.memory = mmap.mmap(self.file.fileno(), size)
        self.header = np.ndarray((4,), np.int64, self.memory)
        self.records = np.ndarray((8,8), np.int64, self.memory, offset=32)
        self.actions = np.ndarray((ticks,players), np.int64, self.memory, offset=(4+64)*8)
        if create:
            self.header[:] = [0,-1,0,0]
            self.records[:] = 0

    def record(self, game):
        self.actions[game.state.tick-1] = [Action(*a).encode() for a in game.actions[-1]]
        self.header[0] = game.state.tick  # Publish after the entire action row.

    def recover(self, plan, config):
        game = RecordedGame(plan, config, [None]*self.players)
        for row in self.actions[:int(self.header[0])]:
            game.advance({i:Action.decode(int(a)) for i,a in enumerate(row)})
        game.controllers = [SimpleNamespace(decision_metrics=native.metrics(self.records[i]))
                            if self.records[i,0] or i == self.header[1] else None
                            for i in range(self.players)]
        return game

    def close(self):
        self.header = self.records = self.actions = None
        self.memory.close()
        self.file.close()


def snapshot(game):
    return {'state':state_to_dict(game.state), 'initial_digest':game.initial_digest,
            'done':game.done, 'timing':timing(game)}


def child_loop(connection):
    # Policy printing must not corrupt the CLI's JSON output.
    with open(os.devnull, 'w') as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    from .league.service import controllers_for
    from .league.store import LeagueStore
    game, journal, controllers = None, None, []
    try:
        while True:
            request = connection.recv()
            op = request['op']
            if op in ('close','shutdown','start','run'):
                if game is not None:
                    close_game(game)
                    game = None
                if journal is not None:
                    native.DIAGNOSTICS = None
                    journal.close()
                    journal = None
            if op == 'shutdown':
                break
            if op == 'close':
                connection.send({'closed':True})
                continue
            try:
                if op in ('start','run'):
                    store = LeagueStore(request['directory'])
                    plan = request['plan']
                    journal = Journal(request['journal'], store.config.max_ticks, len(plan['entrant_ids']))
                    native.DIAGNOSTICS = journal
                    controllers = controllers_for(store, plan)
                    game = RecordedGame(plan, store.config, controllers)
                if op == 'run':
                    while not game.done:
                        game.advance()
                        journal.record(game)
                elif op == 'advance':
                    game.advance({int(k):Action.decode(v) for k,v in request['actions'].items()})
                    journal.record(game)
                response = {'snapshot':snapshot(game)}
                if op in ('run','replay') or game.done:
                    response['blob'] = game.replay_blob()
                if game.done:
                    response['result'] = result(game)
                if op == 'run':
                    # Finish this game's teardown before the coordinator moves
                    # to a new journal. A broken destructor belongs to this game.
                    close_game(game)
                    game = None
                connection.send(response)
            except Exception as exc:
                if game is None:
                    game = RecordedGame(plan, store.config, [None]*len(plan['entrant_ids']))
                    game.controllers = controllers
                connection.send({'failure':str(exc), 'details':getattr(exc,'details',{}),
                    'snapshot':snapshot(game), 'result':failure(game,exc), 'blob':game.replay_blob()})
    except (EOFError, BrokenPipeError):
        pass
    finally:
        if game is not None:
            close_game(game)
        connection.close()


class MatchWorker:
    def __init__(self):
        self.process = self.connection = self.journal = None

    def _start_process(self):
        if self.process is not None and self.process.is_alive():
            return
        if self.process is not None:
            self.process.join()
            self.connection.close()
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=child_loop,args=(child,),daemon=True)
        self.process.start()
        child.close()

    def begin(self, store, plan, run=False):
        self._start_process()
        if self.journal is not None:
            path = self.journal.path
            self.journal.close()
            path.unlink(missing_ok=True)
        self.plan, self.config = plan, store.config
        fd, path = tempfile.mkstemp(prefix='match-',dir=native._private_cache())
        os.close(fd)
        self.journal = Journal(path, store.config.max_ticks, len(plan['entrant_ids']), create=True)
        return self.call({'op':'run' if run else 'start', 'plan':plan,
            'directory':str(store.directory.resolve()), 'journal':path})

    def _failed(self):
        self.process.join(timeout=2)
        seat, tick, stage = map(int,self.journal.header[1:4])
        record = self.journal.records[seat] if seat >= 0 else None
        timeout = record is not None and (record[7] == 1 or self.process.exitcode in (-signal.SIGALRM,128+signal.SIGALRM))
        if record is not None and not (record[4] or record[5]):
            record[4 if timeout else 5] += 1
        elapsed = int(record[6]) if record is not None else 0
        details = {'kind':'timeout' if timeout else 'policy_error', 'player':seat if seat >= 0 else None,
                   'tick':tick, 'elapsed_ns':elapsed, 'limit_ns':native.DECISION_NS if stage==2 else 2_000_000_000,
                   'phase':{1:'initialization',2:'decision',3:'destruction'}.get(stage,'worker'),
                   'exitcode':self.process.exitcode}
        error = native.PolicyError(f'C++ policy {details["phase"]} {details["kind"]}: '
                                   f'{elapsed/1e6:.3f} ms; match process exited',details=details)
        game = self.journal.recover(self.plan,self.config)
        try:
            return {'failure':str(error),'details':details,'snapshot':snapshot(game),
                    'result':failure(game,error),'blob':game.replay_blob()}
        finally:
            close_game(game)

    def call(self, request):
        try:
            self.connection.send(request)
            # Native create/act/destroy have their own signal watchdogs. This
            # outer limit also bounds static initialization or a broken host.
            limit = 10+self.config.max_ticks*len(self.plan['entrant_ids'])*.03 if request['op']=='run' else 10
            if not self.connection.poll(limit):
                self.process.kill()
                self.process.join()
                return self._failed()
            return self.connection.recv()
        except (EOFError,BrokenPipeError,ConnectionResetError):
            return self._failed()

    def run(self, store, plan):
        response = self.begin(store,plan,run=True)
        return response['result'], response['blob']

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                try:
                    self.connection.send({'op':'shutdown'})
                    self.process.join(timeout=3)
                except (EOFError,BrokenPipeError):
                    pass
                if self.process.is_alive():
                    self.process.kill()
            self.process.join()
            self.connection.close()
            self.process = None
        if self.journal is not None:
            path = self.journal.path
            self.journal.close()
            path.unlink(missing_ok=True)
            self.journal = None


class HumanCppGame:
    """Existing LiveArena API, with a single native match process behind it."""
    def __init__(self, store, plan):
        self.plan, self.config = plan, store.config
        self.worker, self.engine = MatchWorker(), self
        try:
            self._receive(self.worker.begin(store,plan))
        except Exception:
            self.worker.close()
            raise

    def _receive(self, response):
        value = response['snapshot']
        self.state = state_from_dict(value['state'])
        self.initial_digest, self.done = value['initial_digest'], value['done']
        records = {m['seat']:m for m in value['timing']}
        self.controllers = [SimpleNamespace(decision_metrics=records[i]) if i in records else None
                            for i in range(len(self.plan['entrant_ids']))]
        self._result, self._blob = response.get('result'), response.get('blob')
        if 'failure' in response:
            error = native.PolicyError(response['failure'],details=response['details'])
            error.partial = (response['result'],response['blob'])
            raise error

    def advance(self, actions=None):
        self._receive(self.worker.call({'op':'advance',
            'actions':{k:v.encode() for k,v in (actions or {}).items()}}))

    def result(self):
        return self._result

    def replay_blob(self):
        if self._blob is None:
            self._receive(self.worker.call({'op':'replay'}))
        return self._blob

    def close(self):
        self.worker.close()
