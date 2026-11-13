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
import threading
from time import monotonic
from functools import wraps
from types import SimpleNamespace

import numpy as np
from generals import Action

from generals_arena.policies import native
from generals_arena.matches.game import RecordedGame
from generals_arena.matches.game import close_game
from generals_arena.matches.game import timing
from generals_arena.matches.game import failure
from generals_arena.policies.native import decision_timeout


_process_lifecycle = threading.RLock()


def lifecycle_locked(function):
    @wraps(function)
    def call(*args, **kwargs):
        # multiprocessing.start() polls/reaps the global child set. Serialize
        # it with join()/is_alive() so coordinator threads cannot race waitpid.
        with _process_lifecycle:
            return function(*args, **kwargs)
    return call


class ForfeitedController:
    """Retain diagnostics without loading or calling a surrendered policy again."""
    inactive = True

    def __init__(self, values):
        self.values = values

    def reset(self, **kwargs):
        pass

    def act(self, observation):
        return Action.wait()

    def on_result(self, move):
        pass

    @property
    def decision_metrics(self):
        return native.metrics(self.values)


class Journal:
    def __init__(self, path, ticks, players, create=False):
        self.path, self.players = Path(path), players
        size = (4+2*8*8+ticks*players)*8
        self.file = self.path.open('w+b' if create else 'r+b')
        if create:
            self.file.truncate(size)
        self.memory = mmap.mmap(self.file.fileno(), size)
        self.header = np.ndarray((4,), np.int64, self.memory)
        self.active_lane = np.zeros(1, np.int64)
        self.records = np.ndarray((8,8), np.int64, self.memory, offset=32)
        self.committed = np.ndarray((8,8), np.int64, self.memory, offset=(4+64)*8)
        self.actions = np.ndarray((ticks,players), np.int64, self.memory, offset=(4+128)*8)
        if create:
            self.header[:] = [0,-1,0,0]
            self.records[:] = 0
            self.committed[:] = 0

    @property
    def journals(self):
        return [self]

    def select(self, lane):
        if lane != 0:
            raise ValueError('single-match journal has one lane')

    def native_view(self):
        return SimpleNamespace(headers=self.header[None], all_records=self.records[None],
            committed=self.committed[None], actions=self.actions[:,None], active_lane=self.active_lane)

    def recover(self, plan, config, forfeits=()):
        game = RecordedGame(plan, config, [None]*self.players, forfeits, journal=self.native_view())
        game.restore(self.actions[:int(self.header[0])])
        return game

    def close(self):
        self.header = self.records = self.committed = self.actions = None
        self.memory.close()
        self.file.close()


def snapshot(game, seats=()):
    return {'summary': game.summary(), 'initial_digest': game.initial_digest,
            'done': game.done, 'timing': timing(game), 'forfeits': game.forfeits,
            'views': {seat: game.view(seat) for seat in seats}}


def child_loop(connection):
    from collections import OrderedDict
    resources = OrderedDict()
    # Policy printing must not corrupt the CLI's JSON output.
    with open(os.devnull, 'w') as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    from generals_arena.matches.runner import build_controllers
    game, journal, controllers = None, None, []
    try:
        while True:
            request = connection.recv()
            op = request['op']
            if op in ('shutdown','start','run','resume'):
                if game is not None:
                    close_game(game)
                    game = None
                if journal is not None:
                    native.DIAGNOSTICS = None
                    journal.close()
                    journal = None
            if op == 'shutdown':
                break
            try:
                if op in ('start','run','resume'):
                    config = SimpleNamespace(max_ticks=request['max_ticks'])
                    plan = request['plan']
                    journal = Journal(request['journal'], config.max_ticks, len(plan['entrant_ids']))
                    native.DIAGNOSTICS = journal
                    controllers = build_controllers(request["entries"], plan, resources=resources)
                    forfeits = request.get('forfeits', [])
                    saved = journal.records.copy()
                    for event in forfeits:
                        seat = event['seat']
                        controllers[seat] = ForfeitedController(journal.records[seat])
                    game = RecordedGame(plan, config, controllers, forfeits, journal=journal.native_view())
                    if op == 'resume':
                        journal.records[:] = saved
                        # Restart only after a hard timeout. Warm survivors with
                        # the original observations/results, using the journal's
                        # actions as authority even for nondeterministic policies.
                        game.restore(journal.actions[:int(journal.header[0])], warm=True)
                        op = request['continue_op']
                if op == 'run':
                    game.run()
                elif op == 'advance' and not game.done:
                    game.advance({int(k):Action.decode(v) for k,v in request['actions'].items()})
                elif op == 'surrender' and not game.done:
                    game.surrender(request['seat'])
                response = {'snapshot':snapshot(game, request.get('view_seats', ())) }
                if op in ('run','replay') or game.done:
                    response['blob'] = game.replay_blob()
                if game.done:
                    response['result'] = game.result()
                if op == 'run':
                    # Finish this game's teardown before the coordinator moves
                    # to a new journal. A broken destructor belongs to this game.
                    close_game(game)
                    game = None
                connection.send(response)
            except Exception as exc:
                details = getattr(exc, 'details', {})
                if decision_timeout(details):
                    connection.send({'decision_timeout': details})
                    continue
                if game is None:
                    game = RecordedGame(plan, config, [None]*len(plan['entrant_ids']))
                    game.controllers = controllers
                connection.send({'failure':str(exc), 'details':getattr(exc,'details',{}),
                    'snapshot':snapshot(game, request.get('view_seats', ())), 'result':failure(game,exc), 'blob':game.replay_blob()})
    except (EOFError, BrokenPipeError):
        pass
    finally:
        if game is not None:
            close_game(game)
        connection.close()


class MatchWorker:
    def __init__(self):
        self.process = self.connection = self.journal = None
        self._batch_worker = None
        self.stop = None

    def run_batch(self, entries, plans, **options):
        if self._batch_worker is None:
            from .batch_worker import BatchWorker
            self._batch_worker = BatchWorker()
        return self._batch_worker.run(entries, plans, **options)

    @lifecycle_locked
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

    def begin(self, entries, plan, *, max_ticks, run=False):
        self._start_process()
        if self.journal is not None:
            path = self.journal.path
            self.journal.close()
            path.unlink(missing_ok=True)
        self.plan, self.config = plan, SimpleNamespace(max_ticks=max_ticks)
        self.context, self.forfeits = {"entries": entries, "max_ticks": max_ticks,
            "view_seats": [seat for seat,key in enumerate(plan["entrant_ids"]) if entries[key]["kind"] == "human"]}, []
        # Process supervision is uniform; native per-decision watchdogs supply
        # exact deadlines. Python/NN adapters leave the decision header idle.
        self.bounded = False
        fd, path = tempfile.mkstemp(prefix='match-',dir=native._private_cache())
        os.close(fd)
        self.journal = Journal(path, max_ticks, len(plan['entrant_ids']), create=True)
        return self.call({'op':'run' if run else 'start', 'plan':plan,
            **self.context, 'journal':path})

    def _failed(self):
        with _process_lifecycle:
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
        if decision_timeout(details):
            return {'decision_timeout': details}
        error = native.PolicyError(f'C++ policy {details["phase"]} {details["kind"]}: '
                                   f'{elapsed/1e6:.3f} ms; match process exited',details=details)
        game = self.journal.recover(self.plan,self.config,self.forfeits)
        try:
            return {'failure':str(error),'details':details,'snapshot':snapshot(game, self.context.get('view_seats', ())),
                    'result':failure(game,error),'blob':game.replay_blob()}
        finally:
            close_game(game)

    def _exchange(self, request):
        try:
            self.connection.send(request)
            # C++ watchdogs remain active in mixed matches. Opaque Python/NN
            # policies have no decision deadline; they cannot inherit the
            # native-only whole-match budget (including load/warmup time).
            if self.bounded:
                limit = 10+self.config.max_ticks*len(self.plan['entrant_ids'])*.03 if request['op'] in ('run','resume') else 10
                ready = self.connection.poll(limit)
            else:
                # Also cover native library static initialization, before the
                # C ABI watchdog is armed. Python inference leaves seat=-1.
                phase, entered = None, monotonic()
                while not (ready := self.connection.poll(.1 if self.stop is not None else 1)):
                    if self.stop is not None and self.stop.is_set():
                        with _process_lifecycle:
                            self.process.kill()
                            self.process.join()
                        raise InterruptedError('match cancelled by scheduler')
                    current = tuple(self.journal.header[1:4])
                    if current != phase:
                        phase, entered = current, monotonic()
                    elif current[0] >= 0 and monotonic() - entered >= 10:
                        break
            if not ready:
                with _process_lifecycle:
                    self.process.kill()
                    self.process.join()
                return self._failed()
            return self.connection.recv()
        except (EOFError,BrokenPipeError,ConnectionResetError):
            return self._failed()

    def call(self, request):
        request.setdefault('view_seats', self.context.get('view_seats', []))
        operation = request.get('continue_op', request['op'])
        human_actions = request.get('actions', {})
        while True:
            response = self._exchange(request)
            if 'decision_timeout' not in response:
                self.forfeits = response['snapshot']['forfeits']
                return response
            details = response['decision_timeout']
            seat = details['player']
            if not 0 <= seat < len(self.plan['entrant_ids']) or any(e['seat'] == seat for e in self.forfeits):
                raise RuntimeError(f'invalid repeated decision timeout: {details}; forfeits={self.forfeits}')
            self.forfeits.append({**details, 'seat':seat, 'decision_tick':details['tick'],
                                  'tick':int(self.journal.header[0]), 'reason':'decision_timeout'})
            inactive = {event['seat'] for event in self.forfeits}
            for other in range(len(self.plan['entrant_ids'])):
                if other not in inactive:
                    self.journal.records[other] = self.journal.committed[other]
            # Never unwind or destroy a timed-out C++ instance in-process.
            # OS cleanup also handles policies that never return. No threads.
            with _process_lifecycle:
                if self.process.is_alive():
                    self.process.kill()
                self.process.join()
                self.journal.header[1:4] = [-1, 0, 0]
                self._start_process()
            request = {'op':'resume', 'continue_op':operation, 'actions':human_actions,
                       **self.context, 'plan':self.plan,
                       'journal':str(self.journal.path), 'forfeits':self.forfeits}

    def run(self, entries, plan, *, max_ticks, stop=None):
        self.stop = stop
        try:
            if stop is not None and stop.is_set():
                raise InterruptedError('match cancelled by scheduler')
            response = self.begin(entries,plan,max_ticks=max_ticks,run=True)
            return response['result'], response['blob']
        finally:
            self.stop = None

    @lifecycle_locked
    def close(self):
        if self._batch_worker is not None:
            self._batch_worker.close()
            self._batch_worker = None
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


class HumanGame:
    """Incremental native match behind the same public raw observation/action boundary."""
    def __init__(self, entries, plan, *, max_ticks):
        self.plan, self.config = plan, SimpleNamespace(max_ticks=max_ticks)
        self.worker, self.engine = MatchWorker(), self
        try:
            self._receive(self.worker.begin(entries,plan,max_ticks=max_ticks))
        except Exception:
            self.worker.close()
            raise

    def _receive(self, response):
        value = response['snapshot']
        self.forfeits = value.get('forfeits', [])
        self._summary, self._views = value['summary'], value['views']
        self.tick = self._summary['ticks']
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

    def surrender(self, seat):
        self._receive(self.worker.call({'op': 'surrender', 'seat': seat}))

    def view(self, seat=None):
        return self._views[-1 if seat is None else seat]

    def decision_timing(self):
        return self._summary.get('decision_timing', [])

    def summary(self):
        return self._summary

    def result(self):
        return self._result

    def replay_blob(self):
        if self._blob is None:
            self._receive(self.worker.call({'op':'replay'}))
        return self._blob

    def close(self):
        self.worker.close()


HumanCppGame = HumanGame  # Public import compatibility; every policy uses HumanGame.
