"""Serial native batches with a shared action journal; no per-tick IPC."""
import mmap
import multiprocessing
import os
from pathlib import Path
import signal
import tempfile
from time import monotonic
from types import SimpleNamespace

import numpy as np

from generals_arena.policies import native
from .worker import Journal, MatchWorker, _process_lifecycle
from .game import close_game, failure
from .resources import cpu_slot


class BatchJournal:
    def __init__(self, path, count, ticks, players, *, create=False):
        self.path, self.count = Path(path), count
        size = (1 + count * (4 + 128 + ticks * players)) * 8
        self.file = self.path.open('w+b' if create else 'r+b')
        if create:
            self.file.truncate(size)
        self.memory = mmap.mmap(self.file.fileno(), size)
        self.active_lane = np.ndarray((1,), np.int64, self.memory)
        self.headers = np.ndarray((count, 4), np.int64, self.memory, offset=8)
        offset = (1 + count * 4) * 8
        self.all_records = np.ndarray((count, 8, 8), np.int64, self.memory, offset=offset)
        self.committed = np.ndarray((count, 8, 8), np.int64, self.memory, offset=offset+count*64*8)
        self.actions = np.ndarray((ticks, count, players), np.int64, self.memory,
                                  offset=offset+count*128*8)
        self.journals = [SimpleNamespace(header=self.headers[i], records=self.all_records[i])
                         for i in range(count)]
        if create:
            self.active_lane[:] = -1
            self.headers[:] = [0, -1, 0, 0]
            self.all_records[:] = self.committed[:] = 0
        self.select(0)

    def select(self, lane):
        self.active_lane[0] = lane
        self.header, self.records = self.headers[lane], self.all_records[lane]

    def record(self, host, active):
        lanes = np.flatnonzero(active)
        ticks = host['tick'][lanes]
        self.actions[ticks-1, lanes] = host['_joint'][lanes]
        self.committed[lanes] = self.all_records[lanes]
        self.headers[lanes, 0] = ticks  # Publish complete joint actions last.

    def metrics(self, lane):
        return [{'seat': seat, **native.metrics(row)}
                for seat, row in enumerate(self.all_records[lane]) if row[0]]

    def extract(self, lane, players, ticks):
        fd, path = tempfile.mkstemp(prefix='match-', dir=native._private_cache())
        os.close(fd)
        journal = Journal(path, ticks, players, create=True)
        journal.header[:] = self.headers[lane]
        journal.records[:] = self.all_records[lane]
        journal.committed[:] = self.committed[lane]
        prefix = int(journal.header[0])
        journal.actions[:prefix] = self.actions[:prefix, lane]
        return journal

    def close(self):
        self.journals = self.header = self.records = None
        self.active_lane = self.headers = self.all_records = self.committed = self.actions = None
        self.memory.close()
        self.file.close()


def _child(connection):
    with open(os.devnull, 'w') as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    from .batch import execute_batch
    from .cache import RuntimeCache
    from generals import MapConfig
    from collections import OrderedDict
    runtimes, maps = RuntimeCache(), OrderedDict()
    try:
        while (request := connection.recv()) is not None:
            maps.update(request.pop('initial_states', {}))
            for plan in request['plans']:
                key = (MapConfig(**plan['map']), plan['map_seed'])
                if key in maps:
                    maps.move_to_end(key)
            journal = BatchJournal(request.pop('journal'), len(request['plans']),
                                   request['max_ticks'], len(request['plans'][0]['entrant_ids']))
            native.DIAGNOSTICS = journal
            try:
                records, timing = execute_batch(**request, journal=journal,
                    runtime_cache=runtimes, initial_states=maps)
                runtimes.trim()
                while len(maps) > 256:
                    maps.popitem(last=False)
                connection.send({'records': records, 'timing': timing, 'maps': list(maps)})
            except Exception as exc:
                details = getattr(exc, 'details', {})
                connection.send({'error': str(exc), 'details': details,
                                 'lane': details.get('lane', int(journal.active_lane[0]))})
            finally:
                native.DIAGNOSTICS = None
                journal.close()
    except (EOFError, BrokenPipeError):
        pass
    finally:
        runtimes.close()
        connection.close()


class BatchWorker:
    def __init__(self):
        self.process = self.connection = None
        self._sent_maps = set()

    def _start(self):
        with _process_lifecycle:
            if self.process is not None and self.process.is_alive():
                return
            self.close()
            context = multiprocessing.get_context('spawn')
            self.connection, child = context.Pipe()
            self.process = context.Process(target=_child, args=(child,), daemon=True)
            self.process.start()
            child.close()

    def close(self):
        self._sent_maps.clear()
        with _process_lifecycle:
            if self.process is not None:
                if self.process.is_alive():
                    self.process.kill()
                self.process.join()
                self.connection.close()
                self.process = self.connection = None

    def _exchange(self, request, journal):
        self._start()
        try:
            request = dict(request)
            request['initial_states'] = {key: state for key, state in request.get('initial_states', {}).items()
                                         if key not in self._sent_maps}
            self.connection.send(request)
            phase, entered = None, monotonic()
            while not self.connection.poll(1):
                lane = int(journal.active_lane[0])
                current = (lane, *journal.headers[lane, 1:]) if lane >= 0 else (-1, -1, 0, 0)
                if current != phase:
                    phase, entered = current, monotonic()
                elif current[1] >= 0 and monotonic()-entered > 10:
                    with _process_lifecycle:
                        self.process.kill()
                    break
            if self.connection.poll():
                response = self.connection.recv()
                self._sent_maps = set(response.pop('maps', ()))
                return response
        except (EOFError, BrokenPipeError, ConnectionResetError):
            pass
        with _process_lifecycle:
            self.process.join(timeout=2)
        lane = int(journal.active_lane[0])
        if not 0 <= lane < journal.count:
            raise RuntimeError('native batch worker exited outside a policy call')
        _, seat, tick, stage = map(int, journal.headers[lane])
        if seat < 0:
            raise RuntimeError('native batch worker exited outside a policy call')
        row = journal.all_records[lane, seat]
        timeout = row[7] == 1 or self.process.exitcode in (-signal.SIGALRM, 128+signal.SIGALRM)
        if not (row[4] or row[5]):
            row[4 if timeout else 5] += 1
        details = {'kind': 'timeout' if timeout else 'policy_error', 'player': seat, 'tick': tick,
                   'elapsed_ns': int(row[6]), 'limit_ns': native.DECISION_NS if stage == 2 else 2_000_000_000,
                   'phase': {1: 'initialization', 2: 'decision', 3: 'destruction'}.get(stage, 'worker'),
                   'exitcode': self.process.exitcode}
        return {'error': 'native batch policy exited', 'details': details, 'lane': lane}

    def run(self, entries, plans, **options):
        """A failed lane resumes from its journal; unpublished peers restart alone."""
        pending, output, timing = list(enumerate(plans)), [None] * len(plans), {}
        while pending:
            current = [plan for _, plan in pending]
            ticks, players = options['max_ticks'], len(current[0]['entrant_ids'])
            fd, path = tempfile.mkstemp(prefix='batch-', dir=native._private_cache())
            os.close(fd)
            journal = BatchJournal(path, len(current), ticks, players, create=True)
            try:
                response = self._exchange(dict(entries=entries, plans=current, journal=path, **options), journal)
                if 'records' in response:
                    for (index, _), record in zip(pending, response['records'], strict=True):
                        output[index] = record
                    for key, value in response['timing'].items():
                        timing[key] = timing.get(key, 0.) + value
                    break
                details, lane = response['details'], response['lane']
                if details.get('player') is None or not 0 <= lane < len(pending):
                    raise RuntimeError(response['error'])
                self.close()  # Never destroy a timed-out native instance in-process.
                index, plan = pending.pop(lane)
                recovered = journal.extract(lane, players, ticks)
                worker = MatchWorker()
                worker.journal, worker.plan = recovered, plan
                worker.config = SimpleNamespace(max_ticks=ticks)
                worker.context = {'entries': entries, 'max_ticks': ticks}
                worker.forfeits, worker.bounded = [], False
                try:
                    if native.decision_timeout(details):
                        seat = details['player']
                        worker.forfeits.append({**details, 'seat': seat, 'decision_tick': details['tick'],
                            'tick': int(recovered.header[0]), 'reason': 'decision_timeout'})
                        for other in range(players):
                            if other != seat:
                                recovered.records[other] = recovered.committed[other]
                        recovered.header[1:4] = [-1, 0, 0]
                        # The failed child released its lease on exit. Acquire
                        # another before starting scalar recovery; waiting here
                        # is never part of a policy's decision deadline.
                        with cpu_slot():
                            worker._start_process()
                            result = worker.call({'op': 'resume', 'continue_op': 'run', 'plan': plan,
                                **worker.context, 'journal': str(recovered.path), 'forfeits': worker.forfeits})
                        output[index] = result['result'], result['blob']
                    else:
                        game = recovered.recover(plan, worker.config)
                        try:
                            error = native.PolicyError(response['error'], details=details)
                            output[index] = failure(game, error), game.replay_blob()
                        finally:
                            close_game(game)
                    if not options.get('record_replays'):
                        output[index] = output[index][0], None
                finally:
                    worker.close()
            finally:
                journal.close()
                Path(path).unlink(missing_ok=True)
        return output, timing
