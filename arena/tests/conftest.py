import pytest


@pytest.fixture
def in_process_batch(monkeypatch):
    """Scheduling unit tests can instrument the same native executor in-process.

    Real subprocess, watchdog, restore and web coverage remains in integration tests.
    """
    from generals_arena.matches.batch_worker import BatchWorker
    original_close = BatchWorker.close

    def run(self, entries, plans, **options):
        from generals_arena.matches.batch import execute_batch
        if not hasattr(self, '_test_cache'):
            self._test_cache = {}
        return execute_batch(entries, plans, runtime_cache=self._test_cache, **options)

    def close(self):
        original_close(self)
        for controller in getattr(self, '_test_cache', {}).values():
            method = getattr(controller, 'close', None)
            if method is not None:
                method()
        self._test_cache = {}

    monkeypatch.setattr(BatchWorker, 'run', run)
    monkeypatch.setattr(BatchWorker, 'close', close)
