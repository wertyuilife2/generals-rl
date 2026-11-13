"""Shared mappings grow between batches and expose only committed action prefixes."""
import numpy as np

from generals_arena.matches.batch_worker import BatchJournal


def test_journal_remap_reset_and_committed_prefix(tmp_path):
    path = tmp_path / 'journal'
    parent = BatchJournal(path, 2, 5, 2, create=True)
    child = BatchJournal(path, 2, 5, 2)
    try:
        for count, ticks, players in ((2, 5, 2), (8, 30, 4), (1, 3, 2), (4, 20, 3)):
            parent.configure(count, ticks, players, grow=True)
            parent.reset()
            child.configure(count, ticks, players)
            assert not child.headers[:, 0].any() and not child.all_records.any()
            assert not child.committed.any()
            child.actions[:] = 999  # Stale, uncommitted contents from a previous batch.
            child.actions[0, 0] = np.arange(players)
            child.headers[0, 0] = 1
            child.all_records[0, 0, 0] = 10
            child.committed[0, 0, 0] = 9
            recovered = parent.extract(0, players, ticks)
            try:
                assert recovered.header[0] == 1
                assert recovered.records[0, 0] == 10 and recovered.committed[0, 0] == 9
                assert list(recovered.actions[0]) == list(range(players))
                assert not recovered.actions[1:].any()
            finally:
                recovered.close()
                recovered.path.unlink()
        mapping = parent.memory
        parent.configure(count, ticks, players, grow=True)
        parent.reset()
        assert parent.memory is mapping
        assert not parent.headers[:, 0].any()
    finally:
        child.close()
        parent.close()
