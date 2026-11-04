"""Reuse must not hide byte changes, replaced files or transaction failures."""
import os
from hashlib import sha256
import pytest
from generals_arena import util
from generals_arena.competitions.config import LeagueConfig
from generals_arena.competitions.store import CompetitionStore


def test_file_hash_reuse_detects_same_size_restored_mtime_and_replacement(tmp_path,monkeypatch):
    path=tmp_path/'weights.pt';path.write_bytes(b'first')
    original=util.sha256_file
    reads=[]
    def read(stream):
        reads.append(stream.name)
        return original(stream)
    monkeypatch.setattr(util,'sha256_file',read)
    before=path.stat()
    assert util.file_digest(path)==util.file_digest(path)==sha256(b'first').hexdigest()
    assert len(reads)==1
    path.write_bytes(b'other');os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
    assert util.file_digest(path)==sha256(b'other').hexdigest() and len(reads)==2
    other=tmp_path/'replace';other.write_bytes(b'third');other.replace(path)
    assert util.file_digest(path)==sha256(b'third').hexdigest() and len(reads)==3
    path.unlink()
    with pytest.raises(FileNotFoundError):util.file_digest(path)


def test_persistent_connection_commits_each_transaction_and_rolls_back_failures(tmp_path):
    store=CompetitionStore.create(tmp_path/'league',LeagueConfig(player_counts=(2,)),reference='random')
    with store.session():
        with store.connect(write=True) as first:
            first.execute("INSERT INTO meta VALUES ('probe','1')")
        with CompetitionStore(store.directory).connect() as reader:
            assert reader.execute("SELECT value FROM meta WHERE key='probe'").fetchone()[0]=='1'
        with pytest.raises(RuntimeError):
            with store.connect(write=True) as second:
                assert second is first
                second.execute("UPDATE meta SET value='2' WHERE key='probe'")
                raise RuntimeError('abort only this transaction')
        with store.connect() as third:
            assert third is first
            assert third.execute("SELECT value FROM meta WHERE key='probe'").fetchone()[0]=='1'
